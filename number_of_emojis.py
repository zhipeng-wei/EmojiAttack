import random
import torch
import os
import argparse
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from our_datasets import HarmfulResponseDataset
from judge.ShieldLM import ShieldLLM_reference
from judge.WildGuard import WildGurad_reference
from judge.LlamaGuard import LlamaGuard_reference
from judge.LlamaGuard2 import LlamaGuard2_reference
from judge.gpt35 import gpt35_reference

from config import SAVE_PATH, Surrogate_Model_PATH, Llama_Guard_Model_PATH

def InsertDelimiterRandomly(tokenizer, sentence, delimiter='😊', nums=1):
    token_ids = tokenizer.encode(sentence)
    if token_ids[0] == 1:
        token_ids = token_ids[1:]

    if nums <= len(token_ids):
        random_tok_ids = random.sample(range(len(token_ids)), nums)
    else:
        random_tok_ids = random.sample(range(len(token_ids)), len(token_ids))

    modified_tokens = []
    for i,one_tok_id in enumerate(token_ids):
        if i in random_tok_ids:
            ori_token_text = tokenizer.decode(one_tok_id)
            if len(ori_token_text) <= 1:
                modified_tokens += [one_tok_id]
                continue
            random.seed(i)
            random_pos = random.randint(1, len(ori_token_text)-1)
            injected_text = ori_token_text[:random_pos] + delimiter + ori_token_text[random_pos:]
            injected_token = tokenizer.encode(injected_text) 
            if (injected_token[0] == 1) & (injected_token[1] == 29871):
                injected_token = injected_token[2:]
            else:
                if injected_token[0] == 1:
                    injected_token = injected_token[1:]
            modified_tokens += injected_token
        else:
            modified_tokens += [one_tok_id]

    # this operation may introduce extra space within one word
    # such as when replacing 'erve' by 'er😊ve', the sentence changes the word "deserve" to "de ser😊ve".
    injected_sentences = tokenizer.decode(modified_tokens) 
    return injected_sentences


def calculate_similarity(emb_model, original_word, splits):
    ori_embedding = emb_model.encode([original_word])
    split_embedding = emb_model.encode(splits)
    cs_output = torch.nn.functional.cosine_similarity(torch.tensor(split_embedding), torch.tensor(ori_embedding))
    order_token_inds = torch.argsort(cs_output, descending=False)
    return order_token_inds[0], cs_output[order_token_inds[0]]

def split_one_word(word, emb_model):
    length = len(word)
    splits = [word[:i+1] + ' ' + word[i+1:] for i in range(length-1)]
    pos, val = calculate_similarity(emb_model, word, splits)
    return pos, val

def InsertDelimiterDiff(saved_dict, emb_model, tokenizer, sentence, delimiter='😊', nums=1):
    token_ids = tokenizer.encode(sentence)
    if token_ids[0] == 1:
        token_ids = token_ids[1:]

    this_saved_dict = {}
    for idx, token_id in enumerate(token_ids):
        one_subword = tokenizer.decode(token_id)
        if len(one_subword) > 1:
            if one_subword in saved_dict.keys():
                pos, val = saved_dict[one_subword]
            else:
                pos, val = split_one_word(one_subword, emb_model)
                saved_dict[one_subword] = (pos, val)
            this_saved_dict[idx] = (pos, val)
        else:
            continue
    sorted_saved_dict = sorted(this_saved_dict.items(), key=lambda item: item[1][1], reverse=False)
    inserted_emojis_nums = min(len(this_saved_dict), nums)
    inserted_ids = [sorted_saved_dict[i][0] for i in range(inserted_emojis_nums)]

    modified_tokens = []
    for i,one_tok_id in enumerate(token_ids):
        if i not in inserted_ids:
            modified_tokens += [one_tok_id]
        else:
            ori_token_text = tokenizer.decode(one_tok_id)
            if len(ori_token_text) <= 1:
                modified_tokens += [one_tok_id]
                continue
            pos, _ = this_saved_dict[i]
            injected_text = ori_token_text[:pos+1] + delimiter + ori_token_text[pos+1:]
            injected_token = tokenizer.encode(injected_text) 
            if (injected_token[0] == 1) & (injected_token[1] == 29871):
                injected_token = injected_token[2:]
            else:
                if injected_token[0] == 1:
                    injected_token = injected_token[1:]
            modified_tokens += injected_token

    # this operation may introduce extra space within one word
    # such as when replacing 'erve' by 'er😊ve', the sentence changes the word "deserve" to "de ser😊ve".
    injected_sentences = tokenizer.decode(modified_tokens) 
    return saved_dict, injected_sentences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type = str,
        default = '0,1'
        )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "bias_exp"
        )

    parser.add_argument(
        "--nums",
        type = int,
        default = 1
        )

    args = parser.parse_args()
    args.saved_base_path = SAVE_PATH
    args.saved_path = os.path.join(args.saved_base_path, args.output_dir)

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.device_count.cache_clear()

    used_dataset = HarmfulResponseDataset()
    tokenizer = AutoTokenizer.from_pretrained(Llama_Guard_Model_PATH)
    benign_responses = [used_dataset[idx][0] for idx in range(len(used_dataset))]

    saved_dict = {}
    emb_model = SentenceTransformer('sentence-transformers/gtr-t5-xl', cache_folder=Surrogate_Model_PATH)
    
    random_responses = []
    cs_responses = []
    for benign_response in benign_responses:
        rand_response = InsertDelimiterRandomly(tokenizer, benign_response, delimiter='😊', nums=args.nums)
        saved_dict, cs_response = InsertDelimiterDiff(saved_dict, emb_model, tokenizer, benign_response, delimiter='😊', nums=args.nums)
        random_responses += [rand_response]
        cs_responses += [cs_response]
        # print (f'**************\n{benign_response}\n{rand_response}\n{cs_response}')

    save_dict = {}
    save_dict['random_response'] = random_responses
    save_dict['cs_response'] = cs_responses
    for func in [LlamaGuard_reference, LlamaGuard2_reference]:
        pres, conts = func(random_responses, batch_size=1)
        save_dict[f'{func.__name__}-random'] = [pres, conts]
        pres, conts = func(cs_responses, batch_size=1)
        save_dict[f'{func.__name__}-cs'] = [pres, conts]

    output_name = os.path.join(args.saved_path, 'Wb-random-cs-emoji-{}.pkl'.format(args.nums))
    with open(output_name, 'wb') as opt:
        pkl.dump(save_dict, opt)
        
        