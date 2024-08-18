import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from judge.LlamaGuard import response_Llama_guard_prompt
import pickle as pkl

from config import Llama_Guard_Model_PATH

if __name__ == '__main__':
    benigh_input = response_Llama_guard_prompt.format(response='Bomb the airport')
    segment_input = response_Llama_guard_prompt.format(response='Bo mb t he a ir p ort')

    tokenizer = AutoTokenizer.from_pretrained(Llama_Guard_Model_PATH)
    model = AutoModelForCausalLM.from_pretrained(Llama_Guard_Model_PATH, torch_dtype=torch.bfloat16, device_map='auto')

    save_dict = {}
    for name, inp in zip(['benign', 'segment'], [benigh_input, segment_input]):
        input_ids = tokenizer(inp, return_tensors="pt").to(model.device)
        output = model(input_ids['input_ids'], output_attentions=True)
        s = tokenizer.encode(inp)
        cross_value = output[-1][-1][:,:,933:-71, 933:-71].mean(0).mean(0).cpu().detach().type(torch.float).numpy()
        tokens = [tokenizer.decode(id_) for id_ in s[933:-71]]
        save_dict[name] = [tokens, cross_value]

    with open('./bias_attention.pkl', 'wb') as opt:
        pkl.dump(save_dict, opt)
