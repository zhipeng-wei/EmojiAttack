import argparse
import os
import glob
import json
import numpy as np
import pickle as pkl
from judge.ShieldLM import ShieldLLM_reference
from judge.WildGuard import WildGurad_reference
from judge.LlamaGuard import LlamaGuard_reference
from judge.LlamaGuard2 import LlamaGuard2_reference
from judge.gpt35 import gpt35_reference
from judge.gpt4 import gpt4_reference
from judge.match import match_reference
import emoji

from config import SAVE_PATH

if __name__ == '__main__':
    output_base_name = SAVE_PATH
    normal_output_folder = 'BlackAttack_gpt-3.5-turbo'
    used_path = os.path.join(output_base_name, normal_output_folder)

    emoji_path = os.path.join(used_path, 'EmojiResponse.pkl')
    with open(emoji_path, 'rb') as ipt:
        info = pkl.load(ipt)
        
    all_used_ids = [i['used_idx'] for i in info]
    all_attack_name = [i['attack'] for i in info]
    all_attack_prompts = [i['used_prompt'] for i in info]
    all_responses = [i['emoji_response'] for i in info]

    plain_responses = [emoji.replace_emoji(i, replace='') for i in all_responses]
    
    match_pres = match_reference(plain_responses)
    match_pres = np.array(match_pres)
    used_idx = np.where(match_pres == 1)[0]

    print (f'After filting by matching, the number of examples is changed to {len(used_idx)} from {len(plain_responses)}.')

    all_used_ids = [all_used_ids[idx] for idx in used_idx]
    all_attack_name = [all_attack_name[idx] for idx in used_idx]
    all_attack_prompts = [all_attack_prompts[idx] for idx in used_idx]
    all_responses = [all_responses[idx] for idx in used_idx]
    
    save_dict = []
    for t_i, idx in enumerate(all_used_ids):
        save_dict.append(
            {
                'used_idx': idx,
                'attack': all_attack_name[t_i],
                'used_prompt': all_attack_prompts[t_i],
                'response': all_responses[t_i]
            }
        )
    for func in [ShieldLLM_reference, WildGurad_reference, LlamaGuard_reference, LlamaGuard2_reference, gpt35_reference, gpt4_reference]:
        pres, conts = func(all_responses, batch_size=1)
        
        for t_i,idx in enumerate(all_used_ids):
            save_dict[t_i][f'{func.__name__}-Pre'] = pres[t_i]
            save_dict[t_i][f'{func.__name__}-Detail'] = conts[t_i]
    
    output_name = os.path.join(used_path, 'Emoji_evaluate.pkl')
    with open(output_name, 'wb') as opt:
        pkl.dump(save_dict, opt)