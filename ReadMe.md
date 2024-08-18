# Environment
```
conda create --name EmojiAttack --file requirements.txt
```

# Dataset and Model
## Our harmful dataset
We sample examples from 
* 574 -- [advbench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)
* 110 -- [red-teams-attempts](https://github.com/anthropics/hh-rlhf/tree/master/red-team-attempts)
* 748 -- [self-defense](https://github.com/poloclub/llm-self-defense). 
We already save these datasets in [dataset](./dataset/). 

## Judge LLMs
* [Llama Guard](https://huggingface.co/meta-llama/LlamaGuard-7b) $\rightarrow$ `Llama_Guard_Model_PATH`
* [Llama Guard 2](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B) $\rightarrow$ `Llama_Guard_2_Model_PATH`
* [ShieldLM](https://huggingface.co/thu-coai/ShieldLM-7B-internlm2) $\rightarrow$ `ShieldLM_Model_PATH`
* [WildGuard](https://huggingface.co/allenai/wildguard) $\rightarrow$ `WildGuard_Model_PATH`, `WildGuard_huggingface_token`
* GPT-3.5, GPT-4 $\rightarrow$ `OPENAI_API_KEY`
## The lightweight surrogate model
* [gtr-t5-xl](https://huggingface.co/sentence-transformers/gtr-t5-xl) $\rightarrow$ `Surrogate_Model_PATH`

Modify the [paths](./config.py) to your own paths.

# Experiments
## Emoji Attack `Table 1` `Figure 5` `Figure 6`
* Experiments of token segmentation bias
```python
python white_box.py --attack space
python white_box.py --attack emoji
```
* Experiments of ablation sudy
```python
python number_of_emojis.py --nums [nums]
python other_delimiters.py --delimiter [delimiter]
```
`[nums]` is the number of inserted emojis.   
`[delimiter]` is the used delimiter, such as `!`, `@`, etc.

## Practical Emoji Attack `Table 2`
We download existing jailbreaking prompts from [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak). EasyJailbreak shares its experimental results by [link](https://drive.google.com/file/d/1Im3q9n6ThL4xiaUEBmD7M8rkOIjw8oWU/view?usp=sharing).
We save these jailbreaking prompts in [Easyjailbreaking-Results](./EasyJailbreaking-Results/)
* Filter the successful jailbreaking prompts reported from EasyJailbreak, and then obtain their corresponding outputs from "gpt-3.5-turbo".
```python
python default_responses.py
```
* Obtain responses with the emoji instruction
```python
python emoji_responses.py
```
* Evaluate these responses
```python
python default_evaluate.py
python emoji_evaluate.py
```

## Visualization of Cross-Attention Valuse `Figure 2`
Generate cross-attention values from these two examples, which saved into the current path.
```python
python bias_attention.py
```

