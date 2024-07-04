import subprocess
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch.nn.functional as F
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)

class script_args:
    base_dir = "/data02/wenhao/jl/ckpt/rm"
    pre_train = "rm-lmsys-FsfairX_epoch4"
    test_file = "/data02/wenhao/jl/datasets/lmsys_all.csv"
    save_path = "scripts/results"
    batch_size = 1
    max_length = 1024
    max_total_length = 1024
    margin = 0.1
    flash_attn = False
    bf16 = True
    disable_fast_tokenizer = False
    device = "cuda:0"


def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    normalize_reward=False,
    use_flash_attention_2=False,
    head_prefix="value_head",
    device_map=None,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        normalize_reward (bool, optional): Whether normalize reward. Defaults to False.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto
            multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.

    Returns:
        nn.Module: pretrained transformer model.
    """
    assert (
         model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=None,
        device_map=device_map,
        **kwargs,
    )

    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, head_prefix="value_head"):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.head_prefix = head_prefix
            setattr(self, head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.head_prefix)(last_hidden_states).squeeze(-1)

            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # normalize reward in eval mode
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            if return_output:
                return reward, outputs
            else:
                return reward

    return RewardModel

def zero_pad_sequences(sequences, max_len, side: str = "left",  value=0):
    assert side in ("left", "right")
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer

args = script_args()
device = torch.device(args.device)
pretrain_path = os.path.join(args.base_dir, args.pre_train)
logging.info(f"pretrain_path: {pretrain_path}")
model = get_llm_for_sequence_regression(
        pretrain_path,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        init_value_head=True,
    )

tokenizer = get_tokenizer(pretrain_path, model, "left", use_fast=not args.disable_fast_tokenizer)
model.to(args.device)
model.eval()
template = "Human: {}\nAssistant: {}"


####################################################################################################
results = []
max_memory_observed = 0
df = pd.read_csv(args.test_file)
df['prompt'] = df['prompt'].replace('null', "'null'")
df['response_a'] = df['response_a'].replace('null', "'null'")
df['response_b'] = df['response_b'].replace('null', "'null'")
eval_dataset = Dataset.from_pandas(df)
length =  len(eval_dataset)
eval_length = int(length*0.1)
acc = 0
tie = 0
for i in tqdm(range(length-eval_length, length)):
    data = eval_dataset[i]
    idx = data["id"]
    resp_a = template.format(data['prompt'], data['response_a'])
    resp_b = template.format(data['prompt'], data['response_b'])

    if not resp_a.endswith(tokenizer.eos_token):
            resp_a += " " + tokenizer.eos_token
    if not resp_b.endswith(tokenizer.eos_token):
            resp_b += " " + tokenizer.eos_token

    resp_a_tokens = tokenizer(
        resp_a,
        max_length=args.max_total_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    resp_b_tokens = tokenizer(
        resp_b,
        max_length=args.max_total_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    resp_a_ids = resp_a_tokens['input_ids']
    resp_a_mask = resp_a_tokens['attention_mask']
    resp_b_ids = resp_b_tokens['input_ids']
    resp_b_mask = resp_b_tokens['attention_mask']

    resp_a_ids[0][-1] = tokenizer.eos_token_id
    resp_b_ids[0][-1] = tokenizer.eos_token_id
    resp_a_mask[0][-1] = True
    resp_b_mask[0][-1] = True

    max_len = max(resp_a_ids.size(-1), resp_b_ids.size(-1))
    resp_a_ids = zero_pad_sequences(resp_a_ids, max_len=max_len, value=tokenizer.pad_token_id)
    resp_b_ids = zero_pad_sequences(resp_b_ids, max_len=max_len, value=tokenizer.pad_token_id)
    resp_a_mask = zero_pad_sequences(resp_a_mask, max_len=max_len)
    resp_b_mask = zero_pad_sequences(resp_b_mask, max_len=max_len)

    resp_ids = torch.cat([resp_a_ids, resp_b_ids], dim=0)
    resp_mask = torch.cat([resp_a_mask, resp_b_mask], dim=0)
    
    with torch.no_grad():
        with autocast():
            torch.cuda.empty_cache()
            resp_ids = resp_ids.to(device)
            resp_mask = resp_mask.to(device)
            # rewards = model(resp_ids, resp_mask).cpu().numpy()
            # resp_a_reward, resp_b_reward = rewards[0], rewards[1]
            resp_a_reward = model(resp_a_ids.to(device), resp_a_mask.to(device)).cpu().item()
            resp_b_reward = model(resp_b_ids.to(device), resp_b_mask.to(device)).cpu().item()
            resp_a_reward = round(resp_a_reward, 3)
            resp_b_reward = round(resp_b_reward, 3)
            # del resp_ids, resp_mask
            logging.info(f"resp_a_reward: {resp_a_reward}, resp_b_reward: {resp_b_reward}")

    result = {
        "id": idx, 
        "resp_a_reward": resp_a_reward,
        "resp_b_reward": resp_b_reward,
        "winner_model_a": data['winner_model_a'],
        "winner_model_b": data['winner_model_b'],
        "winner_tie": data['winner_tie']
    }
    results.append(result)
record = {
    "model": pretrain_path,
    "data": args.test_file,
    "margin": args.margin,
    "length": eval_length,
    "max_memoery": max_memory_observed,
    "acc": acc/eval_length,
}
jsonl_file_path = f"{args.save_path}/results.jsonl"


if not os.path.exists(jsonl_file_path):
    with open(jsonl_file_path, 'w') as jsonl_file:
        pass  

with open(jsonl_file_path, 'a') as jsonl_file: 
    jsonl_file.write(json.dumps(record) + '\n')

df = pd.DataFrame(results)
df.to_csv(f"{args.save_path}/reward_{args.pre_train}_eval.csv", index=False)
print(f"Results saved to {jsonl_file_path}")
