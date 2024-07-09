import subprocess
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import datasets
import json
from typing import Optional
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
# from datasets import load_dataset, Dataset
from torch.utils.data import Dataset
import logging

from torch.cuda.amp import autocast
from threading import Thread
import ast

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def process_multi_turn_dialogue(
    conversations, input_template="Human: {}\nAssistant: ", content_key="content", role_key="role"
):
    result = []
    if type(conversations) == str and '\n' in conversations:
        conversations = conversations.replace('\n', ',')
        conversations = ast.literal_eval(conversations)
    for l in conversations:
        if "user" in l[role_key] or "human" in l[role_key]:
            result.append(input_template.format(l[content_key]))
        else:
            result.append(l[content_key] + "\n")
    return "".join(result)

def exist_and_not_none(d, key):
    return key in d and d[key] is not None



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)

class script_args:
    pretrain = "/data02/wenhao/jl/ckpt/rm/rm-lmsys-FsfairX_epoch4"
    test_file = "'/data02/wenhao/jl/datasets/lmsys_chatbot_arena_conversations.csv'"
    save_path = "scripts/results"
    batch_size = 2
    max_length = 400
    test_sample = 57000
    lora_rank=16
    init_value_head = False
    flash_attn = False
    bf16 = True
    load_in_4bit = False
    disable_fast_tokenizer = False
    device = "cuda:0"

def init_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger

# Initialize logger
logger = init_logger(__name__)

def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=16,
    lora_alpha=8,
    target_modules=['o_proj', 'v_proj'],
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=True,
    ds_config: dict = None,
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
    
    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        bnb_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=False,
           bnb_4bit_compute_dtype=torch.bfloat16
        )
#         bnb_config =  BitsAndBytesConfig(
#             load_in_8bit=True,
#             bnb_8bit_compute_dtype=torch.float16,
#             bnb_8bit_use_double_quant=False,
#         )
    else:
        bnb_config = None


    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=bnb_config,
        device_map=device_map,
        **kwargs,
    )
    
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

def zero_pad_sequences(sequences, max_len=None, side: str = "left", value=0):
    assert side in ("left", "right")
    if max_len is None:
        max_len = max(seq.size(-1) for seq in sequences)
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
####################################################################################################
def zero_pad_sequences(sequences, max_len=None, side: str = "left", value=0):
    assert side in ("left", "right")
    if max_len is None:
        max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

class RewardDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, template="Human: {}\nAssistant: {}"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        
        self.resp_as = []
        self.resp_bs = []
        self.prompts = []
        self.result_ids = []
        self.winners = []
        for data in tqdm(dataset):
            self.prompts.append(data['prompt'])
            self.resp_as.append(data['response_a'])
            self.resp_bs.append(data['response_b'])
            self.result_ids.append(data['id'])
            if data['winner_model_a'] == 1:
                self.winners.append(0)
            elif data['winner_model_b'] == 1:
                self.winners.append(1)
            else:
                self.winners.append(2)
    def __len__(self):
        return len(self.result_ids)

    def __getitem__(self, idx):
        result_id = self.result_ids[idx]
        prompt = self.prompts[idx]
        resp_a = self.resp_as[idx]
        resp_b = self.resp_bs[idx]
        winner = self.winners[idx]

        # Encode prompt, resp_a, and resp_b separately with max_length / 2
        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        resp_a_encoded = self.tokenizer(
            resp_a,
            max_length=self.max_length // 2,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        resp_b_encoded = self.tokenizer(
            resp_b,
            max_length=self.max_length // 2,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # Decode the encoded inputs
        prompt_decoded = self.tokenizer.decode(prompt_encoded['input_ids'][0], skip_special_tokens=True)
        resp_a_decoded = self.tokenizer.decode(resp_a_encoded['input_ids'][0], skip_special_tokens=True)
        resp_b_decoded = self.tokenizer.decode(resp_b_encoded['input_ids'][0], skip_special_tokens=True)
        
        resp_a = self.template.format(prompt_decoded, resp_a_decoded)
        resp_b = self.template.format(prompt_decoded, resp_b_decoded)
        if not resp_a.endswith(self.tokenizer.eos_token):
            resp_a += " " + self.tokenizer.eos_token
        if not resp_b.endswith(self.tokenizer.eos_token):
            resp_b += " " + self.tokenizer.eos_token
        
        resp_a_input = self.tokenizer(
            resp_a,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        
        resp_b_input = self.tokenizer(
            resp_b,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        
        resp_a_input["input_ids"][0][-1] = self.tokenizer.eos_token_id
        resp_b_input["input_ids"][0][-1] = self.tokenizer.eos_token_id
        resp_a_input["attention_mask"][0][-1] = True
        resp_b_input["attention_mask"][0][-1] = True

        return (
            resp_a_input['input_ids'],  
            resp_a_input['attention_mask'],
            resp_b_input['input_ids'],
            resp_b_input['attention_mask'],
            result_id,
            winner
        )
    def collate_fn(self, item_list):
        resp_a_ids = []
        resp_a_masks = []
        resp_b_ids = []
        resp_b_masks = []
        result_ids = []
        winners = []

        for resp_a_id, resp_a_mask, resp_b_id, resp_b_mask, result_id, winner in item_list:
            assert resp_a_id.shape == resp_a_mask.shape
            assert resp_b_id.shape == resp_b_mask.shape
            resp_a_ids.append(resp_a_id)
            resp_a_masks.append(resp_a_mask)
            resp_b_ids.append(resp_b_id)
            resp_b_masks.append(resp_b_mask)
            result_ids.append(result_id)
            winners.append(winner)

        resp_a_ids = zero_pad_sequences(resp_a_ids, value=self.tokenizer.pad_token_id)
        resp_a_masks = zero_pad_sequences(resp_a_masks)
        resp_b_ids = zero_pad_sequences(resp_b_ids, value=self.tokenizer.pad_token_id)
        resp_b_masks = zero_pad_sequences(resp_b_masks)
        
        return resp_a_ids, resp_a_masks, resp_b_ids, resp_b_masks, result_ids


args = script_args()

# GPU T4 x 2
device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")

model_0 = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        lora_rank=args.lora_rank,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        device_map='cuda:0'
    )

model_1 = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        lora_rank=args.lora_rank,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        device_map='cuda:1'
    )


tokenizer = get_tokenizer(args.pretrain, model_0, "left", use_fast=not args.disable_fast_tokenizer)

####################################################################################################
####################################################################################################
model_0.eval()
model_1.eval()
template = "Human: {}\nAssistant: {}"
# Define inference function
def inference(data_loader, model, device):
    results = []
    for resp_a_ids, resp_a_masks, resp_b_ids, resp_b_masks, ids in tqdm(data_loader):
        resp_a_ids = resp_a_ids.squeeze(1).to(device)
        resp_a_masks = resp_a_masks.squeeze(1).to(device).to(device)
        resp_b_ids = resp_b_ids.squeeze(1).to(device).to(device)
        resp_b_masks = resp_b_masks.squeeze(1).to(device).to(device)
        batch_size = resp_a_ids.size(0)
        max_batch_size = max(resp_a_ids.size(-1), resp_b_ids.size(-1))

        resp_a_ids = zero_pad_sequences(resp_a_ids, max_len=max_batch_size, value=tokenizer.pad_token_id)
        resp_a_masks = zero_pad_sequences(resp_a_masks, max_len=max_batch_size)
        resp_b_ids = zero_pad_sequences(resp_b_ids, max_len=max_batch_size, value=tokenizer.pad_token_id)
        resp_b_masks = zero_pad_sequences(resp_b_masks, max_len=max_batch_size)

        resp_ids = torch.cat(([resp_a_ids, resp_b_ids]), dim=0)
        resp_masks = torch.cat(([resp_a_masks, resp_b_masks]), dim=0)

        with torch.no_grad():
            rewards = model(resp_ids, resp_masks).cpu()

        resp_a_rewards = rewards[:batch_size]
        resp_b_rewards = rewards[batch_size:]
        for i in range(batch_size): 
            results.append({
                "id": ids[i],
                "resp_a_reward": resp_a_rewards[i].item(),
                "resp_b_reward": resp_b_rewards[i].item()
            })
    return results


# GPU x 2
from threading import Thread
logger.info("begin-inference")
results = []
df = pd.read_csv(args.test_file, dtype={'prompt': str, 'response_a': str, 'response_a': str})
df['prompt'] = df['prompt'].replace('null', "'null'")
df['response_a'] = df['response_a'].replace('null', "'null'")
df['response_b'] = df['response_b'].replace('null', "'null'")
df['total_length'] = df['prompt'].str.len() + df['response_a'].str.len() + df['response_b'].str.len()
##### TODO
df = df.sample(n=1000, random_state=42)
###### TODO
df_sorted = df.sort_values(by='total_length', ascending=True)
sub_df1 = df_sorted.iloc[::2]  # 索引为 0, 2, 4, ...
sub_df2 = df_sorted.iloc[1::2]  # 索引为 1, 3, 5, ...

eval_data_1 = datasets.Dataset.from_pandas(sub_df1)
eval_data_2 = datasets.Dataset.from_pandas(sub_df2)

eval_data_1 = RewardDataset(eval_data_2, tokenizer, max_length=args.max_length)
eval_data_2 = RewardDataset(eval_data_2, tokenizer, max_length=args.max_length)

data_loader_1 = DataLoader(
    eval_data_1,
    batch_size=1,
    drop_last=False,
    collate_fn=eval_data_1.collate_fn,
    pin_memory=False
)

data_loader_2 = DataLoader(
    eval_data_2,
    batch_size=1,
    drop_last=False,
    collate_fn=eval_data_2.collate_fn,
    pin_memory=False
)

def run_inference(data_loader, model, device, results, index):
    results[index] = inference(data_loader, model, device)

# Dictionary to store results from threads
results = {}
t0 = Thread(target=run_inference, args=(data_loader_1, model_0, device_0, results, 0))
# t1 = Thread(target=run_inference, args=(data_loader_2, model_1, device_1, results, 1))

t0.start()
# t1.start()

t0.join()
# t1.join()

print(results)
df0 = pd.DataFrame(results[0])
df1 = pd.DataFrame(results[1])
df =  pd.concat([df0, df1], axis=0)
df.to_csv(args.reward_data_path, index=False)
logger.info(f"Results saved to {args.reward_data_path}")
logger.info(f'end training, sample: {len(df)}')
