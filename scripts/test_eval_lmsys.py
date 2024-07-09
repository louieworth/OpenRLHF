import subprocess
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
import datasets
# from datasets import load_dataset, Dataset
import logging
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)

class script_args:
    pretrain = "/data02/wenhao/jl/ckpt/rm/rm-lmsys-FsfairX_epoch4"
    test_file = "/data02/wenhao/jl/datasets/lmsys_all.csv"
    save_path = "scripts/results"
    batch_size = 1
    max_length = 1024
    max_total_length = 1024
    init_value_head = False
    margin = 0.1
    flash_attn = True
    bf16 = True
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
    load_in_4bit=True,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
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
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None


    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
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
model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        normalize_reward=True,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
    )

tokenizer = get_tokenizer(args.pretrain, model, "left", use_fast=not args.disable_fast_tokenizer)
# model.to(args.device)
model.eval()
template = "Human: {}\nAssistant: {}"


####################################################################################################
###############
# batch size > 2
# use data collate_fn

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    # for i, seq in enumerate(sequences):
    #     if not isinstance(seq, torch.Tensor):
    #         sequences[i] = torch.tensor(seq).reshape(1, -1)

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
        self.ids = []
        for data in tqdm(dataset):
            self.prompts.append(data['prompt'])
            self.resp_as.append(data['response_a'])
            self.resp_bs.append(data['response_b'])
            self.ids.append(data['id'])
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        result_id = self.ids[idx]
        prompt = self.prompts[idx]
        resp_a = self.resp_as[idx]
        resp_b = self.resp_bs[idx]
        
        resp_a = self.template.format(prompt, resp_a)
        resp_b = self.template.format(prompt, resp_b)
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
            result_id
        )
    def collate_fn(self, item_list):
        resp_a_ids = []
        resp_a_masks = []
        resp_b_ids = []
        resp_b_masks = []
        result_ids = []

        for resp_a_id, resp_a_mask, resp_b_id, resp_b_mask, result_id in item_list:
            assert resp_a_id.shape == resp_a_mask.shape
            assert resp_b_id.shape == resp_b_mask.shape
            resp_a_ids.append(resp_a_id)
            resp_a_masks.append(resp_a_mask)
            resp_b_ids.append(resp_b_id)
            resp_b_masks.append(resp_b_mask)
            result_ids.append(result_id)

        resp_a_ids = zero_pad_sequences(resp_a_ids, value=self.tokenizer.pad_token_id)
        resp_a_masks = zero_pad_sequences(resp_a_masks)
        resp_b_ids = zero_pad_sequences(resp_b_ids, value=self.tokenizer.pad_token_id)
        resp_b_masks = zero_pad_sequences(resp_b_masks)
        
        return resp_a_ids, resp_a_masks, resp_b_ids, resp_b_masks, result_ids

# batch_size > 1
####################################################################################################
logger.info("begin-inference")
results = []
df = pd.read_csv(args.test_file)
df['prompt'] = df['prompt'].replace('null', "'null'")
df['response_a'] = df['response_a'].replace('null', "'null'")
df['response_b'] = df['response_b'].replace('null', "'null'")
eval_dataset = datasets.Dataset.from_pandas(df)
length =  len(eval_dataset)
eval_length = 2500

dataset = RewardDataset(eval_dataset, tokenizer, args.max_length)
# data_loader = DataLoader(eval_dataset, batch_size=2, collate_fn=dataset.collate_fn)
data_loader = DataLoader(
    dataset,
    batch_size=4,
    drop_last=False,
    collate_fn=dataset.collate_fn,
    pin_memory=False
)

results = []
for input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, ids in tqdm(data_loader):
    input_ids_a = input_ids_a.squeeze(1)
    attention_mask_a = attention_mask_a.squeeze(1)
    input_ids_b = input_ids_b.squeeze(1)
    attention_mask_b = attention_mask_b.squeeze(1)

    with torch.no_grad():
        torch.cuda.empty_cache()
        input_ids_a = input_ids_a.to(device)
        attention_mask_a = attention_mask_a.to(device)
        resp_a_reward = model(input_ids_a, attention_mask_a).cpu().detach()
        input_ids_a.to('cpu')
        attention_mask_a.to('cpu')
        del input_ids_a
        del attention_mask_a
        
        torch.cuda.empty_cache()
        input_ids_b = input_ids_b.to(device)
        attention_mask_b = attention_mask_b.to(device)
        resp_b_reward = model(input_ids_b, attention_mask_b).cpu().detach()
        input_ids_b.to('cpu')
        attention_mask_b.to('cpu')
        del input_ids_b
        del attention_mask_b

    for i in range(len(ids)):  # This assumes your batch size
        results.append({
            "id": ids[i],
            "resp_a_reward": resp_a_reward[i].item(),
            "resp_b_reward": resp_b_reward[i].item()
        })

df_results = pd.DataFrame(results)
df_results.to_csv('path_to_save.csv', index=False)
df = pd.DataFrame(results)
df.to_csv(args.reward_data_path, index=False)
logger.info(f"Results saved to {args.reward_data_path}")
logger.info(f'end training, sample: {eval_length}')