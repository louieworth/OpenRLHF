import subprocess
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import logging


class script_args:
    pretrain = "/data02/wenhao/jl/ckpt/rm/rm-lmsys-FsfairX_epoch4"
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


def get_gpu_memory_map():
    """获取当前GPU的显存使用情况"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                            capture_output=True, encoding='utf-8')
    # 转换为整数列表
    gpu_memory = [int(x) for x in result.stdout.strip().split('\n')]
    return gpu_memory

def max_memory_used(device_id):
    """返回指定设备的最大显存使用量"""
    return torch.cuda.max_memory_allocated(device=device_id) / (1024 ** 2)  # 转换为MB


def init_logger(name: str):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create a handler (for example, a console handler)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    
    # Create a formatter and set it to the handler
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
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
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

    try:
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, head_prefix)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

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

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
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
eval_length = length
acc = 0
tie = 0
for i in tqdm(range(length - eval_length, length)):
    data = eval_dataset[i]
    id = data["id"]
    prompt = data["prompt"]
    resp_a = data["response_a"]
    resp_b = data["response_b"]

    prompt_tokens = tokenizer(prompt, max_length=args.max_length, truncation=True, padding=False)
    resp_a_tokens = tokenizer(resp_a, max_length=args.max_length, truncation=True, padding=False)
    resp_b_tokens = tokenizer(resp_b, max_length=args.max_length, truncation=True, padding=False)

    prompt = tokenizer.decode(prompt_tokens['input_ids'], skip_special_tokens=True)
    resp_a = tokenizer.decode(resp_a_tokens['input_ids'], skip_special_tokens=True)
    resp_b = tokenizer.decode(resp_b_tokens['input_ids'], skip_special_tokens=True)


    resp_a = template.format(prompt, resp_a)
    resp_b = template.format(prompt, resp_b)

    if not resp_a.endswith(tokenizer.eos_token):
            resp_a += " " + tokenizer.eos_token
    if not resp_b.endswith(tokenizer.eos_token):
            resp_b += " " + tokenizer.eos_token

    resp_a_token = tokenizer(
            resp_a,
            max_length=args.max_total_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

    resp_b_token = tokenizer(
            resp_b,
            max_length=args.max_total_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

    resp_a_ids = resp_a_token["input_ids"]
    resp_a_mask = resp_a_token["attention_mask"]
    resp_b_ids = resp_b_token["input_ids"]
    resp_b_mask = resp_b_token["attention_mask"]

    resp_a_ids[0][-1] = tokenizer.eos_token_id
    resp_b_ids[0][-1] = tokenizer.eos_token_id
    resp_a_mask[0][-1] = True
    resp_b_mask[0][-1] = True
    
    with torch.no_grad():
        with autocast():
            torch.cuda.empty_cache()
            resp_a_ids = resp_a_ids.to(device)
            resp_a_mask = resp_a_mask.to(device)
            resp_a_reward = round(model(resp_a_ids, resp_a_mask).cpu().item(), 2)
            current_memory = max_memory_used('cuda:0')
            if current_memory > max_memory_observed:
                max_memory_observed = current_memory
                logger.info(f"id: {id} | Max memory observed: {max_memory_observed} MB")
            del resp_a_ids, resp_a_mask
        with autocast():
            torch.cuda.empty_cache()
            resp_b_ids = resp_b_ids.to(device)
            resp_b_mask = resp_b_mask.to(device)
            resp_b_reward = round(model(resp_b_ids, resp_b_mask).cpu().item(), 2)
            if current_memory > max_memory_observed:
                max_memory_observed = current_memory
                logger.info(f"id: {id} | Max memory observed: {max_memory_observed} MB")
            del resp_b_ids, resp_b_mask
    torch.cuda.empty_cache()
    if data['winner_tie'] == 1:
        winner = "tie"
    elif data['winner_model_a'] == 1:
        winner = "a"
    elif data['winner_model_b'] == 1:
        winner = "b"
    else:
        winner = "unknown"
    reward_gap = resp_a_reward - resp_b_reward
    winner_model_a, winner_model_b, winner_tie = 0, 0, 0
    if -args.margin <= reward_gap <= args.margin:
        winner_tie = 1
        if data['winner_tie'] == 1:
            acc += 1
    elif reward_gap > args.margin:
        winner_model_a = 1
        if data['winner_model_a'] == 1:
            acc += 1
    else: 
        winner_model_b = 1
        if data['winner_model_b'] == 1:
            acc += 1
    
    assert winner_model_a + winner_model_b + winner_tie == 1

    result = {
        "id": id, 
        "resp_a_reward": resp_a_reward,
        "resp_b_reward": resp_b_reward,
        "winner_model_a": winner_model_a,
        "winner_model_b": winner_model_b,
        "winner_tie": winner_tie,
        "ground_winner": winner,
    }

    results.append(result)

record = {
    "model": args.pretrain,
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
df.to_csv(f"{args.save_path}/reward.csv", index=False)
print(f"Results saved to {jsonl_file_path}")
