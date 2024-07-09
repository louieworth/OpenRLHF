import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from datasets import Dataset, interleave_datasets, load_dataset
import ast

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


df = pd.read_csv('/data02/wenhao/jl/datasets/lmsys_chatbot_arena_conversations.csv')
dataset = Dataset.from_pandas(df)
dataset = dataset.select(range(10))
for data in dataset:
    prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
    response_a = data["response_a"]
    response_b = data["response_b"]
    response_a = process_multi_turn_dialogue(response_a)
    response_b = process_multi_turn_dialogue(response_b)

    print(prompt + response_a)