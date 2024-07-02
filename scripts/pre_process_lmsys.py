import pandas as pd
from datasets import load_dataset, DatasetDict

# Define the file paths for the original dataset and the processed datasets
original_file_path = '/data02/wenhao/jl/datasets/lmsys_all.csv'
train_output_path = '/data02/wenhao/jl/datasets/lmsys_train.csv'
test_output_path = '/data02/wenhao/jl/datasets/lmsys_test.csv'

# Load the original dataset
dataset = load_dataset('csv', data_files={'train': original_file_path})

# Access the train dataset directly using the key
train_dataset = dataset['train']

# Preprocess function
def preprocess_function(example):
    # Convert types
    example['id'] = int(example['id'])
    example['winner_model_a'] = int(example['winner_model_a'])
    example['winner_model_b'] = int(example['winner_model_b'])
    example['winner_tie'] = int(example['winner_tie'])
    
    # Ensure all model and prompt responses are treated as strings
    example['model_a'] = str(example['model_a'])
    example['model_b'] = str(example['model_b'])
    example['prompt'] = str(example['prompt'])
    example['response_a'] = str(example['response_a'])
    example['response_b'] = str(example['response_b'])
    
    return example

# Apply the preprocess function
train_dataset = train_dataset.map(preprocess_function)

# Remove rows where any response is null or empty
def filter_responses(example):
    return example['response_a'].strip() != '' and example['response_b'].strip() != ''

train_dataset = train_dataset.filter(filter_responses)

# Remove rows where the sum of winner columns is not equal to 1
def filter_winner(example):
    return example['winner_model_a'] + example['winner_model_b'] + example['winner_tie'] == 1

train_dataset = train_dataset.filter(filter_winner)

# Split the train dataset into train and test sets (90% train, 10% test)
train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Handle 'winner_tie' in the train set
train_df = pd.DataFrame(train_dataset)

# 创建一个新的 DataFrame 用于存储处理后的数据
processed_data = []

for idx, example in train_df.iterrows():
    if example['winner_tie'] == 1:
        example_a = example.copy()
        example_b = example.copy()
        
        example_a['winner_model_a'] = 1
        example_a['winner_model_b'] = 0
        example_a['winner_tie'] = 0
        
        example_b['winner_model_a'] = 0
        example_b['winner_model_b'] = 1
        example_b['winner_tie'] = 0
        
        processed_data.append(example_a)
        processed_data.append(example_b)
    else:
        processed_data.append(example)

# 将处理后的数据转换为 DataFrame
processed_df_train = pd.DataFrame(processed_data)
# Shuffle the training dataset
processed_df_train = processed_df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the cleaned and processed data
processed_df_train.to_csv(train_output_path, index=False)
test_dataset.to_csv(test_output_path, index=False)

print('Data preprocessing completed and saved to:', train_output_path, 'and', test_output_path)

# Load the processed datasets
data_files = {
    'train': train_output_path,
    'test': test_output_path
}

dataset = load_dataset('csv', data_files=data_files)

# 分别获取训练集和测试集
train_dataset = dataset['train']
test_dataset = dataset['test']

# 检查加载的数据集
print(train_dataset)
print(test_dataset)