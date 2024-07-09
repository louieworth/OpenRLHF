import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import json
import random
import logging
import datetime
import pytz
from sklearn.preprocessing import LabelEncoder

timezone = pytz.timezone("America/New_York")
current_time = datetime.datetime.now(tz=pytz.utc).astimezone(timezone)
formatted_time = current_time.strftime("%m%d_%H%M")



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


class PredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1, temperature=1.0):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.temperature = temperature

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x / self.temperature # Softmax temperature
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = x / self.temperature
            return F.softmax(x, dim=1), x


def load_data(load_file_path):
    data = pd.read_csv(load_file_path)
    data['reward_gap'] = data['resp_a_reward'] - data['resp_b_reward']
    features = data[['resp_a_reward', 'resp_b_reward', 'reward_gap']].values
    target_mapping = {'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2}
    if 'winner_model_a' in data.columns:
        target_mapping = {'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2}
        targets = data[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map(target_mapping).values
    else:
        targets = data['winner'].values
    return train_test_split(features, targets, test_size=0.2, random_state=42)

def create_dataset(features, targets):
    features_tensor = torch.tensor(features, dtype=torch.float)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(features_tensor, targets_tensor)
    return dataset

def create_inference_dataset(file_path, ratio=0.2, random_sample=True):
    data = pd.read_csv(file_path)
    data['reward_gap'] = data['resp_a_reward'] - data['resp_b_reward']
    features = data[['resp_a_reward', 'resp_b_reward', 'reward_gap']].values

    le = LabelEncoder()
    ids = le.fit_transform(data['id'].values)
    # ids = data['id'].values  # Assuming there is an 'id' column
    features_tensor = torch.tensor(features, dtype=torch.float)
    if 'winner_model_a' in data.columns:
        target_mapping = {'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2}
        targets = data[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map(target_mapping).values
    else:
        targets = data['winner'].values
    dataset = TensorDataset(features_tensor, torch.tensor(ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long))

    if random_sample:
        sample_indices = random.sample(range(len(data)), int(len(data) * ratio))
    else:
        sample_indices = range(int(len(data) * (1 - ratio)), len(data))
    
    sampled_features = features_tensor[sample_indices]
    sampled_ids = torch.tensor(ids[sample_indices], dtype=torch.long)
    sampled_targets = torch.tensor(targets[sample_indices], dtype=torch.long)
    
    dataset = TensorDataset(sampled_features, sampled_ids, sampled_targets)
    return dataset

# 训练函数
def train(model, args, data_loader, test_loader, criterion, optimizer, device, base_path):
    model.train()
    min_loss = 0.45
    for epoch in range(args.epochs):
        save_model_path = None
        total_loss = 0
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        mean_train_loss = round(total_loss / len(data_loader), 4)
        logger.info(f'Epoch {epoch} - Training Loss: {mean_train_loss}')

        # Evaluate the model for every epoch
        with torch.no_grad():
            model.eval()
            total_loss = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                pred, logits = model.predict(inputs)
                loss = criterion(logits, labels)
                total_loss += loss.item()
            mean_eval_loss = round(total_loss / len(test_loader),4)
            logger.info(f'Epoch {epoch} - Evaluation Loss: {mean_eval_loss}')
            
        
        # Hardcoded evaluation
        total_hardcoded_loss = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Compute reward gaps
            reward_gaps = inputs[:, 0] - inputs[:, 1]
            
            # Initialize hardcoded outputs
            hardcoded_outputs = torch.zeros(inputs.size(0), 3, device=device)
            
            # Determine winners based on reward gaps
            epsilon = 0.05
            margin = args.margin
            
            hardcoded_outputs[:, 0] = (reward_gaps > margin).float() * (1 - epsilon * 3) + epsilon
            hardcoded_outputs[:, 1] = (reward_gaps < -margin).float() * (1 - epsilon * 3) + epsilon
            hardcoded_outputs[:, 2] = ((reward_gaps >= -margin) & (reward_gaps <= margin)).float() * (1 - epsilon * 3) + epsilon
            
            # Ensure the sum of probabilities is 1
            hardcoded_outputs = hardcoded_outputs / hardcoded_outputs.sum(dim=1, keepdim=True)
            
            # Compute the loss
            hardcoded_loss = criterion(hardcoded_outputs, labels)
            total_hardcoded_loss += hardcoded_loss.item()

        mean_hardcoded_loss = round(total_hardcoded_loss / len(test_loader), 4)
        logger.info(f'Epoch {epoch} - Hardcoded Evaluation Loss: {mean_hardcoded_loss}')
        if (epoch+1) % 50 == 0 and mean_eval_loss < min_loss:
            min_loss = mean_eval_loss
            save_model_path = f'{args.base_path}/ckpt/{formatted_time}_{base_path}_epoch_{epoch}.pth'
            if not os.path.exists(f"{args.base_path}/ckpt"):
                os.makedirs(f"{args.base_path}/ckpt")
            torch.save(model.state_dict(), save_model_path)
            logger.info(f'New best model saved {save_model_path}')

            result = {
                'dropout_rate': args.dropout_rate,
                'temperature': args.temperature,
                'learning_rate': args.learning_rate,
                'epoch': epoch,
                'train_loss': mean_train_loss,
                'eval_loss': mean_eval_loss,
                'hardcoded_loss': mean_hardcoded_loss,
                'model_path': save_model_path if save_model_path else 'None'
            }

    log_file_path = os.path.join(args.base_path, 'metrics.json')
    if not os.path.exists(args.base_path):
        os.makedirs(args.base_path)

    with open(log_file_path, 'w') as f:
        json.dump(result, f)

def inference(model, device, data_loader, criterion=nn.CrossEntropyLoss()):
    results = []
    total_loss = 0
    with torch.no_grad():
        for inputs, ids, labels in data_loader:
            inputs = inputs.to(device)
            preds, logits = model.predict(inputs)
            labels = labels.to(device)
            for i in range(preds.shape[0]):
                prob = preds[i].cpu().numpy()
                result = {
                    'id': ids[i].item(), 
                    # emove necessary columns for True evaluation
                    'resp_a_reward': round(inputs[i][0].item(), 2),
                    'resp_b_reward': round(inputs[i][1].item(), 2),
                    'winner_model_a': round(prob[0], 2),
                    'winner_model_b': round(prob[1], 2),
                    'winner_tie': round(prob[2], 2),
                    ################################
                    'true_winner': labels[i].item()
                    ################################
                }
                results.append(result)
            loss = criterion(logits, labels).cpu().item()
            total_loss += loss
        mean_eval_loss = round(total_loss / len(data_loader),4)
        logger.info(f'Evaluation Loss: {mean_eval_loss}')

    df = pd.DataFrame(results)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Train a prediction model.")
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training ("cpu" or "cuda").')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature.')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for the hardcoded evaluation.')
    
    parser.add_argument('--load_file_path', type=str, default='scripts/results/reward_total.csv', help='Path to the data file.')
    parser.add_argument('--base_path', type=str, default='/home/wenhao/jiangli/OpenRLHF/rm-to-win', help='Base path for saving the model and metrics.')
    #######
    # for inference
    #######
    parser.add_argument('--infer_data_path', type=str, default='scripts/results/chat_arean_reward_sample_1000.csv', help='Path to the data file.')
    parser.add_argument('--infer_model_path', type=str, default='rm-to-win/ckpt/0704_1723_epoch_149.pth', help='Path to the trained model.')
    parser.add_argument('--train_mode', type=bool, default=False, help='Train or inference mode.')

    return parser.parse_args()


args = parse_args()
device = torch.device(args.device)
criterion = nn.CrossEntropyLoss()
if args.train_mode:
    logger.info(f'Training model with the following parameters: {args}')
    train_features, test_features, train_targets, test_targets = load_data(args.load_file_path)
    # Create datasets
    train_dataset = create_dataset(train_features, train_targets)
    test_dataset = create_dataset(test_features, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = PredictionModel(input_dim=3, 
                            output_dim=3, 
                            dropout_rate=args.dropout_rate, 
                            temperature=args.temperature).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    base_train_path = os.path.basename(args.load_file_path)
    train(model, args, train_loader, test_loader, criterion, optimizer, device, base_train_path)
else:
    logger.info(f'Inference mode: data_path {args.infer_data_path} | model_path {args.infer_model_path}')
    inference_dataset = create_inference_dataset(args.infer_data_path, ratio=1, random_sample=False)
    inference_loader = DataLoader(inference_dataset, batch_size=1024, shuffle=True)


    logger.info(f"eval length: {len(inference_dataset)}")
    model = PredictionModel(input_dim=3, output_dim=3, dropout_rate=args.dropout_rate, temperature=args.temperature).to(device)
    model.load_state_dict(torch.load(args.infer_model_path))
    model.eval()
    infer_data_filename = os.path.basename(args.infer_data_path)
    infer_model_filename = os.path.basename(args.infer_model_path)

    result_path = os.path.join(args.base_path, f'{infer_data_filename}_{infer_model_filename}_submission.csv')
    df = inference(model, device, inference_loader)
    df.to_csv(result_path, index=False)
    logger.info(f'Results saved to {result_path}')
