import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import argparse

class PredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, temperature=1.0):
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



def create_inference_dataset(file_path):
    data = pd.read_csv(file_path)
    data['reward_gap'] = data['resp_a_reward'] - data['resp_b_reward']
    features = data[['resp_a_reward', 'resp_b_reward', 'reward_gap']].values
    ids = data['id'].values  # Assuming there is an 'id' column
    features_tensor = torch.tensor(features, dtype=torch.float)

    dataset = TensorDataset(features_tensor, torch.tensor(ids, dtype=torch.long))
    return dataset


def inference(model, device, data_loader):
    results = []
    with torch.no_grad():
        for inputs, ids in data_loader:
            inputs = inputs.to(device)
            preds, logits = model.predict(inputs)
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
                }
                results.append(result)

    df = pd.DataFrame(results)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Train a prediction model.")
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training ("cpu" or "cuda").')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature.')
    parser.add_argument('--base_path', type=str, default='/home/wenhao/jiangli/OpenRLHF/rm-to-win', help='Base path for saving the model and metrics.')

    parser.add_argument('--infer_data_path', type=str, default='rm-to-win/test_reward_0707.csv', help='Path to the data file.')
    parser.add_argument('--infer_model_path', type=str, default='rm-to-win/ckpt/0704_1723_epoch_149.pth', help='Path to the trained model.')
    return parser.parse_args()


args = parse_args()
device = torch.device(args.device)
inference_dataset = create_inference_dataset(args.infer_data_path)
inference_loader = DataLoader(inference_dataset, batch_size=1024, shuffle=False)
model = PredictionModel(input_dim=3, output_dim=3, temperature=args.temperature).to(device)
model.load_state_dict(torch.load(args.infer_model_path))
model.eval()

result_path = os.path.join(args.base_path, 'test_submission.csv')
df = inference(model, device, inference_loader)
df.to_csv(result_path, index=False)

