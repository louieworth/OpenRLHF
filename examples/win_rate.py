import json
import csv
import argparse
####################
# Dataset formulate = {"prompt": xxx, "response": xxx, "reward": xxx}
####################
def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def compare_rewards(file1, file2, output_path):
    data_1 = load_data(file1)
    data_2 = load_data(file2)

    rewards_1 = {entry['prompt']: entry['reward'] for entry in data_1}
    rewards_2 = {entry['prompt']: entry['reward'] for entry in data_2}

    csv_data = []
    model2_wins = 0
    ties = 0
    total_comparisons = len(rewards_1)

    # Calculate average performance of model 2 for tie threshold
    avg_model2_performance = sum(rewards_2.values()) / total_comparisons
    tie_threshold = avg_model2_performance * 0.05

    for prompt in rewards_1:
        reward1 = rewards_1[prompt]
        reward2 = rewards_2[prompt]
        margin = reward2 - reward1
        csv_data.append([round(reward1, 2), round(reward2, 2), round(margin, 2)])

        if abs(margin) <= tie_threshold:
            ties += 1
        elif margin > 0:
            model2_wins += 1

    model2_winrate = model2_wins / total_comparisons
    tie_rate = ties / total_comparisons
    loss_rate = 1 - model2_winrate - tie_rate
    avg_model1_performance = sum(rewards_1.values()) / total_comparisons


    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["reward1", "reward2", "margin"])
        writer.writerows(csv_data)
        # Write win/tie/loss rates at the end of the CSV
        writer.writerow([])
        writer.writerow([])
        writer.writerow(['Model 1 average reward', f"{round(avg_model1_performance, 2)}"])
        writer.writerow(['Model 2 average reward', f"{round(avg_model2_performance, 2)}"])
        writer.writerow(['Model 2 win rate', f"{model2_winrate * 100:.2f}%"])
        writer.writerow(['Tie rate', f"{tie_rate * 100:.2f}%"])
        writer.writerow(['Loss rate', f"{loss_rate * 100:.2f}%"])


    print(f"Model 2 win rate: {model2_winrate * 100:.2f}%")
    print(f"Tie rate: {tie_rate * 100:.2f}%")
    print(f"Loss rate: {loss_rate * 100:.2f}%")
    print(f"CSV file created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare rewards from two models.")
    parser.add_argument("--file1", type=str, help="Path to the first JSONL file")
    parser.add_argument("--file2", type=str, help="Path to the second JSONL file")
    parser.add_argument("--output", type=str, help="Output path for the CSV file")
    args = parser.parse_args()

    compare_rewards(args.file1, args.file2, args.output)

if __name__ == "__main__":
    main()

