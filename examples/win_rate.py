import json
import csv
import argparse
####################
# Dataset formulate = {"prompt": xxx, "response": xxx, "reward": xxx}
####################
import csv
import json

def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def compare_rewards(file1, file2, output_path, jsonl_output_path):
    data_1 = load_data(file1)
    data_2 = load_data(file2)

    rewards_1 = {entry['prompt']: (entry['response'], entry['reward']) for entry in data_1}
    rewards_2 = {entry['prompt']: (entry['response'], entry['reward']) for entry in data_2}

    csv_data = []
    jsonl_data = []
    model2_wins = 0
    ties = 0
    total_comparisons = len(rewards_1)

    avg_model2_performance = sum(entry[1] for entry in rewards_2.values()) / total_comparisons
    tie_threshold = avg_model2_performance * 0.05

    for prompt, (response1, reward1) in rewards_1.items():
        response2, reward2 = rewards_2[prompt]
        margin = reward2 - reward1
        csv_data.append([round(reward1, 2), round(reward2, 2), round(margin, 2)])
        jsonl_data.append({
            "prompt": prompt,
            "response1": response1,
            "reward1": reward1,
            "response2": response2,
            "reward2": reward2
        })

        if tie_threshold <= margin <= tie_threshold:
            ties += 1
        elif margin > 0:
            model2_wins += 1

    model2_winrate = model2_wins / total_comparisons
    tie_rate = ties / total_comparisons
    loss_rate = 1 - model2_winrate - tie_rate
    avg_model1_performance = sum(entry[1] for entry in rewards_1.values()) / total_comparisons

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reward1", "Reward2", "Margin"])
        writer.writerows(csv_data)
        writer.writerow([])
        writer.writerow(['Model 1 Average Reward', round(avg_model1_performance, 2)])
        writer.writerow(['Model 2 Average Reward', round(avg_model2_performance, 2)])
        writer.writerow(['Model 2 Win Rate', f"{model2_winrate * 100:.2f}%"])
        writer.writerow(['Tie Rate', f"{tie_rate * 100:.2f}%"])
        writer.writerow(['Loss Rate', f"{loss_rate * 100:.2f}%"])

    with open(jsonl_output_path, 'w') as jsonl_file:
        for item in jsonl_data:
            jsonl_file.write(json.dumps(item) + '\n')

    print(f"Model 2 win rate: {model2_winrate * 100:.2f}%")
    print(f"Tie rate: {tie_rate * 100:.2f}%")
    print(f"Loss rate: {loss_rate * 100:.2f}%")
    print(f"CSV file created at: {output_path}")
    print(f"JSONL file created at: {jsonl_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare rewards from two models.")
    parser.add_argument("--file1", type=str, help="Path to the first JSONL file")
    parser.add_argument("--file2", type=str, help="Path to the second JSONL file")
    parser.add_argument("--output", type=str, help="Output path for the CSV file")
    parser.add_argument("--jsonl_output", type=str, help="Output path for the JSONL file")
    args = parser.parse_args()

    compare_rewards(args.file1, args.file2, args.output, args.jsonl_output)

if __name__ == "__main__":
    main()

