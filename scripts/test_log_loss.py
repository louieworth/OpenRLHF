import pandas as pd
from sklearn.metrics import log_loss

# Load the datasets
train_path = '/data02/wenhao/jl/datasets/lmsys_all.csv'
submission_path = 'rm-to-win/test_submission.csv'
train_df = pd.read_csv(train_path)
submission_df = pd.read_csv(submission_path)

# Ensure that the 'id' column is of the same type, typically integer or string
train_df['id'] = train_df['id'].astype(str)
submission_df['id'] = submission_df['id'].astype(str)
submission_ids = submission_df['id']
missing_ids = submission_ids[~submission_ids.isin(train_df['id'])]

if not missing_ids.empty:
    raise KeyError(f"The following ids are missing in train_df: {missing_ids.tolist()}")

# 筛选 train_df 中与 submission_ids 相匹配的行
formualted_train_df = train_df.set_index('id').loc[submission_ids].reset_index()

predictions = submission_df[['winner_model_a', 'winner_model_b', 'winner_tie']]
true_labels = formualted_train_df[['winner_model_a', 'winner_model_b', 'winner_tie']]


logloss_score = log_loss(true_labels, predictions)

print(f"Log Loss: {logloss_score}")