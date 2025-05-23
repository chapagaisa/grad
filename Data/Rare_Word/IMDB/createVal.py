import pandas as pd

# Load the clean training dataset
df_train = pd.read_csv("clean_train.csv")

# Select 2,500 random samples for validation
df_val = df_train.sample(n=2500, random_state=42)

# Remove selected samples from training data
df_train = df_train.drop(df_val.index)

# Save the new training and validation datasets
df_train.to_csv("clean_train.csv", index=False)  # Updated training set
df_val.to_csv("clean_val.csv", index=False)  # New validation set

print("? Successfully split 2,500 samples into 'clean_val.csv' and updated 'clean_train.csv'.")
