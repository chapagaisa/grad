import pandas as pd

# Load the clean and poisoned test datasets
clean_df = pd.read_csv("clean_test.csv")
poisoned_df = pd.read_csv("poisoned_test.csv")

# Sample 2000 rows from each dataset (assuming they have at least 2000 rows)
pilot_clean_df = clean_df.sample(n=2000, random_state=42)
pilot_poisoned_df = poisoned_df.sample(n=2000, random_state=42)

# Save the sampled datasets
pilot_clean_df.to_csv("pilot_clean_2000.csv", index=False)
pilot_poisoned_df.to_csv("pilot_poisoned_2000.csv", index=False)

print("Pilot datasets created successfully!")
