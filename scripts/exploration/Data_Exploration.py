import os
import pandas as pd

# Load the dataset
df = pd.read_csv(os.environ["CREDIT_BN_ROOT"] + '/Data/GiveMeSomeCredit/cs-training.csv')

# Get basic info on the dataset
print(df.shape)
print(df.info())

# Check missing values per column
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # as percentages

# Drop rows with any missing values
df_clean = df.dropna()

# Check how many fully observable rows remain
print(f"Original rows: {len(df)}")
print(f"Fully observable rows: {len(df_clean)}")
print(f"Rows lost: {len(df) - len(df_clean)}")

# Check class distribution in clean subset
print(df_clean['SeriousDlqin2yrs'].value_counts())
print(df_clean['SeriousDlqin2yrs'].value_counts(normalize=True) * 100)