import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/kharada_english.csv")

# Remove accidental spaces in column names
df.columns = df.columns.str.strip()

print("Columns after strip:", df.columns)

df.dropna(inplace=True)

df['kharada'] = df['kharada'].astype(str).str.strip()
df['english'] = df['english'].astype(str).str.strip().str.lower()

df.drop_duplicates(inplace=True)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Preprocessing complete.")