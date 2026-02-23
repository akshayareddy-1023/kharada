import pandas as pd
import unicodedata
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/kharada_english.csv")

assert 'kharada' in df.columns
assert 'english' in df.columns

df.dropna(inplace=True)

df['kharada'] = df['kharada'].apply(
    lambda x: unicodedata.normalize("NFKC", str(x).strip())
)

df['english'] = df['english'].apply(
    lambda x: unicodedata.normalize("NFKC", str(x).strip().lower())
)

# Remove very short sentences
df = df[
    (df['kharada'].str.split().str.len() > 1) &
    (df['english'].str.split().str.len() > 1)
]

# Remove very long sentences (recommended)
df = df[
    (df['kharada'].str.split().str.len() <= 40) &
    (df['english'].str.split().str.len() <= 40)
]

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Data split complete.")