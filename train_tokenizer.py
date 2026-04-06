import pandas as pd
import sentencepiece as spm
import os

os.makedirs("tokenizer", exist_ok=True)

df = pd.read_csv("data/train.csv")

assert "kharada" in df.columns
assert "english" in df.columns

with open("tokenizer/combined.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(str(row["kharada"]).strip() + "\n")
        f.write(str(row["english"]).strip().lower() + "\n")

print("Combined text file created.")

spm.SentencePieceTrainer.train(
    input="tokenizer/combined.txt",
    model_prefix="tokenizer/bpe",
    vocab_size=300,   # for ~173 sentences
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3
)

print("Tokenizer training complete.")