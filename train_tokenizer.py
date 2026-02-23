import os
import sys
import pandas as pd
import sentencepiece as spm


def prepare_text_from_csv(csv_file, output_txt):
    """
    Reads kharada_english.csv and writes all sentences
    (both Kharada and English) into a single text file.
    """

    if not os.path.isfile(csv_file):
        print(f"\n❌ ERROR: CSV file '{csv_file}' not found!")
        sys.exit(1)

    df = pd.read_csv(csv_file)

    if len(df.columns) < 2:
        print("\n❌ ERROR: CSV must have at least two columns!")
        sys.exit(1)

    with open(output_txt, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(str(row[0]).strip() + "\n")
            f.write(str(row[1]).strip() + "\n")

    print(f"✅ Combined text file created: {output_txt}")


def train_tokenizer(
    csv_file="kharada_english.csv",
    model_prefix="tokenizer/bpe",
    vocab_size=800,
):
    print("\n📂 Working Directory:", os.getcwd())

    combined_txt = "combined.txt"

    # Step 1: Convert CSV → TXT
    prepare_text_from_csv(csv_file, combined_txt)

    # Step 2: Create tokenizer folder
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    print("\n🚀 Training tokenizer...\n")

    spm.SentencePieceTrainer.train(
        input=combined_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        hard_vocab_limit=False
    )

    print("\n✅ Tokenizer training completed!")
    print(f"📦 Saved: {model_prefix}.model")
    print(f"📦 Saved: {model_prefix}.vocab")


if __name__ == "__main__":
    train_tokenizer()