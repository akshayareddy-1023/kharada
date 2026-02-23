import os
import sentencepiece as spm


def train_tokenizer(
    input_file="data/dataset.txt",
    model_prefix="tokenizer/bpe",
    vocab_size=800,
    character_coverage=1.0
):
    """
    Trains a SentencePiece BPE tokenizer.
    
    Args:
        input_file (str): Path to training text file
        model_prefix (str): Output prefix for tokenizer files
        vocab_size (int): Vocabulary size
        character_coverage (float): 1.0 for full unicode coverage
    """

    # Create tokenizer directory if not exists
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>"
    )

    print("✅ Tokenizer training completed!")
    print(f"Model saved as: {model_prefix}.model")
    print(f"Vocab saved as: {model_prefix}.vocab")


if __name__ == "__main__":
    train_tokenizer()