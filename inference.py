import torch
import sentencepiece as spm
from model.transformer import KharadaTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
vocab_size = sp.get_piece_size()

# Load model
model = KharadaTransformer(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("model/best_model.pt", map_location=device))
model.eval()

print("Model loaded. Ready for translation.\n")


def translate(sentence, max_len=40):

    src_ids = [bos_id] + sp.encode(sentence, out_type=int) + [eos_id]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    tgt_ids = [bos_id]

    for step in range(max_len):

        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, pad_id)

        logits = output[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == eos_id:
            break

    output_ids = tgt_ids[1:]

    if eos_id in output_ids:
        output_ids = output_ids[:output_ids.index(eos_id)]

    if len(output_ids) == 0:
        return "[Model unsure]"

    return sp.decode(output_ids)


# Interactive loop
while True:
    text = input("Enter Kharada sentence (or type exit): ")

    if text.lower() == "exit":
        break

    print("English Translation:", translate(text))
    print()