import torch
import sentencepiece as spm
from model.transformer import KharadaTransformer
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

vocab_size = sp.get_piece_size()
pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()

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

        # Greedy decoding
        next_token = torch.argmax(logits, dim=-1).item()

        # Prevent immediate EOS at first step
        if step == 0 and next_token == eos_id:
            top2 = torch.topk(logits, 2).indices[0]
            next_token = top2[1].item()

        # Prevent repetition
        if len(tgt_ids) > 2 and next_token == tgt_ids[-1]:
            top2 = torch.topk(logits, 2).indices[0]
            next_token = top2[1].item()

        tgt_ids.append(next_token)

        if next_token == eos_id:
            break

    # Remove BOS
    output_ids = tgt_ids[1:]

    # Remove EOS if present
    if eos_id in output_ids:
        output_ids = output_ids[:output_ids.index(eos_id)]

    if len(output_ids) == 0:
        return "[Model unsure — try similar sentence from training data]"

    return sp.decode(output_ids)


# -------- INTERACTIVE LOOP --------
while True:

    text = input("Enter Kharada sentence (or type exit): ")

    if text.lower() == "exit":
        break

    translation = translate(text)
    print("English Translation:", translation)
    print()