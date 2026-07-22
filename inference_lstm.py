import torch
import sentencepiece as spm
from model.lstm import KharadaLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
vocab_size = sp.get_piece_size()

# Load LSTM model
model = KharadaLSTM(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("model/best_lstm.pt", map_location=device))
model.eval()

print("LSTM Model loaded successfully!\n")


def translate(sentence, max_len=40):

    src_ids = [bos_id] + sp.encode(sentence, out_type=int) + [eos_id]

    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    tgt_ids = [bos_id]

    with torch.no_grad():

        for _ in range(max_len):

            tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

            output = model(src_tensor, tgt_tensor)

            next_token = output[:, -1, :].argmax(dim=-1).item()

            if next_token == eos_id:
                break

            tgt_ids.append(next_token)

    return sp.decode(tgt_ids[1:])


while True:

    sentence = input("Enter Kharada sentence (or exit): ")

    if sentence.lower() == "exit":
        break

    print("English Translation:", translate(sentence))
    print()