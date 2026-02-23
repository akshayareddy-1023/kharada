import torch
import pandas as pd
import sentencepiece as spm
import sacrebleu
from model.transformer import KharadaTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

vocab_size = sp.get_piece_size()

model = KharadaTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("model/best_model.pt"))
model.eval()

df = pd.read_csv("data/test.csv")

predictions = []
references = []

def encode(s): return sp.encode(s, out_type=int)

for _, row in df.iterrows():
    src = torch.tensor(encode(row['kharada'])).unsqueeze(0).to(device)
    output = model(src, src)
    pred_ids = output.argmax(dim=-1).squeeze().tolist()
    pred_text = sp.decode(pred_ids)

    predictions.append(pred_text)
    references.append(row['english'])

bleu = sacrebleu.corpus_bleu(predictions, [references])
print("BLEU score:", bleu.score)