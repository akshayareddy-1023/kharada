import torch
import pandas as pd
import sentencepiece as spm
from model.transformer import KharadaTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

vocab_size = sp.get_piece_size()

# Load training data
df = pd.read_csv("data/train.csv")

def encode(sentence):
    return sp.encode(sentence, out_type=int)

model = KharadaTransformer(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    model.train()

    for _, row in df.iterrows():
        src = torch.tensor(encode(row['kharada'])).unsqueeze(0).to(device)
        tgt = torch.tensor(encode(row['english'])).unsqueeze(0).to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss}")

torch.save(model.state_dict(), "model/best_model.pt")