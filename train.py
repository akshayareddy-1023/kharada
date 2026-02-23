import torch
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from model.transformer import KharadaTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

vocab_size = sp.get_piece_size()
pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()

df = pd.read_csv("data/train.csv")


def encode(sentence):
    return [bos_id] + sp.encode(sentence, out_type=int) + [eos_id]


class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src = torch.tensor(encode(str(row['kharada'])))
        tgt = torch.tensor(encode(str(row['english'])))
        return src, tgt


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_id
    )

    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_id
    )

    return src_batch.to(device), tgt_batch.to(device)


dataset = TranslationDataset(df)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

model = KharadaTransformer(vocab_size=vocab_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=pad_id,
    label_smoothing=0.1
)

best_loss = float('inf')

for epoch in range(30):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        optimizer.zero_grad()

        output = model(src, tgt[:, :-1], pad_id)

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "model/best_model.pt")
        print("Saved best model.")

print("Training complete.")