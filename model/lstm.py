import torch
import torch.nn as nn
import random
from model.attention import Attention


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        embedded = self.dropout(self.embedding(src))

        encoder_outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell = torch.cat((cell[-2], cell[-1]), dim=1)

        hidden = torch.tanh(self.fc_hidden(hidden)).unsqueeze(0)
        cell = torch.tanh(self.fc_cell(cell)).unsqueeze(0)

        return encoder_outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=256,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.attention = Attention(hidden_dim)

        self.lstm = nn.LSTM(
            emb_dim + hidden_dim * 2,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 3, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input,
        hidden,
        cell,
        encoder_outputs
    ):

        input = input.unsqueeze(1)

        embedded = self.dropout(self.embedding(input))

        attention = self.attention(hidden, encoder_outputs)

        attention = attention.unsqueeze(1)

        context = torch.bmm(attention, encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        prediction = self.fc(
            torch.cat(
                (
                    output.squeeze(1),
                    context.squeeze(1)
                ),
                dim=1
            )
        )

        return prediction, hidden, cell


class KharadaLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size,
            emb_dim,
            hidden_dim,
            num_layers,
            dropout
        )

        self.decoder = Decoder(
            vocab_size,
            emb_dim,
            hidden_dim,
            dropout
        )

    def forward(
        self,
        src,
        tgt,
        teacher_forcing_ratio=0.5
    ):

        batch_size = src.shape[0]

        trg_len = tgt.shape[1]

        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            trg_len,
            vocab_size,
            device=src.device
        )

        encoder_outputs, hidden, cell = self.encoder(src)

        input = tgt[:, 0]

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(
                input,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = tgt[:, t] if teacher_force else top1

        return outputs