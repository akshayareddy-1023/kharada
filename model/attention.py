import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # Encoder is bidirectional -> hidden_dim*2
        # Decoder hidden -> hidden_dim
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)

        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden : [1, batch, hidden]
        # encoder_outputs : [batch, src_len, hidden*2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.squeeze(0)

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(
            self.attn(
                torch.cat((hidden, encoder_outputs), dim=2)
            )
        )

        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)