import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, text_field):
        self.vocab_size = len(text_field.vocab)
        self.embed = 100
        self.hidden_size = 256
        self.num_classes = 4
        self.bidirectional = True
        self.dropout = 0.5
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]
        self.pad_size = 32
        self.num_layers = 2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed, padding_idx=config.pad_idx)

        self.lstm = nn.LSTM(config.embed,
                            config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=config.bidirectional,
                            batch_first=True,
                            dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        # x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
