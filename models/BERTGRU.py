import torch
import torch.nn as nn
from transformers import BertModel


class Config:
    def __init__(self):
        self.hidden_dim = 256
        self.output_dim = 4
        self.n_layers = 2
        self.bidirectional = True
        self.dropout = 0.25
        self.bert = BertModel.from_pretrained('bert-base-uncased')


class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()

        self.bert = config.bert

        embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          config.hidden_dim,
                          num_layers=config.n_layers,
                          bidirectional=config.bidirectional,
                          batch_first=True,
                          dropout=0 if config.n_layers < 2 else config.dropout)

        self.out = nn.Linear(config.hidden_dim * 2 if config.bidirectional else config.hidden_dim, config.output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output
