import torch
import torch.nn as nn


class Config:
    def __init__(self, text_field):
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.output_dim = 4
        self.n_layers = 2
        self.bidirectional = True
        self.dropout = 0.5
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)

        self.rnn = nn.LSTM(config.embedding_dim,
                           config.hidden_dim,
                           num_layers=config.n_layers,
                           bidirectional=config.bidirectional,
                           dropout=config.dropout)

        self.fc = nn.Linear(config.hidden_dim * 2, config.output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):
        text, text_lengths = text
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        #         packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)
