import torch
import torch.nn as nn


class Config:
    def __init__(self, text_field, embedding_dim, model_name):
        self.model_name = model_name
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = 256
        self.output_dim = 4
        self.n_layers = 2
        self.bidirectional = True
        self.dropout = 0.5
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model_name = f"{config.model_name.lower()}-{config.embedding_dim}"
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim,
                                      padding_idx=config.pad_idx)
        self.rnn = nn.LSTM(input_size=config.embedding_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.n_layers,
                           bidirectional=config.bidirectional,
                           dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 2, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):
        text, text_lengths = text  # text = [seq len, batch size], text_lengths = [batch size]
        embed = self.embedding(text)  # embed = [seq len, batch size, embedding dim]
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, text_lengths,
                                                         enforce_sorted=False)  # packed_embed.data = [all words in the batch, embedding dim]
        packed_output, (hidden, cell) = self.rnn(
            packed_embed)  # hidden = [num layers * num directions, batch size, hidden dim]
        # Concat last layer with bidirectional
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # hidden = [batch size, hidden dim * 2]
        out = self.fc(hidden)
        return out
