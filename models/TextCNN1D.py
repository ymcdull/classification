import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, text_field, embedding_dim, model_name):
        self.model_name = model_name
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = embedding_dim
        self.n_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.output_dim = 4
        self.dropout = 0.5
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model_name = f"{config.model_name.lower()}-{config.embedding_dim}"
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim,
                                      padding_idx=config.pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config.embedding_dim,
                      out_channels=config.n_filters,
                      kernel_size=fs)
            for fs in config.filter_sizes
        ])
        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embed = self.embedding(text)  # embed = [batch size, sent len, embedding dim]
        embed = embed.permute(0, 2, 1)  # embed = [batch size, embedding dim, sent len]
        conved = [F.relu(conv(embed)) for conv in
                  self.convs]  # conved_n = [batch size, n_filters, sent len - filter_size_n + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))  # cat = [batch size, len(filter_sizes) * n_filters]
        out = self.fc(cat)  # out = [batch size, output_dim]
        return out
