import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, text_field):
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = 100
        self.n_filters = 100
        self.filter_sizes = [3, 4, 5]
        self.output_dim = 4
        self.dropout = 0.5
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]


# Use for loop to take any number of filters
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=config.n_filters,
                      kernel_size=(fs, config.embedding_dim))
            for fs in config.filter_sizes
        ])

        self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, config.output_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

# Hard-code version with 3 different sized filters:
# class Model(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
#
#         self.conv_0 = nn.Conv2d(in_channels=1,
#                                 out_channels=config.n_filters,
#                                 kernel_size=(config.filter_sizes[0], config.embedding_dim))
#
#         self.conv_1 = nn.Conv2d(in_channels=1,
#                                 out_channels=config.n_filters,
#                                 kernel_size=(config.filter_sizes[1], config.embedding_dim))
#
#         self.conv_2 = nn.Conv2d(in_channels=1,
#                                 out_channels=config.n_filters,
#                                 kernel_size=(config.filter_sizes[2], config.embedding_dim))
#
#         self.fc = nn.Linear(len(config.filter_sizes) * config.n_filters, config.output_dim)
#
#         self.dropout = nn.Dropout(config.dropout)
#
#     def forward(self, text):
#         # text = [batch size, sent len]
#
#         embedded = self.embedding(text)
#
#         # embedded = [batch size, sent len, emb dim]
#
#         embedded = embedded.unsqueeze(1)
#
#         # embedded = [batch size, 1, sent len, emb dim]
#
#         conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
#         conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
#         conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
#
#         # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
#
#         pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
#         pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
#         pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
#
#         # pooled_n = [batch size, n_filters]
#
#         cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
#
#         # cat = [batch size, n_filters * len(filter_sizes)]
#
#         return self.fc(cat)
