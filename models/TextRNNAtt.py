import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, text_field, embedding_dim, model_name):
        self.model_name = model_name
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = embedding_dim
        self.hidden_size = 256
        self.num_classes = 4
        self.bidirectional = True
        self.dropout = 0.5
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]
        self.pad_size = 32
        self.num_layers = 2
        self.hidden_size2 = 64


class Model(nn.Module):
    """
    Paper: https://www.aclweb.org/anthology/P16-2034.pdf
    Ref: https://zhuanlan.zhihu.com/p/352129643
    (1) Input layer: input sentence to this model;
    (2) Embedding layer: map each word into a low
    dimension vector;
    (3) LSTM layer: utilize BLSTM to get high level features from step (2);
    (4) Attention layer: produce a weight vector,
    and merge word-level features from each time step
    into a sentence-level feature vector, by multiplying
    the weight vector;
    (5) Output layer: the sentence-level feature vector is finally used for relation classification.
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.model_name = f"{config.model_name.lower()}-{config.embedding_dim}"
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, text):
        # input: text = [batch size, seq len]
        embed = self.embedding(text)  # embed = [batch size, seq len, embedding dim]
        H, _ = self.lstm(embed)  # H = [batch size, seq len, hidden size * 2]
        M = self.tanh(H)  # M = [batch size, seq len, hidden size * 2]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # alpha = [batch size, seq len, 1]
        out = H * alpha  # out = [batch size, seq len, hidden size * 2]
        out = F.relu(torch.sum(out, 1))  # out = [batch size, hidden size * 2]
        out = self.fc1(out)  # out = [batch size, hidden_size2]
        out = self.fc(out)  # out = [batch size, num_classes]
        return out
