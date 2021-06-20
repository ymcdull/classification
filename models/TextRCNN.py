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
        self.pad_size = 64
        self.num_layers = 2


class Model(nn.Module):
    """
    Paper: http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf
    Ref: https://zhuanlan.zhihu.com/p/55015587
    就深度学习领域来说，RNN和CNN作为文本分类问题的主要模型架构，都存在各自的优点及局限性。
    RNN擅长处理序列结构，能够考虑到句子的上下文信息，但RNN属于“biased model”，一个句子中越往后的词重要性越高，这有可能影响最后的分类结果，因为对句子分类影响最大的词可能处在句子任何位置。
    CNN属于无偏模型，能够通过最大池化获得最重要的特征，但是CNN的滑动窗口大小不容易确定，选的过小容易造成重要信息丢失，选的过大会造成巨大参数空间。
    为了解决二者的局限性，这篇文章提出了一种新的网络架构，用双向循环结构获取上下文信息，这比传统的基于窗口的神经网络更能减少噪声，而且在学习文本表达时可以大范围的保留词序。
    其次使用最大池化层获取文本的重要部分，自动判断哪个特征在文本分类过程中起更重要的作用。
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.model_name = f"{config.model_name.lower()}-{config.embedding_dim}"
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=config.bidirectional,
                            batch_first=True,
                            dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embedding_dim, config.num_classes)

    def forward(self, text):
        # text = [batch size, seq len]
        embed = self.embedding(text)  # embed = [batch size, seq len, embedding dim]=[64, 32, 64]
        out, _ = self.lstm(embed)  # out = [batch size, seq len, hidden size * 2]
        out = torch.cat((embed, out), 2)  # out = [batch size, seq len, embedding dim + hidden size * 2]
        out = F.relu(out)  # out = [batch size, seq len, embedding dim + hidden size * 2]
        out = out.permute(0, 2, 1)  # out = [batch size, embedding dim + hidden size * 2, seq len]
        out = F.max_pool1d(out, out.shape[2]).squeeze(2)  # out = [batch size, embedding dim + hidden size * 2]
        out = self.fc(out)  # out = [batch size, num_classes]
        return out
