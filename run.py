import argparse
from importlib import import_module

import torch
import torch.nn as nn
from torch import optim

from train_eval import train
from utils import fix_seed, load_data, get_iterator, get_text_field, get_label_field, get_fields, \
    build_vocab_label, build_vocab_text

parser = argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
args = parser.parse_args()

if __name__ == '__main__':
    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    model_name = args.model
    text_field = get_text_field(model_name)
    label_field = get_label_field()
    fields = get_fields(text_field, label_field)
    train_data, valid_data, test_data = load_data(fields)
    build_vocab_label(label_field, train_data)
    train_iterator, valid_iterator, test_iterator = get_iterator(train_data, valid_data, test_data, device)
    x = import_module('models.' + model_name)

    if model_name != 'BERTGRU':
        # Config
        build_vocab_text(text_field, train_data)
        config = x.Config(text_field)
        model = x.Model(config)
        pretrained_embeddings = text_field.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = text_field.vocab.stoi[text_field.unk_token]
        PAD_IDX = text_field.vocab.stoi[text_field.pad_token]
        EMBEDDING_DIM = 100

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    else:
        config = x.Config()
        model = x.Model(config)
        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False

    optimizer = optim.Adam(model.parameters())
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-class
    # criterion = nn.BCEWithLogitsLoss()  # For binary
    criterion = criterion.to(device)
    train(model, train_iterator, valid_iterator, optimizer, criterion, model_name)
