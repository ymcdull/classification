import argparse
import time
from importlib import import_module

import torch
import torch.nn as nn
from torch import optim

from train_eval import train, evaluate
from utils import fix_seed, load_data, get_iterator, get_text_field, get_label_field, get_fields, \
    build_vocab_label, build_vocab_text

parser = argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding_dim', type=int, required=False, default=100, help='set embedding dimension')
args = parser.parse_args()

if __name__ == '__main__':
    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    model_name = args.model
    embedding_dim = args.embedding_dim
    text_field = get_text_field(model_name)
    label_field = get_label_field()
    fields = get_fields(text_field, label_field)
    train_data, valid_data, test_data = load_data(fields)
    build_vocab_label(label_field, train_data)
    train_iterator, valid_iterator, test_iterator = get_iterator(train_data, valid_data, test_data, device)
    x = import_module('models.' + model_name)

    if model_name not in {'BERTGRU', 'BERT'}:
        # Config
        build_vocab_text(text_field, train_data, embedding_dim)
        config = x.Config(text_field, embedding_dim, model_name)
        model = x.Model(config)
        pretrained_embeddings = text_field.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = text_field.vocab.stoi[text_field.unk_token]
        PAD_IDX = text_field.vocab.stoi[text_field.pad_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
    elif model_name == 'BERTGRU':
        config = x.Config()
        model = x.Model(config)
        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
    elif model_name == 'BERT':
        config = x.Config()
        model = x.Model(config)

    # Train
    optimizer = optim.Adam(model.parameters())
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-class
    criterion = criterion.to(device)
    train(model, train_iterator, valid_iterator, optimizer, criterion)

    # Evaluate
    stime = time.time()
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
    running_time = time.time() - stime
    print(f"Model: {model_name}, Embedding Dimension: {embedding_dim}, ACC: {valid_acc:.4f}, Time: {running_time:.3f} sec")
