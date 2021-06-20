import time
from importlib import import_module

import spacy
import torch
import torch.nn as nn

from train_eval import evaluate
from utils import load_data, get_text_field, get_label_field, get_fields, \
    build_vocab_label, build_vocab_text, get_iterator


def predict_sentiment(text_field, model, sentence, min_len=5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [text_field.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.argmax(model(tensor), dim=1)
    return prediction.item()

def run_eval(model_name, embedding_dim):
    text_field = get_text_field(model_name)
    label_field = get_label_field()
    fields = get_fields(text_field, label_field)
    train_data, valid_data, test_data = load_data(fields)
    build_vocab_label(label_field, train_data)
    train_iterator, valid_iterator, test_iterator = get_iterator(train_data, valid_data, test_data, device)

    x = import_module('models.' + model_name)

    if model_name != 'BERTGRU':
        build_vocab_text(text_field, train_data, embedding_dim)
        config = x.Config(text_field, embedding_dim, model_name)
        model = x.Model(config)
    else:
        # TODO: change input
        config = x.Config()
        model = x.Model(config)

    model.load_state_dict(torch.load(f'{model.model_name}-model.pt'))
    criterion = nn.CrossEntropyLoss()  # For multi-class
    criterion = criterion.to(device)

    stime = time.time()
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
    running_time = time.time() - stime
    print(
        f"Model: {model_name}, Embedding Dimension: {embedding_dim}, ACC: {valid_acc:.4f}, Time: {running_time:.3f} sec")

    # print(predict_sentiment(text_field, model, "This film is terrible"))


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in ['TextCNN', 'TextCNN1D', 'TextRNN', 'TextRCNN', 'TextRNNAtt']:
        for embedding_dim in [100, 200, 300]:
            run_eval(model_name, embedding_dim)
