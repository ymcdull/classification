import argparse
from importlib import import_module

import spacy
import torch

from utils import load_data, get_text_field, get_label_field, get_fields, \
    build_vocab_label, build_vocab_text

parser = argparse.ArgumentParser(description='Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
args = parser.parse_args()


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


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model
    text_field = get_text_field(model_name)
    label_field = get_label_field()
    fields = get_fields(text_field, label_field)
    train_data, valid_data, test_data = load_data(fields)
    build_vocab_label(label_field, train_data)

    x = import_module('models.' + model_name)

    if model_name != 'BERTGRU':
        build_vocab_text(text_field, train_data)
        config = x.Config(text_field)
        model = x.Model(config)
    else:
        config = x.Config()
        model = x.Model(config)

    model.load_state_dict(torch.load(f'{model_name.lower()}-model.pt'))
    print(predict_sentiment(text_field, model, "This film is terrible"))
