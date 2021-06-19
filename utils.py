import random

import numpy as np
import torch
from torchtext.legacy import data
from transformers import BertTokenizer

SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_text_field(model_name):
    if model_name in {'TextCNN', 'TextCNN1D', 'Transformer', 'TextRNNAtt'}:
        text_field = data.Field(tokenize='spacy',
                                lower=True,
                                tokenizer_language='en_core_web_sm',
                                batch_first=True)
    elif model_name in {'TextRNN'}:
        text_field = data.Field(tokenize='spacy',
                                lower=True,
                                tokenizer_language='en_core_web_sm',
                                include_lengths=True)
    elif model_name in {'TextRCNN'}:
        text_field = data.Field(tokenize='spacy',
                                lower=True,
                                tokenizer_language='en_core_web_sm',
                                batch_first=True)
    elif model_name == 'BERTGRU':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        init_token_idx = tokenizer.cls_token_id
        eos_token_idx = tokenizer.sep_token_id
        pad_token_idx = tokenizer.pad_token_id
        unk_token_idx = tokenizer.unk_token_id
        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:max_input_length - 2]
            return tokens

        text_field = data.Field(batch_first=True,
                                use_vocab=False,
                                tokenize=tokenize_and_cut,
                                preprocessing=tokenizer.convert_tokens_to_ids,
                                init_token=init_token_idx,
                                eos_token=eos_token_idx,
                                pad_token=pad_token_idx,
                                unk_token=unk_token_idx)

    return text_field


def get_label_field():
    return data.LabelField()


def get_fields(text_field, label_field):
    return [(None, None), ('text', text_field), ('label', label_field)]


def load_data(fields, verbose=False):
    train_data, test_data = data.TabularDataset.splits(
        path='data',
        train='train.csv',
        validation='test.csv',
        format='tsv',
        fields=fields,
        skip_header=True
    )

    if verbose:
        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of testing examples: {len(test_data)}')

    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    if verbose:
        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')
        print(f'Number of testing examples: {len(test_data)}')
    return train_data, valid_data, test_data


def build_vocab_text(text_field, train_data):
    text_field.build_vocab(train_data,
                           max_size=MAX_VOCAB_SIZE,
                           vectors="glove.6B.100d",
                           unk_init=torch.Tensor.normal_)


def build_vocab_label(label_field, train_data):
    label_field.build_vocab(train_data)


def get_iterator(train_data, valid_data, test_data, device):
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
        # Use `sort=False` here, otherwise will get TypeError: '<' not supported between instances of 'Example' and 'Example'
        sort=False,
    )
    return train_iterator, valid_iterator, test_iterator


def fix_seed():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
