import os
import spacy
import torch
import pandas as pd
import torch.utils.data as data_utils


DATA_ROOT = '../citizen-lab-data/labeled-data'


def clean_games_ds(verbose=False):
    path = os.path.join(DATA_ROOT, 'games_keywords.csv')
    df = pd.read_csv(path)
    df.drop(df.columns[3:], axis=1, inplace=True)

    df.Theme = df.Theme.str.lower()
    df = df[~df.Theme.isin(['misc', 'tech'])]

    if verbose:
        print('----- GAMES DATASET -----')
        print('Columns:', list(df.columns))
        print('Labels:', df.Theme.unique())
        print('Rows:', df.shape[0])

        for label in df.Theme.unique():
            print('\t', label, len(df[df.Theme.isin([label])]))
        print('\n')

    theme_to_numeric(df)
    df = df.filter(['Keywords', 'Theme'])
    return df.rename({'Keywords': 'Keyword'}, axis=1)


def clean_wechat_ds(verbose=False):
    path = os.path.join(DATA_ROOT, 'wechat_keywords.csv')
    df = pd.read_csv(path)
    df.Theme = df.Theme.str.strip()
    df.Theme = df.Theme.str.lower()

    if verbose:
        print('----- WECHAT DATASET -----')
        print('Columns:', list(df.columns))
        print('Labels:', df.Theme.unique())
        print('Rows:', df.shape[0])

        for label in df.Theme.unique():
            print('\t', label, len(df[df.Theme.isin([label])]))
        print('\n')

    theme_to_numeric(df)
    return df.filter(['Keyword', 'Theme'])


def theme_to_numeric(df):
    df['Theme'].loc[df['Theme'] == 'social'] = 0.
    df['Theme'].loc[df['Theme'] == 'political'] = 1.
    df['Theme'].loc[df['Theme'] == 'people'] = 1.
    df['Theme'].loc[df['Theme'] == 'event'] = 1.


def get_data(batch_size, from_file=False):
    df1 = clean_wechat_ds()
    df2 = clean_games_ds()
    all_data = pd.concat([df1, df2])

    # Embed
    if from_file:
        inputs = torch.load('inputs.pt')
        labels = torch.load('labels.pt')
    else:
        inputs = torch.zeros((all_data.shape[0], 300))  # embedding dim is 300
        labels = torch.zeros(all_data.shape[0])
        nlp = spacy.load('zh_core_web_lg')
        for i in range(all_data.shape[0]):
            inputs[i] = torch.tensor(nlp(all_data.Keyword.values[i]).vector)
            labels[i] = all_data.Theme.values[i]
        labels = torch.tensor(labels, dtype=torch.long)
        torch.save(inputs, 'inputs.pt')
        torch.save(labels, 'labels.pt')

    # dataset = data_utils.TensorDataset(inputs, labels)
    train, valid, test = split_data(inputs, labels)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = data_utils.DataLoader(valid, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def batchify(list_file, batch_size=128):
    f = open(list_file)
    nlp = spacy.load('zh_core_web_lg')
    words = f.readlines()
    embeds = torch.zeros((len(words), 300))  # embedding dim is 300

    for i in range(len(words)):
        word = words[i].strip()
        embeds[i] = torch.tensor(nlp(word).vector)

    data = data_utils.TensorDataset(embeds)
    return data_utils.DataLoader(data, batch_size=batch_size)


def split_data(inputs, labels, train_split=0.8):
    total = len(inputs)
    i1 = int(total * train_split)
    i2 = (total - i1) // 2

    train = data_utils.TensorDataset(inputs[:i1], labels[:i1])
    valid = data_utils.TensorDataset(inputs[i1:i1+i2], labels[i1:i1+i2])
    test = data_utils.TensorDataset(inputs[i1+i2:], labels[i1+i2:])
    return train, valid, test


if __name__ == '__main__':
    clean_games_ds(verbose=True)
    clean_wechat_ds(verbose=True)
