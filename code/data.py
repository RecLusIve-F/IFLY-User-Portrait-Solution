import torch
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text, sequence


def process_time(x, max_len):
    x_l = len(x)
    if x_l < max_len:
        x = x + [0] * (max_len - x_l)
    elif x_l > max_len:
        x = x[:max_len]
    return x


def get_data(max_len):
    nan_data = np.load('./data/filled_data.npy')
    train = pd.read_csv('./data/train-update.txt', header=None,
                        names=['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
    test = pd.read_csv('./data/test-update.txt', header=None,
                       names=['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])

    data = pd.concat([train, test])
    data = data.reset_index(drop=True)
    data["age"] = nan_data[:, 1]
    data["gender"] = nan_data[:, 0]
    data['tagid'] = data['tagid'].fillna('[-1]')
    data['city'] = data['city'].fillna("kong")
    data['province'] = data['province'].fillna("kong")
    data['time'] = data['time'].fillna("[1876780786905.0]")

    data['tagid'] = data['tagid'].apply(lambda x: [str(i) for i in eval(x)])
    data['time'] = data['time'].apply(lambda x: [float(i) / 1876780786905.0 for i in eval(x)])

    data['tagid_f'] = data['tagid'].apply(lambda x: x[:max_len])
    data['tagid_b'] = data['tagid'].apply(lambda x: x[-max_len:])

    tagid = data['tagid']
    tagid_f = data['tagid_f']
    tagid_b = data['tagid_b']
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(tagid))
    tagid_f = tokenizer.texts_to_sequences(tagid_f)
    tagid_f = sequence.pad_sequences(tagid_f, maxlen=max_len, padding='post', truncating='post')
    tagid_b = tokenizer.texts_to_sequences(tagid_b)
    tagid_b = sequence.pad_sequences(tagid_b, maxlen=max_len, padding='post', truncating='post')

    data['tagid_f'] = tagid_f.tolist()
    data['tagid_b'] = tagid_b.tolist()

    data['time_len'] = data['time'].apply(lambda x: len(x))
    data['time_sum'] = data['time'].apply(lambda x: sum(x))
    data['time_nunique'] = data['time'].apply(lambda x: np.unique(x).shape[0])
    for feature in ['time_len', 'time_sum', 'time_nunique']:
        data[feature] = data[feature] / (data[feature].max() - data[feature].min())
    data['time_avg'] = data['time'].apply(lambda x: np.mean(x))
    data['time_f'] = data['time'].apply(lambda x: process_time(x[:max_len], max_len))
    data['time_b'] = data['time'].apply(lambda x: process_time(x[-max_len:], max_len))

    city_unique = data['city'].unique()
    data['city'] = data['city'].map({city_unique[i]: i + 1 for i in range(city_unique.shape[0])})
    province_unique = data['province'].unique()
    data['province'] = data['province'].map({province_unique[i]: i + 1 for i in range(province_unique.shape[0])})

    word_index = tokenizer.word_index
    VOCAB_SIZE = len(word_index) + 1
    CITY_VOCAB_SIZE = data['city'].max() + 1
    PROVINCE_VOCAB_SIZE = data['province'].max() + 1

    train = data[~data['label'].isna()]
    test = data[data['label'].isna()]
    y = train['label'].values

    features = ['gender', 'age', 'time_len', 'time_sum', 'time_nunique', 'time_avg']

    X_train_tagid_f, X_test_tagid_f = torch.LongTensor(train["tagid_f"].to_list()), torch.LongTensor(
        test['tagid_f'].to_list())
    X_train_tagid_b, X_test_tagid_b = torch.LongTensor(train["tagid_b"].to_list()), torch.LongTensor(
        test['tagid_b'].to_list())

    X_train_cityid, X_test_cityid = torch.LongTensor(train["city"].to_list()), torch.LongTensor(test["city"].to_list())
    X_train_provinceid, X_test_provinceid = torch.LongTensor(train["province"].to_list()), torch.LongTensor(
        test["province"].to_list())

    X_train_time_f, X_test_time_f = torch.Tensor(train["time_f"].to_list()), torch.Tensor(test["time_f"].to_list())
    X_train_time_b, X_test_time_b = torch.Tensor(train["time_b"].to_list()), torch.Tensor(test["time_b"].to_list())

    X_train_onehot, X_test_onehot = torch.LongTensor(train[features].values.tolist()), torch.LongTensor(
        test[features].values.tolist())

    y = torch.LongTensor(y)

    X_train = [X_train_tagid_f, X_train_tagid_b, X_train_cityid, X_train_provinceid, X_train_time_f, X_train_time_b,
               X_train_onehot, y]
    X_test = [X_test_tagid_f, X_test_tagid_b, X_test_cityid, X_test_provinceid, X_test_time_f, X_test_time_b,
              X_test_onehot]

    return X_train, X_test, VOCAB_SIZE, CITY_VOCAB_SIZE, PROVINCE_VOCAB_SIZE
