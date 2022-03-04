import gc

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.utils.data as Data
from sklearn.metrics import f1_score
from torch.nn import functional as F
from sklearn.model_selection import KFold

from args import make_args
from data import get_data
from Model import FocalLoss, Model


def train_test_model(model, data_loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    y_ture = []
    _loss = 0
    y_pred = []

    for data in data_loader:
        y_ture = y_ture + data[-1].data.numpy().tolist()
        data = [i.to(device) for i in data]
        if train:
            optimizer.zero_grad()
            output = model(*data[:-1])
            loss = criterion(output, data[-1])
            pred = output.cpu().data.max(dim=1)[1].numpy().tolist()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(*data[:-1])
                loss = criterion(output, data[-1])
                pred = output.cpu().data.max(dim=1)[1].numpy().tolist()
        _loss += loss.data.item()
        y_pred += pred
    score = f1_score(y_ture, y_pred)
    return _loss / len(data_loader), score


def main():
    args = make_args()

    # check cuda
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    X_train, X_test, VOCAB_SIZE, CITY_VOCAB_SIZE, PROVINCE_VOCAB_SIZE = get_data(args.max_len)

    oof_f1 = np.zeros(10)
    folds = KFold(n_splits=10, shuffle=True, random_state=2021)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train[0])):
        print(f"fold n{fold_ + 1}")

        X_trn, X_val = [i[trn_idx] for i in X_train], [i[val_idx] for i in X_train]

        train_dataset = Data.TensorDataset(*X_trn)
        val_dataset = Data.TensorDataset(*X_val)

        trn_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
        )

        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
        )

        criterion = FocalLoss()
        model = Model(VOCAB_SIZE, CITY_VOCAB_SIZE, PROVINCE_VOCAB_SIZE, args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
        max_f1 = 0
        patience = 0

        for epoch in tqdm(range(1, args.epoch + 1), desc='  - (Training)  '):
            if 8 <= epoch < 13:
                optimizer = optim.Adam(model.parameters(), lr=args.lr / 5, weight_decay=0.0001)  # , weight_decay=0.01
            elif epoch >= 13:
                optimizer = optim.Adam(model.parameters(), lr=args.lr / 10)  # , weight_decay=0.01
            epoch_loss, epoch_f1 = train_test_model(model, trn_loader, optimizer, criterion, device)
            eval_loss, eval_f1 = train_test_model(model, val_loader, criterion, criterion, False)

            print(f"epoch{'%3d' % epoch} train loss: {'%.8f' % epoch_loss} f1: {'%.8f' % epoch_f1}", end='\t')
            print(f"eval loss: {'%.8f' % eval_loss} f1: {'%.8f' % eval_f1}")
            if eval_f1 > max_f1:
                max_f1 = eval_f1
                patience = 0
                torch.save(model.state_dict(), f'../model/model_{fold_}.pkl')
            elif patience < 4:
                patience += 1
            else:
                oof_f1[fold_] = max_f1
                print(
                    f'-------------Fold{fold_ + 1} cv result: {"%.8f" % (oof_f1[:(fold_ + 1)].mean())}-------------\n')
                break

        test_dataset = Data.TensorDataset(*X_test)
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2,
        )
        model = Model(VOCAB_SIZE, CITY_VOCAB_SIZE, PROVINCE_VOCAB_SIZE, args).to(device)
        outputs_tensor = torch.zeros(size=(100000, 2)).to(device)
        model_nums = 10

        for model_num in range(model_nums):
            predictions = []
            model.load_state_dict(torch.load(f'../model/model_{model_num}.pkl'))
            model.eval()
            for data in tqdm(test_loader, desc='  - (test)  '):
                data = [i.to(device) for i in data]
                with torch.no_grad():
                    output = model(*data)
                    output = F.softmax(output, dim=1)
                    predictions.append(output)
            outputs_tensor += torch.cat(predictions, dim=0)
            gc.collect()

        outputs_tensor = outputs_tensor / model_nums
        outputs = outputs_tensor.data.cpu().numpy()
        np.save(f'./{str(oof_f1.mean()).replace(".", "_")}.npy', outputs, outputs)


if __name__ == '__main__':
    main()
