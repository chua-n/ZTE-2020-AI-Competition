import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from utils import load_data, data_set, plotACC


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(2, 4, 3), nn.ReLU())  # 32 -> 30
        self.conv2 = nn.Sequential(nn.Conv1d(4, 8, 3), nn.ReLU())  # 30 -> 28
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, 3), nn.AvgPool1d(3), nn.ReLU())  # 28 -> 26 -> 8
        self.conv4 = nn.Sequential(
            nn.Conv1d(16, 32, 2), nn.AvgPool1d(3), nn.ReLU())  # 8 -> 7 -> 2
        self.conv5 = nn.Sequential(nn.Conv1d(32, 64, 2), nn.ReLU())  # 2 -> 1
        self.fc1 = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        x = nn.Sequential(self.conv1, self.conv2, self.conv3,
                          self.conv4, self.conv5)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x.squeeze()

    def predict(self, file="../XuChuanyi_NJU_predict.txt",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.eval()
        text_a, text_b = load_data("../source/test.txt")
        text_a.append(torch.zeros(32, dtype=torch.int))
        text_b.append(torch.zeros(32, dtype=torch.int))
        test_set = data_set(text_a, text_b)

        text_a, text_b, *_ = test_set[:-1]
        text_a.squeeze_(-1)
        text_b.squeeze_(-1)
        text_a.unsqueeze_(1)
        text_b.unsqueeze_(1)
        x = torch.cat((text_a, text_b), dim=1).to(
            dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = self.forward(x)
        with open(file, 'w') as obj:
            for label in pred:
                label = 1 if label.item() > 0.5 else 0
                label = str(label) + '\n'
                obj.write(label)
        return


def train_cnn():
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = 64
    LR = 0.001
    EPOCH = 30
    data = data_set(*load_data())
    boundary = 200000
    train_set = DataLoader(TensorDataset(*data[:boundary]), BS, shuffle=True)
    eval_set = DataLoader(TensorDataset(*data[boundary:]), 2*BS, shuffle=False)
    cnn = CNN().to(device=device)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(cnn.parameters(), lr=LR)
    begin = time.time()
    ACC_train, ACC = [], []
    for epoch in range(EPOCH):
        cnn.train()
        acc_train = 0
        for i, (a, b, y, *_) in enumerate(train_set):
            a.squeeze_(-1)
            b.squeeze_(-1)
            a.unsqueeze_(1)
            b.unsqueeze_(1)
            x = torch.cat((a, b), dim=1).to(dtype=torch.float32, device=device)
            y = y.to(dtype=torch.float32, device=device)

            optim.zero_grad()
            y_pre = cnn(x)
            loss = loss_fn(y_pre, y)
            loss.backward()
            optim.step()
            y_pre[y_pre > 0.5] = 1
            y_pre[y_pre <= 0.5] = 0
            acc_train += len(torch.nonzero(y == y_pre))

            if (i + 1) % 500 == 0 or (i + 1) == len(train_set):
                time_cost = int(time.time() - begin)
                print('Time cost so far: {}h {}min {}s'.format(
                    time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
                print("Epoch[{}/{}], Step [{}/{}], Loss {:.4f}".
                      format(epoch + 1, EPOCH, i + 1, len(train_set), loss.item()))
        acc_train /= len(train_set.dataset)
        ACC_train.append(acc_train)

        cnn.eval()
        with torch.no_grad():
            eval_loss, acc = 0, 0
            for i, (a, b, y, *_) in enumerate(eval_set):
                a.squeeze_(-1)
                b.squeeze_(-1)
                a.unsqueeze_(1)
                b.unsqueeze_(1)
                x = torch.cat((a, b), dim=1).to(
                    dtype=torch.float32, device=device)
                y = y.to(dtype=torch.float32, device=device)

                y_pre = cnn(x)
                eval_loss += loss_fn(y_pre, y)
                y_pre[y_pre > 0.5] = 1
                y_pre[y_pre <= 0.5] = 0
                acc += len(torch.nonzero(y == y_pre))
            eval_loss /= len(eval_set)
            acc /= len(eval_set.dataset)
            ACC.append(acc)
            time_cost = int(time.time() - begin)
            print('\nTime cost so far: {}h {}min {}s'.format(
                time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
            print('Evaluation set: loss: {:.4f}, acc: {:.4f}\n'.format(
                eval_loss, acc))
            if acc > 0.8:
                cnn.predict(device=device)
    return cnn, ACC_train, ACC


if __name__ == '__main__':
    cnn, ACC_train, ACC = train_cnn()
    plotACC(ACC_train, ACC)
    cnn.predict()
