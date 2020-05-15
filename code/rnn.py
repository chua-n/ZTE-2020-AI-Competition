import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from utils import load_data, sample_corpus, data_set, predict, plotACC, Timer, ThresholdTester, WORD_COUNT


class RNN(nn.Module):
    def __init__(self, embedded_size, hidden_size, num_layers=1, num_directions=1, dropout_p=0.5, batch_first=True):
        super().__init__()
        self.bidir = True if num_directions == 2 else False
        self.embed = nn.Embedding(
            WORD_COUNT+1, embedded_size, padding_idx=WORD_COUNT)
        self.rnn = nn.LSTM(embedded_size, hidden_size,
                           num_layers, batch_first=batch_first, bidirectional=self.bidir)  # embedded_size -> hidden_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 4, 3, bias=False), nn.BatchNorm1d(4), nn.ReLU())   # hidden_size -> hidden_size-2
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, 3, bias=False),  nn.BatchNorm1d(8), nn.ReLU())  # hidden_size-2 -> hidden_size-4
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, 3, bias=False),  nn.BatchNorm1d(16), nn.ReLU())  # hidden_size-4 -> hidden_size-6
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(16*(num_directions*hidden_size-6), 1), nn.Sigmoid())  # num_dir*hidden_size-6 -> 1

    def forward(self, a, b, la, lb):
        """
        Parameters:
        -----------
        a, b: torch.Tensor, (batch_size, seq_len)
        """
        a = self.embed(a)
        b = self.embed(b)
        a = pack_padded_sequence(
            a, la, batch_first=True, enforce_sorted=False)
        b = pack_padded_sequence(
            b, lb, batch_first=True, enforce_sorted=False)

        out_a, _ = self.rnn(a)
        out_b, _ = self.rnn(b)
        out_a = pad_packed_sequence(out_a, batch_first=True)[0]
        out_b = pad_packed_sequence(out_b, batch_first=True)[0]
        # 只保留最后一刻的状态(使用掩码实现)
        out_a = out_a[torch.arange(out_a.size(0)), la-1, :]
        out_b = out_b[torch.arange(out_b.size(0)), lb-1, :]

        # 使用卷积的情况
        out_a.unsqueeze_(1)
        out_b.unsqueeze_(1)
        x = torch.cat((out_a, out_b), dim=1)
        x = nn.Sequential(self.conv1, self.conv2, self.conv3)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.squeeze()


def prepare_data(bs, rate=0.95, sample=False):
    """Prepare the train-set and evaluation-set for the model.

    Parameters:
    ------------
    bs: batch_size
    rate: 从有标注数据中提取的用作训练集的数据比例

    Returns:
    --------
    train_set, eval_set: DataLoader"""
    import numpy as np
    text_a, text_b, label = load_data("../source/train.txt")
    if sample:
        corpus_a, corpus_b, corpus_label = sample_corpus(
            "../source/corpus.txt")
        text_a.extend(corpus_a)
        text_b.extend(corpus_b)
        label.extend(corpus_label)
    data = data_set(text_a, text_b, label)
    nSamples = int(len(data) * rate)
    selected = np.random.choice(len(data), nSamples, replace=False)
    ind_train = np.zeros(len(data), dtype=np.bool)
    ind_train[selected] = True
    ind_eval = np.ones(len(data), dtype=np.bool)
    ind_eval[selected] = False

    train_set = DataLoader(TensorDataset(*data[ind_train]), bs, shuffle=True)
    eval_set = DataLoader(TensorDataset(*data[ind_eval]), 2*bs, shuffle=False)
    return train_set, eval_set


def eval_rnn(rnn, eval_set, loss_fn, optim, tsd: ThresholdTester):
    """Evaluate the current model capability."""
    rnn.eval()
    with torch.no_grad():
        eval_loss = 0
        for a, b, y, la, lb in eval_set:
            a = a.squeeze(dim=-1).to(device=device)
            b = b.squeeze(dim=-1).to(device=device)
            y = y.to(dtype=torch.float32, device=device)

            y_pre = rnn(a, b, la, lb)
            eval_loss += loss_fn(y_pre, y)
            tsd.add_count(y_pre, y, mode="eval")
        eval_loss /= len(eval_set)
        tsd.cal_acc(mode="eval")
        print('\nLoss of Evaluation Set: {:.4f}'.format(eval_loss))
        tsd.print()
        acc_max = max(tsd.ACC_eval[threshold][-1]
                      for threshold in tsd.thresholds)
        if acc_max > 0.9:
            torch.save({"model_state_dict": rnn.state_dict(),
                        "optim_state_dict": optim.state_dict(),
                        "acc": acc_max}, "../state_dict-conv{:.4f}.rar".format(acc_max))
    return


def train_rnn(device="cuda"):

    BS = 64
    LR = 1e-3
    EPOCH = 30

    loss_fn = nn.BCELoss()
    rnn = RNN(256, 512, 2).to(device=device)
    optim = torch.optim.Adam(rnn.parameters(), lr=LR)

    train_set, eval_set = prepare_data(BS, rate=0.95, sample=False)

    thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    tsd = ThresholdTester(thresholds,
                          len(train_set.dataset), len(eval_set.dataset))

    timer = Timer()
    for epoch in range(EPOCH):
        rnn.train()
        tsd.init()
        for i, (a, b, y, la, lb) in enumerate(train_set):
            a = a.squeeze(dim=-1).to(device=device)
            b = b.squeeze(dim=-1).to(device=device)
            y = y.to(dtype=torch.float32, device=device)

            optim.zero_grad()
            y_pre = rnn(a, b, la, lb)
            loss = loss_fn(y_pre, y)
            loss.backward()
            optim.step()
            tsd.add_count(y_pre, y, mode="train")

            if (i + 1) % 500 == 0 or (i + 1) == len(train_set):
                timer.post_time()
                print("Epoch[{}/{}], Step [{}/{}], Loss {:.4f}".
                      format(epoch + 1, EPOCH, i + 1, len(train_set), loss.item()))
        tsd.cal_acc(mode="train")

        eval_rnn(rnn, eval_set, loss_fn, optim, tsd)
    return rnn, tsd


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn, tsd = train_rnn(device)
    ACC_train, ACC = tsd.ACC_train, tsd.ACC_eval
    fig = plotACC(ACC_train, ACC)
    # threshold = 0.7
    # predict(rnn, "../XuChuanyi_NJU_predict.txt", threshold, device)
