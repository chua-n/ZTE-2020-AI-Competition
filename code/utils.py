import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import time
import matplotlib.pyplot as plt


# total number of words in vocabulary(including "train/test/corpus.txt")
WORD_COUNT = 51158

POSITIVE = 143229  # number of positive samples in "train.txt"
NEGATIVE = 104771  # number of negative samples in "train.txt"


def load_data(file="../source/train.txt"):
    text_a, text_b, label = [], [], []
    with open(file, 'r') as file_obj:
        for line in file_obj:
            line = line[:-1].split('\t')
            line = [ele.split() for ele in line]
            text_a.append(torch.LongTensor(tuple(map(int, line[0]))))
            if len(line) > 1:
                text_b.append(torch.LongTensor(tuple(map(int, line[1]))))
            if len(line) > 2:
                label.append(torch.tensor(
                    tuple(map(int, line[2])), dtype=torch.int8))
    if len(text_b) == 0:
        return text_a
    elif len(label) == 0:
        return text_a, text_b
    else:
        return text_a, text_b, label


def sample_corpus(file="../source/corpus.txt"):
    number = 2 * (POSITIVE - NEGATIVE)
    corpus = load_data(file)
    corpus.sort(key=len)
    corpus = corpus[1:1+number]  # corpus[0]只含单个元素，舍弃
    text_a = corpus[::2]
    text_b = corpus[1::2]
    label = [torch.tensor([0], dtype=torch.int8)] * len(text_a)
    return text_a, text_b, label


def data_set(text_a, text_b, label=None):
    length_a = [len(seq) for seq in text_a]
    length_b = [len(seq) for seq in text_b]

    text_a = pad_sequence(text_a, batch_first=True, padding_value=WORD_COUNT)
    text_b = pad_sequence(text_b, batch_first=True, padding_value=WORD_COUNT)
    if label is not None:
        label = torch.tensor(label)
    length_a = torch.tensor(length_a)
    length_b = torch.tensor(length_b)
    text_a.unsqueeze_(-1)
    text_b.unsqueeze_(-1)

    return TensorDataset(text_a, text_b, length_a, length_b) if label is None \
        else TensorDataset(text_a, text_b, label, length_a, length_b)


def predict(model, file="../XuChuanyi_NJU_predict.txt", threshold=0.5, device="cuda"):
    model.eval()
    test_set = data_set(*load_data("../source/test.txt"))

    a, b, la, lb = test_set[:]
    a = a.squeeze(dim=-1).to(device=device)
    b = b.squeeze(dim=-1).to(device=device)

    with torch.no_grad():
        y_pre = model(a, b, la, lb)
    with open(file, 'w') as obj:
        for label in y_pre:
            label = 1 if label.item() > threshold else 0
            label = str(label) + '\n'
            obj.write(label)
    return


def plotACC(ACC_train, ACC_eval):
    fig = plt.figure()
    plt.style.use("seaborn")

    if type(ACC_eval) is dict:
        x = [i+1 for i in range(len(next(iter(ACC_eval.values()))))]
    else:
        x = [i+1 for i in range(len(ACC_eval))]

    if type(ACC_train) is dict:
        for threshold in ACC_train.keys():
            plt.plot(x, ACC_train[threshold], '--',
                     label="Train Set -- "+str(threshold))
    else:
        plt.plot(x, ACC_train, '--r', label="Train Set")

    if type(ACC_eval) is dict:
        for threshold in ACC_eval.keys():
            plt.plot(x, ACC_eval[threshold], ':',
                     label="Evaluation Set -- "+str(threshold))
    else:
        plt.plot(x, ACC_eval, ':g', label="Evaluation Set")

    plt.title("ACC of Train Set And Evaluation Set")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    return fig


class Timer:
    def __init__(self):
        self.init = time.time()

    def post_time(self):
        time_cost = int(time.time() - self.init)
        print('Time cost so far: {}h {}min {}s'.format(
            time_cost // 3600, time_cost % 3600 // 60, time_cost % 3600 % 60 // 1))
        return


class ThresholdTester:
    def __init__(self, thresholds, trainSet_size, evalSet_size):
        self.thresholds = thresholds
        self.trainSet_size = trainSet_size
        self.evalSet_size = evalSet_size
        self.ACC_train = {threshold: [] for threshold in thresholds}
        self.ACC_eval = {threshold: [] for threshold in thresholds}

    def init(self):
        for threshold in self.thresholds:
            self.ACC_train[threshold].append(0)
            self.ACC_eval[threshold].append(0)

    def add_count(self, y_pre, y, mode="train"):
        for threshold in self.thresholds:
            y_pre_copy = y_pre.clone()
            y_pre_copy[y_pre > threshold] = 1
            y_pre_copy[y_pre <= threshold] = 0
            if mode == "train":
                self.ACC_train[threshold][-1] += len(
                    torch.nonzero(y_pre_copy == y))
            elif mode == "eval":
                self.ACC_eval[threshold][-1] += len(
                    torch.nonzero(y_pre_copy == y))
            else:
                raise("Mode Error!")
        return

    def cal_acc(self, mode="train"):
        for threshold in self.thresholds:
            if mode == "train":
                self.ACC_train[threshold][-1] /= self.trainSet_size
            elif mode == "eval":
                self.ACC_eval[threshold][-1] /= self.evalSet_size
            else:
                raise("Mode Error!")

    def print(self):
        for threshold in self.thresholds:
            print("Threshold {:.2f} ---- Train-Set {:.4f}, Eval-Set {:.4f}".format(
                threshold, self.ACC_train[threshold][-1], self.ACC_eval[threshold][-1]))
