import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils import load_data, data_set, predict, plotACC, WORD_COUNT

MAX_LENGTH = WORD_COUNT + 2
SOS_token = 0
EOS_token = WORD_COUNT + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        # input_size: 输入语言中包含的词个数
        # hidden_size: 每个词embed为hidden_size维的向量
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        # output_size: 输出语言包含的所有单词数
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, MAX_LENGTH)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # 把256(hidden_size)个特征转换为输出语言的词汇个数
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        Parameters:
        -------------
        input: 每一步的输入
        hidden: 上一步结果
        encoder_outputs: 编码的状态矩阵

        Returns:
        ------------
        output: 各词出现的概率
        """
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


class Trainer:
    def __init__(self, encoder, decoder, learning_rate=0.01, device=device):
        self.encoder = encoder
        self.decoder = decoder
        self.optimE = optim.SGD(encoder.parameters(), lr=learning_rate)
        self.optimD = optim.SGD(decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()
        self.device = device

    def train(self, data, n_iters, plot_every=100):
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        selectedIndices = np.random.choice(len(data), n_iters, replace=False)

        for times, ind in enumerate(selectedIndices):
            input_tensor = data[ind].to(device=self.device)
            target_tensor = input_tensor

            loss = self.train_once(input_tensor, target_tensor)
            print_loss_total += loss
            plot_loss_total += loss
            print(loss)
            if (times + 1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    def train_once(self, input_tensor, target_tensor):
        """每调用一次train训练一个句子。"""
        encoder_hidden = self.encoder.initHidden()

        self.optimE.zero_grad()
        self.optimD.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            MAX_LENGTH, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):  # 每次传入序列中的一个元素
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            # seq_len为1, batch_size为1,大小为hidden_size
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor(
            [[SOS_token]], device=self.device)  # SOS为标记句首

        decoder_hidden = encoder_hidden  # 把编码的最终状态作为解码的初始状态

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):  # 每次预测一个元素
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)  # 将可能性最大的预测值加入译文序列
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        self.optimE.step()
        self.optimD.step()

        return loss.item() / target_length


if __name__ == '__main__':
    data = load_data("../source/corpus.txt")
    for i, text in enumerate(data):
        data[i] = torch.cat([text+1, torch.tensor([EOS_token])])
        data[i].unsqueeze_(-1)
    hidden_size = 256
    encoder = Encoder(MAX_LENGTH, hidden_size).to(device)
    decoder = AttnDecoder(hidden_size, MAX_LENGTH, 0.1).to(device)
    trainer = Trainer(encoder, decoder, 0.001, device)

    # trainer.train(data, 7500)
