# 图片说明

1. ./cnn 为最早期使用 cnn 模型的作品, 此时尚不会 rnn, 更不会 embedding
2. ./no embedding 文件夹为用 rnn 早期未加入 embedding layer 时的作品
3. BS：batch size, 均为 64 组数据为一批, 为什么是 64 是因为在用 cnn 解题时得出的结论
4. 0.9: 训练集占标注数据(train.txt)的比例, 即训练/测试集划分比; 未标明 0.9 的为之前按绝对数量设置的训练集容量, 200000, 可参见`cnn.py`
5. conv 开头的指以双通道形式拼接文本, 以卷积进行特征提取; linear 开头的指直接对两个文本向量拼接, 以全连接层进行特征提取。
6. lr: learning rate, 未注明的为 0.001 的学习率
7. L5/L4: 5 nn layers / 4 nn layers(rnn is always one layer)
8. rnn1 / rnn2 / rnn3: rnn 内部设置的层数, 1/2/3 layers
9. dropout: 表明添加了 dropout 层, 0.5/0.1 表示随机舍弃神经元的概率; dropout(embedding 层)表示在 embedding 层之后加入的 dropout, 未作如此命名的为在最后一层全连接前加入的 dropout
10. embed256: embedding 空间的维度, 即嵌入为多少 256 维的向量
11. hidden256: 隐状态向量设置为 256 维
12. bidirection: rnn 层传播方向设为双向传播
13. rnn256_512_2: 最终提交版结果, rnn 算法, embed 为 256 维, hidden 为 512 维, 2 层 rnn, 若有 sampleCorpus 表明其是从 corpus.txt 进行了采样的输出结果
