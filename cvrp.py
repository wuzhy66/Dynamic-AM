#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
import copy
import torch as t
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import time
import numpy as np

if t.cuda.is_available():
    DEVICE = t.device('cuda:0')
else:
    DEVICE = t.device('cpu')

# 设置种子
t.manual_seed(0)
random.seed(0)
np.random.seed(0)

# 向量大小
embedding_size = 128
# 结点规模
city_size = 21
# 批次大小
batch = 128
# 每周期批次数
times = 10000
# 训练周期数
epochs = 50
# 验证批次数
tepoch = 10
min = 10000
# 容量大小
l = 30
# 多头注意力头数
M = 8
# 超参数
C = 10
K = 10
nan = 0

# 用于数据查找
mask_size = t.LongTensor(batch).to(DEVICE)
for i in range(batch):
    mask_size[i] = city_size * i


class act_net(nn.Module):
    def __init__(self):
        super().__init__()
        # embedding参数
        self.embedding = nn.Linear(3, embedding_size)
        self.embedding_p = nn.Linear(2, embedding_size)

        # 第一层参数
        self.wq1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wk1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wv1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w1 = nn.Linear(embedding_size, embedding_size, bias=False)

        # 第二层参数
        self.wq2 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wk2 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wv2 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w2 = nn.Linear(embedding_size, embedding_size, bias=False)

        # 第三层参数
        self.wq3 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wk3 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wv3 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w3 = nn.Linear(embedding_size, embedding_size, bias=False)

        # 解码器参数
        self.wq = nn.Linear(embedding_size * 2 + 1, embedding_size, bias=False)
        self.wk = nn.Linear(embedding_size, embedding_size, bias=False)
        self.wv = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w = nn.Linear(embedding_size, embedding_size, bias=False)

        self.q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.k = nn.Linear(embedding_size, embedding_size, bias=False)

        # 全连接层参数
        self.fw1 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb1 = nn.Linear(embedding_size * 4, embedding_size)

        self.fw2 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb2 = nn.Linear(embedding_size * 4, embedding_size)

        self.fw3 = nn.Linear(embedding_size, embedding_size * 4)
        self.fb3 = nn.Linear(embedding_size * 4, embedding_size)

    def forward(self, s, d, l, train):
        # if train == 2:
        #     for i in self.parameters():
        #         i.requires_grad = True
        # else:
        #     for i in self.parameters():
        #         i.requires_grad = False

        # 计算距离矩阵
        s1 = t.unsqueeze(s, dim=1)
        s1 = s1.expand(batch, city_size, city_size, 2)
        s2 = t.unsqueeze(s, dim=2)
        s2 = s2.expand(batch, city_size, city_size, 2)
        ss = s1 - s2
        dis = t.norm(ss, 2, dim=3, keepdim=True)

        # 存贮概率值
        pro = t.FloatTensor(batch, city_size * 2).to(DEVICE)
        # 存储解序列
        seq = t.LongTensor(batch, city_size * 2).to(DEVICE)
        # 上一时刻访问结点
        index = t.LongTensor(batch).to(DEVICE)
        # 标记是否被访问
        tag = t.ones(batch * city_size).to(DEVICE)
        # 已行驶总距离
        distance = t.zeros(batch).to(DEVICE)
        # 剩余容量
        rest = t.LongTensor(batch, 1, 1).to(DEVICE)

        dd = t.LongTensor(batch, city_size).to(DEVICE)
        rest[:, 0, 0] = l
        dd[:, :] = d[:, :, 0]
        index[:] = 0
        sss = t.cat([s, d.float()], dim=2)
        node = self.embedding(sss)
        # node[:, 0, :] = self.embedding_p(s[:, 0, :])
        # node[0] = self.embedding_p(sss[0])
        x111 = node

        for i in range(city_size * 2):
            flag = t.sum(dd, dim=1)
            f1 = t.nonzero(flag > 0).view(-1)
            f2 = t.nonzero(flag == 0).view(-1)

            mask1 = index == 0
            mask2 = flag > 0
            mask = mask1 * mask2
            f3 = t.nonzero(mask).view(-1)
            node = x111[f3]
            mask = tag.view(batch, -1, 1) < 0.5
            mask[:, 0, 0] = 0
            mask = mask.expand(batch, city_size, M)
            mask = t.unsqueeze(mask, dim=1)
            mask = mask.expand(batch, city_size, city_size, M)

            if (f3.size()[0] != 0):
                # if i == 0:
                numf = f3.size()[0]

                # 第一层
                query1 = self.wq1(node)
                query1 = t.unsqueeze(query1, dim=2)
                query1 = query1.expand(numf, city_size, city_size, embedding_size)
                key1 = self.wk1(node)
                key1 = t.unsqueeze(key1, dim=1)
                key1 = key1.expand(numf, city_size, city_size, embedding_size)
                value1 = self.wv1(node)
                value1 = t.unsqueeze(value1, dim=1)
                value1 = value1.expand(numf, city_size, city_size, embedding_size)
                x = query1 * key1
                x = x.view(numf, city_size, city_size, M, -1)
                x = t.sum(x, dim=4)
                x.masked_fill_(mask[f3], -float('inf'))
                x = F.softmax(x, dim=2)
                x = t.unsqueeze(x, dim=4)
                x = x.expand(numf, city_size, city_size, M, 16)
                x = x.contiguous()
                x = x.view(numf, city_size, city_size, -1)
                x = x * value1
                x = t.sum(x, dim=2)
                x = self.w1(x)

                x = x + x111[f3]
                x = t.tanh(x)
                x1 = self.fw1(x)
                x1 = F.relu(x1)
                x1 = self.fb1(x1)
                x = x + x1
                x = t.tanh(x)
                x1 = x

                # 第二层
                query2 = self.wq2(x)
                query2 = t.unsqueeze(query2, dim=2)
                query2 = query2.expand(numf, city_size, city_size, embedding_size)
                key2 = self.wk2(x)
                key2 = t.unsqueeze(key2, dim=1)
                key2 = key2.expand(numf, city_size, city_size, embedding_size)
                value2 = self.wv2(x)
                value2 = t.unsqueeze(value2, dim=1)
                value2 = value2.expand(numf, city_size, city_size, embedding_size)
                x = query2 * key2
                x = x.view(numf, city_size, city_size, M, -1)
                x = t.sum(x, dim=4)
                x.masked_fill_(mask[f3], -float('inf'))
                x = F.softmax(x, dim=2)
                x = t.unsqueeze(x, dim=4)
                x = x.expand(numf, city_size, city_size, M, 16)
                x = x.contiguous()
                x = x.view(numf, city_size, city_size, -1)
                x = x * value2
                x = t.sum(x, dim=2)
                x = self.w2(x)

                x = x + x1
                x = t.tanh(x)
                x1 = self.fw2(x)
                x1 = F.relu(x1)
                x1 = self.fb2(x1)
                x = x + x1
                x = t.tanh(x)
                x1 = x

                # 第三层
                query3 = self.wq3(x)
                query3 = t.unsqueeze(query3, dim=2)
                query3 = query3.expand(numf, city_size, city_size, embedding_size)
                key3 = self.wk3(x)
                key3 = t.unsqueeze(key3, dim=1)
                key3 = key3.expand(numf, city_size, city_size, embedding_size)
                value3 = self.wv3(x)
                value3 = t.unsqueeze(value3, dim=1)
                value3 = value3.expand(numf, city_size, city_size, embedding_size)
                x = query3 * key3
                x = x.view(numf, city_size, city_size, M, -1)
                x = t.sum(x, dim=4)
                x.masked_fill_(mask[f3], -float('inf'))
                x = F.softmax(x, dim=2)
                x = t.unsqueeze(x, dim=4)
                x = x.expand(numf, city_size, city_size, M, 16)
                x = x.contiguous()
                x = x.view(numf, city_size, city_size, -1)
                x = x * value3
                x = t.sum(x, dim=2)
                x = self.w3(x)

                x = x + x1
                x = t.tanh(x)
                x1 = self.fw3(x)
                x1 = F.relu(x1)
                x1 = self.fb3(x1)
                x = x + x1
                x = t.tanh(x)

                feature = t.ones(batch, city_size, embedding_size).to(DEVICE)
                if i != 0:
                    feature[:, :, :] = features[:, :, :]
                feature[f3, :, :] = x[:, :, :]
                x = feature
                features = feature
                mask = mask[:, 0, :, 0]
                mask = mask.view(batch, city_size, 1) == 0
                mask = mask.expand(batch, city_size, embedding_size)

                avg = t.ones(batch, embedding_size).to(DEVICE)
                if i != 0:
                    avg[:, :] = avgs[:, :]
                avgt = x[f3] * mask.float()[f3]
                avgt = t.sum(avgt, dim=1)
                r_city = t.sum(mask[f3, :, 0], dim=1)
                r_city = t.unsqueeze(r_city, dim=1)
                r_city = r_city.expand(numf, embedding_size)
                avgt = avgt / r_city.float()
                avg[f3, :] = avgt[:, :]
                avgs = avg

            # 解码过程
            if f1.size()[0] == 0:
                pro[:, i:] = 1
                seq[:, i:] = 0
                temp = dis.view(-1, city_size, 1)[index + mask_size]
                distance = distance + temp.view(-1)[mask_size]
                break

            ind = index + mask_size
            tag[ind] = 0
            # depot = x.view(-1, embedding_size)[mask_size]
            start = x.view(-1, embedding_size)[ind]

            end = rest[:, :, 0]
            end = end.float()

            graph = t.cat([avg, start, end], dim=1)
            query = self.wq(graph)
            query = t.unsqueeze(query, dim=1)
            query = query.expand(batch, city_size, embedding_size)
            key = self.wk(x)
            value = self.wv(x)
            temp = query * key
            temp = temp.view(batch, city_size, M, -1)
            temp = t.sum(temp, dim=3)

            mask = tag.view(batch, -1, 1) < 0.5
            mask1 = dd.view(batch, city_size, 1) > rest.expand(batch, city_size, 1)
            flag = t.nonzero(index).view(-1)
            mask = mask + mask1
            mask = mask > 0
            mask[f2, 0, 0] = 0
            if flag.size()[0] > 0:
                mask[flag, 0, 0] = 0

            mask = mask.expand(batch, city_size, M)
            temp.masked_fill_(mask, -float('inf'))
            temp = F.softmax(temp, dim=1)
            temp = t.unsqueeze(temp, dim=3)
            temp = temp.expand(batch, city_size, M, 16)
            temp = temp.contiguous()
            temp = temp.view(batch, city_size, -1)
            temp = temp * value
            temp = t.sum(temp, dim=1)
            temp = self.w(temp)

            query = self.q(temp)
            key = self.k(x)
            query = t.unsqueeze(query, dim=1)
            query = query.expand(batch, city_size, embedding_size)
            temp = query * key
            temp = t.sum(temp, dim=2)
            temp = t.tanh(temp) * C

            # mask = tag.view(batch, -1) < 0.5
            mask = mask[:, :, 0]
            temp.masked_fill_(mask, -float('inf'))
            p = F.softmax(temp, dim=1)

            indexx = t.LongTensor(batch).to(DEVICE)
            if train != 0:
                indexx[f1] = t.multinomial(p[f1], 1)[:, 0]
            else:
                indexx[f1] = (t.max(p[f1], dim=1)[1])

            indexx[f2] = 0
            mask3 = indexx == 0
            mask3 = mask3.view(batch, 1, 1)
            rest.masked_fill_(mask3, l)

            p = p.view(-1)
            pro[:, i] = p[indexx + mask_size]
            pro[f2, i] = 1
            rest = rest - (dd.view(-1)[indexx + mask_size]).view(batch, 1, 1)
            dd = dd.view(-1)
            dd[indexx + mask_size] = 0
            dd = dd.view(batch, city_size)

            temp = dis.view(-1, city_size, 1)[index + mask_size]
            distance = distance + temp.view(-1)[indexx + mask_size]

            index = indexx
            seq[:, i] = index[:]

        # temp = dis.view(-1, city_size, 1)[index + mask_size]
        # distance = distance + temp.view(-1)[mask_size]
        if train == 0:
            seq = seq.detach()
            pro = pro.detach()
            distance = distance.detach()
        return seq, pro, distance


# net = act_net()
net = t.load('vrp20.pth')
# for i in net.parameters():
#    i.requires_grad = True

net = net.to(DEVICE)
opt = optim.Adam(net.parameters(), 0.0001)

tS = t.rand(batch * tepoch, city_size, 2)
tD = np.random.randint(1, 10, size=(batch * tepoch, city_size, 1))
tD = t.LongTensor(tD)
S = t.rand(batch * times, city_size, 2)
D = np.random.randint(1, 10, size=(batch * times, city_size, 1))
D = t.LongTensor(D)
tD[:, 0, 0] = 0
D[:, 0, 0] = 0

for epoch in range(epochs):
    for i in range(times):
        # t.cuda.empty_cache()
        s = S[i * batch: (i + 1) * batch]
        d = D[i * batch: (i + 1) * batch]
        s = s.to(DEVICE)
        d = d.to(DEVICE)

        t1 = time.time()
        seq2, pro2, dis2 = net(s, d, l, 0)
        seq1, pro1, dis1 = net(s, d, l, 2)
        t2 = time.time()
        print(t2 - t1)

        pro = t.log(pro1)
        loss = t.sum(pro, dim=1)
        score = dis1 - dis2

        score = score / dis2
        bias = t.mean(dis2)
        score = score * bias

        score = score.detach()
        loss = score * loss
        loss = t.sum(loss) / batch

        opt.zero_grad()
        loss.backward()

        flag = t.isnan(net.wq1.weight.grad)
        if t.sum(flag) != 0:
            print('nan', epoch, i, t.sum(flag))
            nan += 1
            opt.zero_grad()

        nn.utils.clip_grad_norm_(net.parameters(), 1)
        opt.step()
        # 当前周期，批次，贪婪策略平均距离，采样距离平均距离，方差，均值
        print(epoch, i, t.mean(dis1), t.mean(dis2), t.mean((dis1 - dis2) * (dis1 - dis2)), t.mean(t.abs(dis1 - dis2)),
              nan)

        if (i + 1) % 100 == 0:
            length = t.zeros(1).to(DEVICE)
            for j in range(tepoch):
                t.cuda.empty_cache()
                s = tS[j * batch: (j + 1) * batch]
                d = tD[j * batch: (j + 1) * batch]
                s = s.to(DEVICE)
                d = d.to(DEVICE)
                seq, pro, dis = net(s, d, l, 0)
                length = length + t.mean(dis)
            length = length / tepoch
            if length < min:
                t.save(net, 'vrp20_0.pth')
                min = length
            print(min, length)
