import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from torchsummary import summary # 总结

# todo 1.定义函数，构建数据集
def creat_dataset():
    data = pd.read_csv('./data/手机价格预测.csv')
    # 取x的特征列（前20列）和y的标签列（最后1列）
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # x转为float32便于求导微分, y是输出整数结果不需要float32
    x = x.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)
    # 优化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# todo 2.搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.output = nn.Linear(512, output_dim)
        #nn.init.kaiming_normal_(self.linear1.weight)
        #nn.init.zeros_(self.linear1.bias)
        #nn.init.kaiming_normal_(self.linear2.weight)
        #nn.init.zeros_(self.linear2.bias)
        #nn.init.kaiming_normal_(self.output.weight)
       #nn.init.zeros_(self.output.bias)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        #nn.CrossEntropyLoss()有softmax,so
        x = self.output(x)
        return x

# todo 3.模型训练
def train(train_dataset, input_dim, output_dim):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    # 训练
    epochs = 300 # 轮数
    # 每轮训练
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0 # 损失值, 批次数
        start = time.time() # 开始时间
        for x, y in train_loader:
            model.train() # 切换状态
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f'轮数: {epoch + 1}, loss: {total_loss / batch_num}, time: {time.time() - start}s')
    #print(f'\n\n\n参数:{model.state_dict()}\n\n\n\n')
    torch.save(model.state_dict(), './model/phone.pth')

# todo 4.模型测试
def evaluate(test_dataset, input_dim, output_dim):
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('./model/phone.pth'))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    correct = 0 # 正确样本个数
    for x, y in test_loader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        #print(f'\n\n\n\n\n{y_pred} {y}\n\n\n\n\n\n')
        correct += (y_pred == y).sum()
    print(f'准确率:{correct / len(test_dataset)}')

# todo 5.测试
if __name__ == '__main__':
    # 1
    train_dataset, test_dataset, input_dim, output_dim = creat_dataset()
    # 2
    #model = PhonePriceModel(input_dim, output_dim)
    #summary(model, input_size=(16, input_dim))
    # 3
    train(train_dataset, input_dim, output_dim)
    # 4
    evaluate(test_dataset, input_dim, output_dim)
