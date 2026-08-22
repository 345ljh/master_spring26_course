import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def us_read(file_name):
    # 读取文件
    with open(file_name, 'r') as f:
        lines = f.readlines()
        # 文件每4行为一帧采样
        # 每个文件采样600帧, 采样深度500, 即保存为(600, 4, 500)的ndarray
        # print(lines[0].replace('\n').split(' '))
        data = np.zeros((600, 4, 500))
        for line_idx in range(len(lines) - 1):
            line = lines[line_idx].replace('\n', '').split(' ')[2:]
            data[line_idx // 4, line_idx % 4, :] = np.array(line, dtype=np.float32)
        # 每帧每通道归一化
        for i in range((len(lines) - 1) // 4):
            for j in range(4):
                data[i, j, :] = (data[i, j, :] - np.min(data[i, j, :])) / (np.max(data[i, j, :]) - np.min(data[i, j, :]))

        # # 绘制通道0的波形, 横轴为时间(axis=0), 纵轴为深度(axis=2), 颜色为值大小
        # plt.imshow(data[:, 0, :], aspect='auto', cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        # plt.plot(data[100, 0, :])
        # plt.show()

        # 保存pickle
        with open(fr'processed/{file_name.split("-")[1]}.pkl', 'wb') as f:
            pickle.dump(data, f)

def us_readall(folder_name):
    us_data = {}
    for filename in os.listdir(f'{folder_name}'):
        if filename.endswith(".txt"):
            print(f'read file: {filename}')
            # 读取文件
            with open(f'{folder_name}/{filename}', 'r') as f:
                lines = f.readlines()
                # 文件每4行为一帧采样
                # 每个文件采样600帧, 采样深度500, 即保存为(600, 4, 500)的ndarray
                # print(lines[0].replace('\n').split(' '))
                data = np.zeros((600, 4, 500))
                for line_idx in range(len(lines) - 1):
                    line = lines[line_idx].replace('\n', '').split(' ')[2:]
                    data[line_idx // 4, line_idx % 4, :] = np.array(line, dtype=np.float32)
                # 每帧每通道归一化
                for i in range((len(lines) - 1) // 4):
                    for j in range(4):
                        data[i, j, :] = (data[i, j, :] - np.min(data[i, j, :])) / (np.max(data[i, j, :]) - np.min(data[i, j, :]))

            if filename.split("-")[1] in us_data:
                us_data[filename.split("-")[1]] = np.concatenate((us_data[filename.split("-")[1]], data))
            else:
                us_data[filename.split("-")[1]] = data
        
    print(us_data)
    # 保存pickle
    with open(fr'processed/all.pkl', 'wb') as f:
        pickle.dump(us_data, f)


class US_Model(nn.Module):
    def __init__(self):
        super(US_Model, self).__init__()
        self.conv1 = nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16 * 125, 128)
        self.fc2 = nn.Linear(128, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # 输入(n, 4, 500)
        x = self.dropout(F.relu(self.bn1(self.conv1(x)))) # 卷积层 + batchnorm + relu, (n, 4, 500)
        x = F.max_pool1d(x, 2)  # (n, 4, 250)
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))  # (n, 16, 250)
        x = F.max_pool1d(x, 2)  # (n, 16, 125)
        x = x.view(x.size(0), -1)  # (n, 16*125)
        x = self.dropout(F.relu(self.fc1(x)))  # (n, 128)
        x = self.fc2(x)  # (n, 8)
        return x


if __name__ == "__main__":
    ############################# 数据读取 #############################
    # for filename in os.listdir("data"):
    #     if filename.endswith(".txt"):
    #         us_read(f"data/{filename}")

    ############################# 跨样本数据读取 #############################
    # us_readall("all_data")

    ############################# 数据加载 #############################
    # np.random.seed(42)
    # torch.manual_seed(42)
    # postures = []
    # for filename in os.listdir("processed"):
    #     if filename.endswith(".pkl"):
    #         with open(f"processed/{filename}", 'rb') as f:
    #             data = pickle.load(f)
    #             np.random.shuffle(data)
    #             postures.append(data)

    # # 构建训练, 测试集(80/20)
    # train_data = np.zeros((0, 4, 500))
    # train_label = np.zeros((0))  # 可直接用int
    # test_data = np.zeros((0, 4, 500))
    # test_label = np.zeros((0))
    # for i in range(len(postures)):
    #     train_data = np.concatenate((train_data, postures[i][:480, :, :]), axis=0)
    #     train_label = np.concatenate((train_label, np.full((480), i)), axis=0)
    #     test_data = np.concatenate((test_data, postures[i][480:, :, :]), axis=0)
    #     test_label = np.concatenate((test_label, np.full((120), i)), axis=0)

    ############################# 跨样本数据加载 #############################
    np.random.seed(42)
    torch.manual_seed(42)
    with open(f"all_processed/all.pkl", 'rb') as f:
        data = pickle.load(f)
        for data_key in data:
            np.random.shuffle(data[data_key])

    # 构建训练, 测试集(80/20)
    train_data = np.zeros((0, 4, 500))
    train_label = np.zeros((0))  # 可直接用int
    test_data = np.zeros((0, 4, 500))
    test_label = np.zeros((0))
    i = 0
    for data_key in data:
        train_data = np.concatenate((train_data, data[data_key][:3360, :, :]), axis=0)
        train_label = np.concatenate((train_label, np.full((3360), i)), axis=0)
        test_data = np.concatenate((test_data, data[data_key][3360:, :, :]), axis=0)
        test_label = np.concatenate((test_label, np.full((840), i)), axis=0)
        i += 1

    ############################# 模型训练 #############################
    # 转tensor并构建数据集
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_label, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # 训练
    model = US_Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(100):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 预测
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            print(f'Predicted: {predicted}, Labels: {batch_labels}')
            print('Accuracy: ', accuracy_score(batch_labels, predicted))
            print('Confusion Matrix: ', confusion_matrix(batch_labels, predicted))