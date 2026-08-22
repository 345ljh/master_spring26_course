import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

######################### SVM #########################

def svm_train(train_data, train_labels, kernel='linear', C=1.0):
    svm_model = SVC(kernel=kernel, C=C)  # C: 正则化参数, 更大会使得分类更准确但可能过拟合
    svm_model.fit(train_data, train_labels)
    return svm_model

def svm_predict(svm_model: SVC, predict_data):
    return svm_model.predict(predict_data)

# 可视化前3维主方向投影特征
def pca_visualize(data, labels):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=80)
    data_pca = pca.fit_transform(data)
    print(np.cumsum(pca.explained_variance_ratio_))
    ax = plt.axes(projection='3d')
    ax.scatter3D(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels)
    
    plt.show()

def svm_process(train_data, train_labels, predict_data, predict_labels):
    pca_visualize(train_data, train_labels)

    acc_score = []
    for i in np.arange(-15, 15, 0.5):
        svm_model = svm_train(train_data, train_labels, kernel='poly', C=2**i)
        predict_result = svm_predict(svm_model, predict_data)
        acc_score.append(accuracy_score(predict_labels, predict_result))
    print(np.max(acc_score))
    plt.plot(np.arange(-15, 15, 0.5), acc_score)
    plt.show()

######################### CNN #########################
class MyDataset(Dataset):
    def __init__(self, data_list, label_list):
        """
        初始化数据集
        
        参数:
            data_list: list of numpy arrays, 每个元素是二维ndarray
            label_list: list of int, 标签值(0,1,2,3)
            transform: 可选的转换函数
        """
        assert len(data_list) == len(label_list), "数据和标签数量必须匹配"
        self.data_list = data_list
        self.label_list = label_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data_list[idx]).float()
        label = torch.tensor(self.label_list[idx], dtype=torch.long)

        data = data.permute(1, 0)
            
        return data, label
    
class EMG_Model(nn.Module):
    def __init__(self):
        super(EMG_Model, self).__init__()
        self.conv1 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64*500, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) # 卷积层 + batchnorm + relu, (n, 16, 4000)
        x = F.max_pool1d(x, 2)  # (n, 16, 2000)
        x = F.relu(self.bn2(self.conv2(x)))  # (n, 32, 2000)
        x = F.max_pool1d(x, 2)  # (n, 32, 1000)
        x = F.relu(self.bn3(self.conv3(x)))  # (n, 64, 1000)
        x = F.max_pool1d(x, 2)  # (n, 64, 500)
        x = x.view(x.size(0), -1)  # (n, 64*500)
        x = F.relu(self.fc1(x))  # (n, 128)
        x = F.relu(self.fc2(x))  # (n, 32)
        x = self.fc3(x)  # (n, 4)
        return x

    # def __init__(self):
    #     super(EMG_Model, self).__init__()
    #     self.conv1 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
    #     self.bn1 = nn.BatchNorm1d(16)
    #     self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
    #     self.bn2 = nn.BatchNorm1d(32)
    #     self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
    #     self.bn3 = nn.BatchNorm1d(64)
    #     self.convr = nn.Conv1d(16, 16, kernel_size=1)
    #     self.fc1 = nn.Linear(48*1000, 128)
    #     self.fc2 = nn.Linear(128, 32)
    #     self.fc3 = nn.Linear(32, 4)
    #     self.dropout = nn.Dropout(0.5)

    # def forward(self, x):
    #     x1 = F.relu(self.bn1(self.conv1(x))) # 卷积层 + batchnorm + relu, (n, 16, 4000)
    #     x1 = F.max_pool1d(x1, 2)  # (n, 16, 2000)
    #     x2 = F.relu(self.bn2(self.conv2(x1)))  # (n, 32, 2000)
    #     x2 = F.max_pool1d(x2, 2)  # (n, 32, 1000)
    #     # x3 = F.relu(self.bn3(self.conv3(x2)))  # (n, 64, 1000)
    #     # x3 = F.max_pool1d(x3, 2)  # (n, 64, 500)

    #     x_ = F.relu(self.convr(x))  # (n, 16, 4000)
    #     x_ = F.avg_pool1d(x_, 4)  # (n, 16, 500)

    #     x = torch.concat([x2, x_], dim=1)  # (n, 80, 500)
    #     x = x.view(x.size(0), -1)  # (n, 80*500)
    #     x = self.dropout(F.relu(self.fc1(x)))  # (n, 128)
    #     x = self.dropout(F.relu(self.fc2(x)))  # (n, 32)
    #     x = self.fc3(x)  # (n, 4)
    #     return x

def cnn_process(train_data, train_labels, predict_data, predict_labels):
        train_dataset = MyDataset(train_data, train_labels)  # (segments, 16, 4000)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 训练
        model = EMG_Model()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(100):
            for batch_data, batch_labels in train_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # 预测
        predict_dataset = MyDataset(predict_data, predict_labels)
        predict_dataloader = DataLoader(predict_dataset, batch_size=len(predict_data), shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in predict_dataloader:
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                print(f'Predicted: {predicted}, Labels: {batch_labels}')
                print('Accuracy: ', accuracy_score(batch_labels, predicted))
                print('Confusion Matrix: ', confusion_matrix(batch_labels, predicted))





if __name__ == '__main__':
    # 固定种子
    np.random.seed(42)
    torch.manual_seed(42)
    # with open('features/T005/emgseries.pkl', 'rb') as f:
    #     loaded_list = pickle.load(f)
    #     train_data1 = np.concatenate([loaded_list[0][0], loaded_list[2][0], loaded_list[4][0]])
    #     train_labels1 = np.concatenate([loaded_list[0][1], loaded_list[2][1], loaded_list[4][1]])
    #     predict_data1 = np.concatenate([loaded_list[1][0], loaded_list[3][0]])
    #     predict_labels1 = np.concatenate([loaded_list[1][1], loaded_list[3][1]])
    with open('features/T004/emgseries.pkl', 'rb') as f:
        loaded_list = pickle.load(f)
        train_data = np.concatenate([loaded_list[0][0], loaded_list[2][0], loaded_list[4][0]])
        train_labels = np.concatenate([loaded_list[0][1], loaded_list[2][1], loaded_list[4][1]])
        predict_data = np.concatenate([loaded_list[1][0], loaded_list[3][0]])
        predict_labels = np.concatenate([loaded_list[1][1], loaded_list[3][1]])
        # train_data = np.concatenate([train_data, train_data1])
        # train_labels = np.concatenate([train_labels, train_labels1])
        # predict_data = np.concatenate([predict_data, predict_data1])
        # predict_labels = np.concatenate([predict_labels, predict_labels1])
        # 数据归一化
        train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
        predict_data = (predict_data - np.mean(predict_data, axis=0)) / np.std(predict_data, axis=0)
        cnn_process(train_data, train_labels, predict_data, predict_labels)

