import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


train_dataset = torchvision.datasets.MNIST(root='dataset/', train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='dataset/', train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 修改卷积层的kernel_size为5
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 400)
        self.fc3 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练函数
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in tqdm(dataloader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

    loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return loss, accuracy

# 测试函数
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy





def pretrain_teacher_model():
    teacher_model = SimpleCNN().to(device)
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    total_epochs = 200  # 训练200轮

    for epoch in range(total_epochs):
        model_loss, model_acc = train_model(teacher_model, train_dataloader, optimizer, criterion)

        if model_acc > best_acc:
            best_acc = model_acc
            best_model = teacher_model.state_dict()  # 保存最优模型参数

    # 保存最优教师模型
    torch.save(best_model, 'best_teacher_model_5x5.pth')
    return teacher_model

# 运行实验
teacher_model = pretrain_teacher_model()
teacher_model.load_state_dict(torch.load('best_teacher_model.pth'))
