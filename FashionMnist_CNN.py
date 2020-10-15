import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 40
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("./data",
                          train=True,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST("./data",
                          train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(model, train_loader, optimizer):
    model.train()  # 모델을 학습 모드로 전환, 드롭아웃 등의 레이어에서 동작이 다름

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)  # 데이터도 처리할 디바이스로 보내야 함
        optimizer.zero_grad()  # 매 반복마다 기울기를 새로 계산해야 함
        output = model(data)
        loss = F.cross_entropy(output, target)  # 결과물과 정답 사이 크로스엔트로피 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 수정

        if batch_idx % 200 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))


def evaluate(model, test_loader):
    model.eval()  # 모델을 평가 모드로 전환
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 평가시엔 기울기 계산 필요없음
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            pred = output.max(1, keepdim=True)[1]  # 모델의 예측 클래스
            correct += pred.eq(target.view_as(pred)).sum().item()  # 정답이면 +1

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy


for epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))

