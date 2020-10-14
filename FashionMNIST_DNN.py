import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 30
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.FashionMNIST(
    root = "./data/",
    train = True,
    download = True,
    transform = transform
)

testset = datasets.FashionMNIST(
    root = "./data/",
    train = False,
    download = True,
    transform = transform
)

train_loader = data.DataLoader(  # 에폭마다 네트워크에 데이터를 주입해줌
    dataset = trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = data.DataLoader(
    dataset = testset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
"""
dataiter = iter(train_loader)
images, labels = next(dataiter)

img = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))
plt.show()

CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

for label in labels:
    index = label.item()
    print(CLASSES[index])

idx = 1
item_img = images[idx]
item_npimg = item_img.squeeze().numpy()
plt.title(CLASSES[labels[idx].item()])
plt.imshow(item_npimg, cmap="gray")
plt.show()
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net().to(DEVICE)  # cpu 혹은 cuda로 네트워크를 보냄
optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(model, train_loader, optimizer):
    model.train()  # 모델을 학습 모드로 전환, 드롭아웃 등의 레이어에서 동작이 다름

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)  # 데이터도 처리할 디바이스로 보내야 함
        optimizer.zero_grad()  # 매 반복마다 기울기를 새로 계산해야 함
        output = model(data)
        loss = F.cross_entropy(output, target)  # 결과물과 정답 사이 크로스엔트로피 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 수정


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