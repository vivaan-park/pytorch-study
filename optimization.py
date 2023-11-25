import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

### 모델 매개변수 최적화하기
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

### 하이퍼파라미터(Hyperparameter)
# 하이퍼파라미터는 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수
# 서로 다른 하이퍼파라미터 값은 모델 학습과 수렴률(convergence rate)에 영향을 미칠 수 있음

# 에폭(epoch) 수: 데이터셋을 반복하는 횟수
# 배치 크기(batch_size): 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플 수
# 학습률(learning rate): 각 배치/에폭에서 모델의 매개변수를 조절하는 비율
#                      값이 적을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할수 없는 동작이 발생할 수 있음
learning_rate = 1e-3
batch_size = 64
epochs = 5

### 최적화 단계(Optimization Loop)
# 하이퍼파라미터를 설정한 뒤 최적화 단계를 통해 모델을 학습하고 최적화할 수 있음
# 최적화 단계의 각 반복(iteration)을 에폭이라 부름
# 하나의 에폭은 다음 두 부분으로 구성됨
# 1. 학습 단계(train loop): 학습용 데이터셋을 반복하고 최적의 매개변수로 수렴
# 2. 검증/테스트 단계(validation/test loop): 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋 반복

# 손실 함수(loss function)
# 학습용 데이터를 제공하면, 학습되지 않은 신경망은 정답을 제공하지 않을 확률이 높음
# 손실함수는 획득한 결과와 실제 값 사이의 틀린 정도(degree of dissimilarty)를 측정하며, 학습 중에 이 값을 최소화하려고 함
# 주어진 데이터 샘플을 입력으로 계산한 예측과 정답을 비교하여 loss를 계산함
# 일반적인 손실함수에는 회귀 문제(regression task)에 사용하는 nn.MSELoss(평균 제곱 오차)
# 분류(classification)에 사용하는 nn.NLLLoss(음의 로그 우도)
# nn.LogSoftmax와 nn.NLLLoss를 합친 nnCrossEntropyLoss 등이 있음
loss_fn = nn.CrossEntropyLoss()

### 옵티마이저(Optimizer)
# 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정
# 최적화 알고리즘은 이 과정이 수행되는 방식(여기서는 확률적 경사하강법(SGD))을 정의
# 모든 최적화 절차는 optimizer 객체에 캡슐화됨
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 학습 단계에서 최적화는 세단계로 이뤄짐
# 1. optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정
#    기본적으로 변화도는 더해지기 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정
# 2. loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파함
#    PyTorch는 각 매개변수에 대한 손실의 변화도를 저장함
# 3. 변화도를 계산한 뒤 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정함
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 모델을 train 모드로 설정
    # 배치 정규화(batch normalization 및 드롭아웃(dropout) 레이어들에 중요함
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 예측과 손실 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f'loss: {loss} [{current}/{size}]')

def test_loop(dataloader, model, loss_fn):
    # 모델을 평가(eval) 모드로 설정
    # 배치 정규화(batch normalization 및 드롭아웃(dropout) 레이어들에 중요함
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # torch.no_grad()를 사용하여 테스트 시 gradient를 계산하지 않도록 함
    # 이는 requires_grad=True로 설정된 텐서들의 불필요한 변화도 연산 및 메모리 사용량 또한 줄여줌
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {100 * correct}%, Avg loss: {test_loss}')

# 손실함수와 옵티마이저를 초기화하고 train_loop오 test_loop에 전달
# 모델의 성능 향상을 알아보기 위해 자유롭게 에폭 수를 증가시킬 수 ㅣㅆ음
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

eopchs = 10
for t in range(epochs):
    print(f'Epoch {t + 1}')
    print('-----------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print('DONE!')