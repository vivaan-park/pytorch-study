import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### 신경망 모델 구성
# 신경망은 데이터에 대한 연산을 수행하는 계층(layer), 모듈(module)로 구성됨
# torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소를 제공
# PyTorch의 모든 모듈은 nn.Module의 하위 클래스
# 신경망은 다른 모듈(layer)로 구성된 모듈

# 학습 장치 선택
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Using {device} device')

### 클래스 정의하기
# 신경망 모델을 nn.Module의 하위클래스로 정의하고 __init__에서 신경망 계층을 초기화함
# nn.Module을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산을 구현함
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

# NeuralNetwork의 인스턴스(instance)를 생성하고 이를 device로 이동한 뒤 구조(structure)를 출력
model = NeuralNetwork().to(device)
print(model)

# 모델을 사용하기 위해 입력 데이터 전달
# 이는 일부 백그라운드 연산들과 함께 forward를 실행함
# !!!! model.forward()를 직접 호출하면 안됨 !!!!
# 모델에 입력을 전달하여 호출하면 2차원 텐서를 반환함
# 2차원 텐서의 dim=0은 각 분류(class)에 대한 원시(raw) 예측값 10개가, dim=1에는 각 출력의 개별 값들이 해당
# raw 예측값을 nn.Softmax 모듈의 인스턴스에 통과시켜 예측 확률을 얻음
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f'Predicted class: {y_pred}')

### 모델 계층(Layer)
# 28x28 사이즈의 이미지 3개로 구성된 미니배치 생성
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten 계층을 초기화하여 28x28의 2D 이미지를 784개의 픽셀 값을 갖는 연속된 배열로 변환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear 계층은 저장된 가중치(weight)와 편향(bias)을 사용하여 입력에 선형 변환(linear transformation)을 적용함
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU는 비선형 활성화(activation) 함수로 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 만듬
# 비선형 활성화는 선형 변환 후에 적용되어 비선형성(nonlinearity)를 도입하고,
# 신경망이 다양한 현상을 학습할 수 있도록 함
print(f'Before ReLU: {hidden1}')
hidden1 = nn.ReLU()(hidden1)
print(f'After ReLU: {hidden1}')

# nn.Sequential은 순서를 갖는 모듈의 컨테이너
# 데이터는 정의된 것과 같은 순서로 모든 모듈을 통해 전달됨
# 순차 컨테이너(sequential container)를 사용하여 seq_modules와 같은 신경망을 빠르게 만들 수 있음
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# 신경망의 마지막 선형 계층은 nn.Softmax 모듈에 전달될 ([-infty, infty] 범위의 원시 값(raw value)인) logits를 반환
# logits는 모델의 각 class에 대한 예측 확률을 나타내도록 [0, 1] 범위로 비례하여 scale됨
# dim 매개변수는 값의 합이 1이 되는 차원을 나타냄
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

### 모델 매개변수
# 신경망 내부의 많은 계층들은 매개변수화(parameterize) 됨
# 즉, 학습 중에 최적화되는 weight, bias와 연관지어짐
# nn.Module을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 추적(track)되며,
# 모델의 parameters() 및 named_parameters() 메소드로 모든 매개변수에 접근 가능하게 됨
print(f'Model structure: {model}')

for name, param in model.named_parameters():
    print(f'Layer: {name} | Size: {param.size()} | Values: {param[:2]}')
