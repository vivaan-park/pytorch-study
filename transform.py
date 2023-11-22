import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


### 변형(Transform)
# 데이터가 항상 머신러닝 알고리즘 학습에 필요한 최종 처리가 된 형태로 제공되지 않음
# 따라서 변형(Transform)을 통해 데이터를 조작하고 학습에 적합하게 만들어야함
# 모든 TorchVision 데이터셋들은 변형 로직을 갖는, 호출 가능한 객체(callable)를 받는
# 매개변수 두 개(feature)를 변경하기 위한 transform과
# label을 변경하기 위한 target_transform을 갖음
# torchvision.transforms 모듈은 주로 사용하는 몇가지 transform을 제공

### ToTensor()
# ToTensro는 PIL image나 nump ndarray를 FloatTensor로 변환하고,
# 이미지의 픽셀 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)함

### Lambda 변형
# Lambda Transform은 사용자 정의 lambda 함수를 적용

# 여기서는 정수를 one-hot으로 부호화된 텐서로 바꾸는 함수를 정의
# 크기 10짜리 zero 텐서를 만들고 scatter_를 호출하여 정답 y에 해당하는 인덱스에 value=1을 할당
ds = datasets.FashionMNIST(
    root='data',
    train='True',
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

