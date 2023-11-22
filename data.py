import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd


### 데이터셋 불러오기
# root: 학습/테스트 데이터 저장 경로
# train: 학습/테스트용 데이터셋 여부
# download=True: root에 데이터가 없을 경우 인터넷에서 다운 여부
# transform, target_transform: 특징(feature), 정답(label), 변형(transform) 지정
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

### 데이터셋 순회 및 시각화
# Dataset에 리스트처럼 접근 가능
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()

### 사용자 정의 데이터셋
# 사용자 정의 Dataset class는 반드시 아래 3개의 함수를 구현해야 함
# __init__, __len__, __getitem__

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        '''
        Dataset 객체가 생성될 때 한 번만 실행
        여기서는 이미지 주석파일이 포함된 디렉토리, 두가지 변형(transform, target_transform)을 초기화

        labels.csv는 다음과 같음
        tshirt1.jpg, 0
        tshirt2.jpg, 0
        ...
        ankleboot999, 9
        '''
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        데이터셋의 샘플 수 반환
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        주어진 idx에 해당하는 샘플을 데이터셋에서 불러오고 반환
        인덱스를 기반으로 이미지의 위치를 식별하고, read_image를 통해 이미지를 텐서로 변환하고,
        self.img_labels로부터 해당하는 정답(label)을 가져오고,
        (해당하는 경우) transform 함수를 호출한 뒤, 텐서 이미지와 label을 dict형으로 반환
        '''
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform(image):
            label = self.target_transform(label)
        return image, label

### DataLoader로 학습용 데이터 준비
# Dataset은 데이터셋의 feature를 가져오고 하나의 샘플에 label을 지정하는 일을 한 번에 함
# 일반적으로 모델 학습 시 샘플들을 미니배치(minibatch)로 전달하고, 매 에폭(epoch)마다 데이터를 다시 섞어서
# 과적합(overfit)을 막고, 파이썬의 multiprocessing을 사용하여 데이터 검색 속도를 높이고자 함
# DataLoader는 간단한 API로 위 복잡한 과정들을 추상화한 순회 가능한 iterable 객체
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

### DataLoader를 통해 순회하기(iterate)
# DataLoader에 데이터셋을 불러온 뒤 필요에 따라 순회(iterate) 가능
# 아래의 각 iteration은 각각 batch_size=64의 feature와 label을 포함하는
# train_features와 train_labels의 묶음(batch)을 반환
# shuffle=True로 지정하면 모든 배치를 순회한 뒤 데이터가 섞임
# (순서를 보다 세밀하게(fine-grained) 제어하려면 sampler 사용)
train_features, train_labels = next(iter(train_dataloader))
print(f'Featrues batch shape: {train_features.size()}')
print(f'Labels batch shape: {train_labels.size()}')
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap='gray')
plt.show()
print(f'Label: {label}')
