import torch
import numpy as np


### 텐서 초기화
# 데이터로부터 직접 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# numpy 배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# 다른 텐서로부터 생성
# 명시적으로 override하지 않으면 인자로 주어진 텐서의 속성(shape, datatype)을 유지
x_ones = torch.ones_like(x_data)
print(f'Ones Tensor: {x_ones}')
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씀
print(f'Random Tensor: {x_rand}')

# random 또는 constant 값 사용
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f'Random Tensor: {rand_tensor}')
print(f'Ones Tensor: {ones_tensor}')
print(f'Zeors Tensor: {zeros_tensor}')

### 텐서 속성
# 속성은 shape, datatype, 어느 장치(cpu/gpu)에 저장되는지를 나타냄
tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')

### 텐서 연산
# transposing, indexing, slicing, 수학 계산, 선형 대수, random sampling 등
# 100가지 이상의 텐서 연산들이 있음(https://pytorch.org/docs/stable/torch.html)
# 기본적으로 텐서는 CPU에 생성됨
# .to 메소드를 사용하면 (GPU 가용성 확인 후) GPU로 텐서를 명시적으로 이동 가능
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# numpy식의 표준 indexing, slicing
tensor = torch.ones(4, 4)
print(f'First row: {tensor[0]}')
print(f'First column: {tensor[:, 0]}')
print(f'Last column: {tensor[..., -1]}')
tensor[:, 1] = 0
print(tensor)

# torch.cat 주어진 차원에 따라 일련의 텐서를 연결할 수 있음(차원 수 동일)
# torch.stack 주어진 차원에 따라 일련의 텐서를 쌓음(차원 수 +1)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
t2 = torch.stack([tensor, tensor, tensor], dim=1)
print(t2)

# 행렬 곱 계산
# tensor.T는 텐서의 전치(transpose) 반환
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱 계산
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 경우
# item()을 사용하여 숫자 값으로 변환 가능
agg = tensor.sum()
print(agg)
agg_item = agg.item()
print(agg_item, type(agg_item))

# 연산 결과를 피연산자에 저장(바꿔치기 연산)
# history가 삭제(원본 텐서를 직접 수정함)되어 도함수(derivative) 계산에 문제가 발생할 수 있음
# 사용을 권장하지 않음
print(tensor)
tensor.add_(5)
print(tensor)

### numpy 변환
# CPU 상의 텐서와 numpy 배열은 메모리 공간을 공유하기 때문에
# 하나를 변경하면 다른 하나도 변경됨
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')
t.add_(1)
print(f't: {t}')
print(f'n: {n}')

# numpy 배열의 변경사항이 텐서에 반영됨
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')