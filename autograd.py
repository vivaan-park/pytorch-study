import torch


### torch.autograd를 사용한 자동 미분
# 신경망을 학습할 때 자주 사용되는 알고리즘은 역전파로
# 역전파에서 매개변수(모델 가중치)는 주어진 매개변수에 대한 손실 함수의 변화도(gradient)에 따라 조정됨
# 이러한 gradient를 계산하기 위해 torch.autograd라는 자동 미분 엔진을 사용
# 모든 계산 그래프에 대한 gradient의 자동 계산을 지원함

# 여기서 w, b는 최적화를 해야하는 매개변수임
# 이러한 변수들에 대한 손실 함수의 gradient를 계산할 수 있어야함
x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 연산 그래프를 구성하기 위해 텐서에 적용하는 함수는 Function 클래스의 객체임
# 이 객체는 순전파 방향으로 함수를 계산하는 방법과, 역전파 단계에서 도함수(derivative)를 계산하는 방법을 알고있음
# 역전파 함수에 대한 참조(reference)는 텐서의 grad_fn 속성에 저장됨
print(f'Gradient function for z = {z.grad_fn}')
print(f'Gradient function for loss = {loss.grad_fn}')

### Gradient 계산
# 신경망에서 매개변수의 가중치를 최적화하려면 매개변수에 대한 손실함수의 도함수(derivative)를 계산해야 함
# 연산 그래프의 leaf 노드들 중 requires_grad 속성이 True로 설정된 노드들의 grad 속성만 구할 수 있음
# 그래프의 다른 모든 노드에서는 변화도가 유효하지 않음
# 성능 상의 이유로, 주어진 그래프에서의 backward를 사용한 변화도 계산은 한 번만 수행할 수 있음
# 만약 동일한 그래프에서 여러번의 backward 호출이 필요하면, backward 호출 시에 retrain_graph=True를 전달해야함
loss.backward()
print(w.grad)
print(b.grad)

### 변화도 추적 멈추기
# 기본적으로 requires_grad=True인 모든 텐서들은 연산 기록을 추적하고 변화도 계산을 지원함
# but 모델을 학습한 뒤 입력 데이터를 단순히 적용하기만 하는 경우와 같이 순전파 연산만 필요한 경우에는
# 이러한 추적이나 지원이 필요 없을 수 있음
# 연산코드를 torch.no_grad() 블록으로 둘러싸서 추적을 멈출 수 있음
# 변화도 추적을 멈춰야 하는 이유
# 1. 신경망의 일부 매개변수를 고정된 매개변수(frozen parameter)로 표시
# 2. 변화도를 추적하지 않는 텐서의 연산이 더 효율적이기 때문에, 순전파 단계만 수행할 때 연산 속도 향상됨
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# 이와 동일한 결과를 얻는 다른 방법은 detach() 메소드를 사용하면 됨
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

### 연산 그래프에 대한 추가 정보
# 개념적으로, autograd는 텐서의, 실행된 모든 연산들의 기록을 Function 객체로 구성된
# 방향성 비순환 그래프(DAG: Directed Acyclic Graph)에 저장함
# 이 방향성 비순환 그래프의 leaf는 입력 텐서이고, root는 결과 텐서
# 이 그래프를 root에서부터 leaf까지 추적하면 연쇄 법칙(chain rule)에 따라 변화도를 자동으로 계산 가능

# 순전파 단계에서, autograd는 다음 두 가지 작업을 동시에 수행
# 1. 요청된 연산을 수행하여 결과 텐서를 계산
# 2. DAG에 연산의 변화도 기능(gradient function)을 유지

# 역전파 단계에서는 DAG root에서 .backward()가 호출될 때 시작
# autograd는 이때
# 1. 각 .grad_fn으로부터 변화도를 계산
# 2. 각 텐서의 .grad 속성에 계산 결과를 쌓고(accumulate)
# 3. 연쇄 법칙을 사용하여, 모든 leaf 텐서들까지 전파(propagate)함

# 선택적으로 읽기(Optional Reading): 텐서 변화도와 야코비안 곱(Jacobian Product)
# 대부분의 경우, 스칼라 손실 함수를 가지고 일부 매개변수와 관련한 변화도를 계산함
# 그러나 출력함수가 임의의 텐서인 경우 실제 변화도가 아닌 야코비안 곱을 계산

inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f'First call \n{inp.grad}')
out.backward(torch.ones_like(out), retain_graph=True)
print(f'Second call \n{inp.grad}')
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f'Call after zeroing gradients \n{inp.grad}')

# 동일한 인자로 backward를 두차례 호출하면 변화도 값이 달라짐
# 역전파를 수행할 때 PyTorch가 변화도를 누적해주기 때문
# 즉, 계산된 변화도의 값이 연산 그래프의 모든 leaf 노드의 grad 속성에 추가됨
# 따라서 제대로 된 변화도를 계산하기 위해서는 grad 속성을 먼저 0으로 만들어야 함
# 실제 학습 과정에서는 optimizer가 이 과정을 도와줌