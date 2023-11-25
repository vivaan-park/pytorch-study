import torch
import torchvision.models as models


### 모델 가중치 저장하고 불러오기
# PyTorch 모델은 학습한 매개변수를 state_dict라고 불리는 내부 상태 사전에 저장함
# 이 상태 값들은 torch.save를 통해 저장할 수 있음
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# 모델 가중치를 불러오기 위해서는 먼저 동일한 모델의 인스턴스를 생성한 다음
# load_state_dict() 메소드를 사용하여 매개변수들을 불러옴
model = models.vgg16() # 여기서는 가중치를 지정하지 않았으므로, 학습되지 않은 모델을 생성
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

### 모델의 형태를 포함하여 저장하고 불러오기
# 모델의 가중치를 불러올 때, 신경망의 구조를 정의하기 위해 모델 클래스를 먼저 생성해야 함
# 이 클래스의 구조를 모델과 함께 저장하고 싶으면, model을 저장 함수에 전달함
torch.save(model, 'model.pth')
torch.load('model.pth')
