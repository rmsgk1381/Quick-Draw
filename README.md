# Quick-Draw

PyTorch와 CNN을 활용한 Quick Draw 낙서 분류 프로젝트입니다. 사람이 그린 낙서를 인식하여 사과, 바나나, 포도 중 하나로 분류하는 딥러닝 모델을 구현했습니다.

## 프로젝트 개요

이 프로젝트는 Google Quick Draw 데이터셋을 사용하여 흑백 낙서 이미지를 3가지 클래스로 분류하는 CNN 모델을 학습합니다. 모델은 2개의 Convolutional Layer와 Fully Connected Layer로 구성되어 있으며, 테스트 세트에서 약 94%의 정확도를 달성했습니다.

## 기술 스택

- **언어**: Python
- **딥러닝 프레임워크**: PyTorch
- **데이터 처리**: NumPy
- **시각화**: Matplotlib
- **개발 환경**: Google Colab

## 주요 기능

### 데이터 전처리

- 픽셀 값을 0~255 범위에서 0.0~1.0으로 정규화
- 1차원 배열(784)을 28x28 이미지로 재구성
- 커스텀 DataLoader 구현

### 모델 구조

CNN 모델은 다음과 같은 구조로 구성되어 있습니다:

- 첫 번째 레이어: Conv2d(1채널 → 16채널) → ReLU → MaxPool2d
- 두 번째 레이어: Conv2d(16채널 → 32채널) → ReLU → MaxPool2d
- 분류 레이어: Fully Connected Layer (32 * 7 * 7 → 3 classes)
- 손실 함수: CrossEntropyLoss (Softmax 포함)

### 오버피팅 방지

- train_test_split을 통한 학습/검증 데이터 분리 (8:2 비율)
- 학습 데이터와 테스트 데이터를 분리하여 모델의 일반화 성능 평가

## 데이터셋

Google Quick Draw 데이터셋에서 다음 3가지 클래스를 사용합니다:

- 사과 (Apple)
- 바나나 (Banana)
- 포도 (Grapes)

각 클래스당 3,000개의 샘플을 사용하여 총 9,000개의 이미지로 학습을 진행했습니다.

## 사용 방법

### 환경 설정

필요한 라이브러리를 설치합니다:

```bash
pip install torch torchvision numpy matplotlib
```

### 데이터 준비

Google Quick Draw 데이터셋에서 다음 파일들을 다운로드하여 프로젝트 디렉토리에 배치합니다:

- `full_numpy_bitmap_apple.npy`
- `full_numpy_bitmap_banana.npy`
- `full_numpy_bitmap_grapes.npy`

### 모델 학습

`QuickDraw_Model.ipynb` 노트북을 실행하여 모델을 학습합니다. 노트북은 다음 단계를 포함합니다:

1. 데이터 로드 및 전처리
2. 모델 정의
3. 학습 및 평가
4. 결과 시각화
5. 모델 저장

학습이 완료되면 `quickdraw_cnn.pth` 파일로 모델 가중치가 저장됩니다.

### 모델 사용

학습된 모델을 로드하여 예측을 수행할 수 있습니다:

```python
import torch
from QuickDraw_Model import QuickDrawCNN

model = QuickDrawCNN()
model.load_state_dict(torch.load('quickdraw_cnn.pth'))
model.eval()

# 예측 코드 작성
```

## 성능

- 테스트 세트 정확도: 약 94.89%
- 학습 에포크: 5
- 배치 크기: 64
- 학습률: 0.001

## 프로젝트 구조

```
Quick-Draw/
├── QuickDraw_Model.ipynb    # 모델 학습 노트북
├── quickdraw_cnn.pth         # 학습된 모델 가중치
├── README.md                 # 프로젝트 문서
├── LICENSE                   # MIT 라이선스
└── .gitignore                # Git 제외 파일 목록
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 참고 자료

- [Google Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
