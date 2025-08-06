# Chapter 07. AI 모델링 필수 개념 이해하기

## SECTION 01. AI란 무엇인가?

### 1. AI(Artificial Intelligence)의 정의
- 인간의 지능적 행동(이해, 추론, 판단, 학습 등)을 컴퓨터가 모방하도록 하는 기술.
- 기존의 자동화 시스템보다 고차원적인 판단과 추론을 수행함.
- **4차 산업혁명**의 핵심 기술 중 하나로 다양한 산업에서 활용됨.

### 2. AI와 머신러닝, 딥러닝의 관계

| 구분       | 설명                                         |
|------------|----------------------------------------------|
| 인공지능(AI) | 가장 상위 개념. 인간처럼 '지능적인' 작업을 수행하는 시스템. |
| 머신러닝(ML) | AI의 하위 분야. 데이터를 기반으로 스스로 학습하여 규칙을 생성. |
| 딥러닝(DL)   | 머신러닝의 하위 분야. 인공신경망 기반으로 복잡한 문제 해결. |

※ 세 개념은 포함 관계로 연결되어 있음.

---

## SECTION 02. AI 학습 방법 이해하기

AI 학습 방식은 주로 다음의 세 가지로 분류됨:
- 지도학습(Supervised Learning)
- 비지도학습(Unsupervised Learning)
- 강화학습(Reinforcement Learning)

### 1. 지도학습 이해하기
- **정답(label)** 이 있는 데이터를 사용해 학습.
- 입력과 출력 쌍이 주어진 데이터를 기반으로 모델을 학습시킴.
- 대표적인 문제 유형: **회귀(Regression)**, **분류(Classification)**
- 실제 AICE Associate도 회귀 또는 분류 문제 중 하나로 출제

예시:
- 동그라미와 세모의 이미지를 보고, 정답을 기준으로 분류하는 방식.
- 회귀: 시험 점수 예측 (연속값)
- 분류: 합격 여부 판단 (이산값)

### 2. 딥러닝 이해하기
- 인공신경망(ANN)을 활용한 머신러닝의 하위 분야.
- 인간의 뇌를 모방한 구조로, **입력층 – 은닉층 – 출력층** 구조의 레이어 사용.
- **오차역전파(Backpropagation)** 기법을 통해 학습 성능 향상.
- **레이어(layer)**를 깊게 쌓아 복잡한 문제를 해결함.
- 계산량이 많고 자원이 많이 필요하나 높은 성능을 보임.

### 3. 비지도학습 이해하기
- 정답(label)이 없는 데이터를 학습.
- **군집화(Clustering)**, 차원 축소(Dimensionality Reduction) 등에 사용.
- 대표 알고리즘: **K-means**
  - 데이터를 k개의 그룹으로 자동 분류.
  - k는 사용자가 지정해야 함.

## SECTION 03. AI 모델링 프로세스 이해하기

AI를 통한 학습은 목적과 데이터의 형태에 따라 다양한 방식으로 수행되며, 전체적인 학습 프로세스를 이해하는 것이 중요.

---

### 1. AI 모델링 프로세스

AI 학습은 일반적으로 다음과 같은 **6단계**를 반복적으로 수행.

1. 데이터 확인
2. 데이터 전처리
3. AI 모델 설정
4. 학습 데이터 분할
5. 모델 학습
6. 성능 평가

이 과정은 **성능 향상을 위한 반복 수행**을 전제로 하며, 적절한 AI 모델 탐색 및 하이퍼파라미터 조정 등도 포함

---

#### 1. 데이터 확인

- 데이터의 **구조와 속성**을 파악하는 단계입니다.
- 예시: 텍스트, 이미지, 영상 등 유형 확인 및 결측치, 이상치 등 전처리 필요 여부 파악.
- 데이터가 분석 목적에 어떤 영향을 줄 수 있을지 확인해야 합니다.

---

#### 2. 데이터 전처리

- 학습에 불필요한 데이터 제거 및 정제 과정.
- 주요 작업:
  - 중복 데이터 제거
  - 결측치 처리
  - 이상치 제거
  - 범주형 데이터 인코딩
  - 스케일링 등

---

#### 3. AI 모델 설정

- 데이터의 형태, 분석 목적에 따라 **적합한 AI 알고리즘**을 선택하는 과정.
- 단순한 모델부터 복잡한 딥러닝 모델까지 다양하며, 모델 선택은 성능과 리소스 고려가 필요.
- 하이퍼파라미터 조정 등을 통해 반복적으로 성능 개선을 시도함.

---

#### 4. 학습 데이터 분할

- 데이터를 용도에 따라 **훈련(Train)**, **검증(Validation)**, **테스트(Test)** 세 부분으로 나눔.
- 일반적인 비율: `6:2:2` 또는 `7:2:1`
- 이유:
  - 훈련 데이터: 모델 학습
  - 검증 데이터: 모델 튜닝 및 비교
  - 테스트 데이터: 최종 성능 평가
 
    

##### 데이터 분할 목적
> 전체 데이터를 한 번에 학습에 사용하면 **과적합(overfitting)** 우려 발생 → 검증용 데이터가 반드시 필요함.

---

#### 5. 모델 학습

- 입력 데이터를 기반으로 예측 결과를 출력하도록 AI 모델이 학습하는 과정.
- 다양한 알고리즘 방식에 따라 학습 형태가 달라지며, 이후의 섹션에서 구체적으로 실습 예정.

---

#### 6. 성능 평가

- 테스트 데이터를 기반으로 모델의 **최종 성능**을 평가함.
- 평가 지표는 예측하고자 하는 출력값의 형태에 따라 결정됨.
  - 분류 문제 → 정확도, 정밀도, 재현율 등
  - 회귀 문제 → MSE, RMSE 등

> 성능이 낮을 경우 다시 전처리, 모델 선택, 하이퍼파라미터 등을 조정하며 반복 수행함.

---

### 데이터셋 용도 정리 (표)

| 사용 여부             | Training | Validation | Test |
|----------------------|----------|------------|------|
| 학습 과정             | O        | O          | X    |
| 모델 가중치 설정      | O        | X          | X    |
| 모델 성능 평가        | X        | O          | O    |

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
X, y = load_iris(return_X_y=True)

# 2. train(60%) / validation(20%) / test(20%) 분할
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)  # 0.25 x 0.8 = 0.2

# 3. 모델 학습 (훈련 데이터 사용)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)  # 학습 과정: Training dataset

# 4. 검증 (Validation dataset으로 평가)
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.2f}")  # 모델 성능 평가: Validation dataset

# 5. 테스트 (Test dataset으로 최종 평가)
test_preds = model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc:.2f}")  # 모델 성능 평가: Test dataset
```



## SECTION 04. 학습 데이터의 분할 방법 이해하기

---

### 1. 학습 데이터 분할하기 (train_test_split)

- **사이킷런(sklearn)**의 `train_test_split` 함수를 사용하면 학습 데이터를 손쉽게 나눌 수 있습니다.
- 전체 데이터를 직접 분할하기보다는 해당 함수를 활용하는 것이 효율적입니다.

#### 사용 예시
```python
x_train, x_valid, y_train, y_valid = train_test_split(
    data, target, 
    test_size=None, 
    train_size=None, 
    random_state=None, 
    shuffle=True, 
    stratify=None
)
```

#### 주요 파라미터
- **test_size**: 테스트 데이터 비율 (0.0 ~ 1.0)
- **train_size**: 학습 데이터 비율 (0.0 ~ 1.0)
- **random_state**: 무작위 분할 시 시드값 고정
- **shuffle**: 데이터를 섞을지 여부 (`True` 권장)
- **stratify**: 특정 클래스(Label) 비율을 유지하고자 할 때 사용 (`Target`)

---

#### Stratify의 의미

- `stratify=None`이면 데이터 불균형이 발생할 수 있음.
- `stratify=target`을 사용하면 분할된 데이터에서도 원본의 클래스 분포가 유지됩니다.

#### 비교 예시

| Stratify=None | Stratify=Target |
|---------------|-----------------|
| 클래스 불균형 발생 가능 | 클래스 비율 유지 |

---
### 2. k-fold 교차 검증하기

- 하나의 데이터셋을 여러 번 평가하여 성능 편차를 최소화함.
- 주로 하이퍼파라미터 최적화 과정에서 사용됩니다.

### 1. k-fold 분할

- 데이터를 **k개의 fold**로 나누어 각 fold를 번갈아 가며 검증에 사용합니다.
- 모든 데이터가 학습 및 검증에 한 번씩 사용됩니다.

**k=5 예시**

| Fold | 사용 역할 |
|------|-----------|
| 1~4  | Train     |
| 5    | Validation |

---

### 2. 교차 검증 (Cross Validation)

- k개의 fold로 분할된 데이터를 교차하여 학습하고 검증함.
- 모든 데이터가 검증에 활용되므로 일반화 성능을 높일 수 있음.
- 단, 계산량이 많아 시간이 오래 걸릴 수 있음.

---

## 요약

| 분할 방식          | 설명 |
|-------------------|------|
| train_test_split  | 간단하고 빠르게 학습/검증 데이터 분할 |
| Stratified Split  | 클래스 비율을 유지하며 데이터 분할 |
| k-fold            | 모든 데이터를 학습 및 검증에 활용 |
| k-fold CV         | 교차 검증 방식으로 모델 성능 평가 |


### 3. 학습 과정을 시각화하여 과적합 확인

- 학습 과정(Epochs)에서 **정확도(Accuracy)**와 **오차(Loss)**를 추적하여 과적합 여부를 확인합니다.
- **과적합(Overfitting)**: 학습 데이터 성능은 높지만 검증 데이터 성능은 저하되는 현상.

### 과적합 현상 예시
- Train Accuracy ↑, Validation Accuracy ↓
    <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/d2b57cd9-bd00-4a2a-abbc-d8d357c9245d" />
- Train Loss ↓, Validation Loss ↑
    <img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/e9587e6c-e763-471a-af32-3dcbf21553f3" />

- **Train Loss**: Epoch이 증가할수록 손실은 지속적으로 감소합니다.
- **Validation Loss**: 초반에는 감소하지만 어느 시점 이후 오히려 증가합니다.
- **과적합 발생 시점**: Validation Loss가 증가하는 지점.

---

### 🧠 과적합이 발생하는 이유

| 구분               | 설명 |
|--------------------|------|
| Train Dataset      | 모델이 직접 보고 학습한 데이터 → 반복할수록 정확도는 높아짐 |
| Validation Dataset | 모델이 학습 중 보지 않았던 데이터 → 과하게 외운 경우 정확도 낮아짐 |
| 과적합 발생 원인    | 모델이 학습 데이터를 너무 잘 외우다 보니, 새로운 데이터에 대해 일반화하지 못함 |

---

### 과적합 방지 및 대응 방법

| 기법                | 설명 |
|---------------------|------|
| Early Stopping      | Validation Loss가 증가하면 학습 중단 |
| Regularization      | L1, L2 정규화를 통해 과도한 가중치 제어 |
| Dropout             | 일부 뉴런을 랜덤하게 제거하여 과적합 방지 |
| Data Augmentation   | 학습 데이터를 증강하여 일반화 향상 |
| 모델 단순화         | 파라미터 수를 줄여 모델 복잡도를 낮춤 |

---

