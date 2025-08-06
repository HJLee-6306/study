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

## 과적합 방지 및 대응 기법 통합 정리

| 기법                       | 설명                                                            | 작동 원리 / 목적                                           | 구현 예시                                                             |
|----------------------------|-----------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------|
| **Early Stopping**         | 검증 손실(`val_loss`)이 더 이상 줄어들지 않고 증가하는 시점에서 훈련을 자동으로 중단 | 모델이 검증 데이터에 최적화될 수 있도록 조기에 학습 종료             | `EarlyStopping(monitor='val_loss', patience=5)`                       |
| **Regularization (정규화)** | 손실 함수에 가중치 패널티 항목을 추가하여 복잡한 모델을 억제                            | 너무 큰 가중치가 형성되는 것을 방지하여 모델 복잡도 제어               | `regularizers.l1(0.01)` 또는 `regularizers.l2(0.01)`                  |
| **Dropout**                | 학습 중 일부 뉴런을 랜덤하게 비활성화하여 과적합 방지                                 | 특정 뉴런이나 경로에 의존하지 않게 하여 다양한 경로로 학습 유도          | `Dropout(0.5)` → 학습 시 50% 뉴런 제거                                |
| **Data Augmentation**      | 이미지나 텍스트 등의 데이터를 회전·뒤집기 등으로 변형하여 데이터셋 확장                | 다양한 변형된 데이터를 학습시켜 모델의 일반화 성능 향상                | `ImageDataGenerator(rotation_range=20, horizontal_flip=True)`        |
| **모델 단순화**            | 층 수나 노드 수를 줄여 모델 구조를 간결하게 구성                                   | 복잡한 모델이 훈련 데이터를 외우는 현상 방지                           | CNN에서 Conv 층 5개 → 2~3개로 축소                                   |
| **Batch Normalization**    | 각 층의 출력값을 정규화하여 학습을 안정화                                          | 내부 공변량 변화(Internal Covariate Shift) 억제 → 빠르고 안정된 학습 유도 | `BatchNormalization()` 레이어 사용                                    |
| **Cross-validation**       | 데이터를 여러 조각으로 나누어 검증을 반복 수행                                   | 데이터 분할에 따른 편향을 줄이고 과적합 가능성을 사전에 파악            | `StratifiedKFold`, `cross_val_score()` 등 사용                        |
| **Ensemble Learning**      | 여러 개의 모델을 조합하여 예측 결과를 종합                                       | 단일 모델보다 예측의 안정성과 일반화 성능 향상                          | Random Forest, Voting Classifier, Stacking 등                         |
  |    |

---
## SECTION 05. AI 모델 평가 이해하기

# 오차행렬 (Confusion Matrix) 이해하기

오차행렬은 분류 모델의 예측 결과를 실제 정답과 비교하여 정리한 표이며, 모델이 어떤 종류의 오류를 범했는지를 한눈에 파악할 수 있음

---

## 1. 기본 구조

| 예측값 ↓ / 실제값 → | Positive (참) | Negative (거짓) |
|---------------------|---------------|-----------------|
| Positive (양성) 예측 | True Positive (TP) | False Positive (FP) |
| Negative (음성) 예측 | False Negative (FN) | True Negative (TN) |

- **True Positive (TP)**: 실제로 Positive인 것을 Positive라고 예측함  
- **False Positive (FP)**: 실제로는 Negative인데 Positive로 잘못 예측함 (Type I Error)  
- **False Negative (FN)**: 실제로는 Positive인데 Negative로 잘못 예측함 (Type II Error)  
- **True Negative (TN)**: 실제로 Negative인 것을 Negative라고 정확히 예측함  

---

## 2. 주요 평가 지표

| 지표 | 의미 | 공식 |
|------|------|------|
| **Accuracy (정확도)** | 전체 중 맞춘 비율 | (TP + TN) / (TP + FP + FN + TN) |
| **Precision (정밀도)** | Positive로 예측한 것 중 실제로 Positive인 비율 | TP / (TP + FP) |
| **Recall (재현율)** | 실제 Positive 중에 올바르게 예측한 비율 | TP / (TP + FN) |
| **F1 Score** | Precision과 Recall의 조화 평균 | 2 * (Precision * Recall) / (Precision + Recall) |

---

## 3. 예시로 이해하기

- 전체 데이터: 100건 중 실제 Positive 70건, Negative 30건  
- 모델이 모든 데이터를 **Positive**로 예측했다고 가정하면:

| 예측값 ↓ / 실제값 → | Positive | Negative |
|---------------------|----------|----------|
| Positive            | 70       | 30       |
| Negative            | 0        | 0        |

- **Precision** = 70 / (70 + 30) = **0.70**  
- **Recall** = 70 / (70 + 0) = **1.00**  
- **Accuracy** = (70 + 0) / 100 = **0.70**  
- **F1 Score** = 2 * (0.70 * 1.00) / (0.70 + 1.00) = **0.82**

> 정확도만 보면 괜찮아 보이지만, Negative를 아예 예측하지 못한 모델이므로 한쪽에만 편향된 성능을 보임

---

## 4. 요약

- 오차행렬은 모델의 정답과 오류를 모두 파악할 수 있게 도와주는 기본 도구입니다.  
- **Accuracy**만 볼 경우, **데이터가 불균형할 때** 성능을 과대평가할 수 있으므로,  
  **Precision, Recall, F1 Score**를 함께 고려해야 함


## 빅테크 기업의 실무 평가 지표 활용 방식

빅테크 기업들(Google, Amazon, Meta, Microsoft, Apple 등)은 **모델의 목적, 제품 특성, 데이터 불균형 여부**에 따라 평가 지표를 전략적으로 선택합니다. 다음은 실무에서의 실제 활용 방식입니다.

---

### 1. 정확도 (Accuracy)

- **사용 사례**: 클래스 불균형이 심하지 않은 일반 분류 문제  
- **한계**: 불균형 데이터에서는 쓸모 없음 (예: 사기 탐지 등에서 부적절)  
- **예시**:
  - Google의 **검색 결과 순위 분류**
    - 90% 이상의 정확도를 요구하는 상황에서 사용

---

### 2. 정밀도 (Precision)

- **사용 사례**: False Positive가 **위험한 경우**
- **예시**:
  - Gmail의 **스팸 필터**
    - 정상 메일을 스팸으로 분류하면 안 되므로 Precision 우선
  - **의료 진단 모델**
    - "암이 있다"는 잘못된 진단은 환자에게 큰 부담 → 고정밀도 필요

---

### 3. 재현율 (Recall)

- **사용 사례**: 놓치면 안 되는 **중요 케이스**
- **예시**:
  - Facebook의 **불법 콘텐츠 탐지 시스템**
    - 유해 콘텐츠는 반드시 검출되어야 함
  - Amazon의 **리뷰 조작 탐지**
    - 조작된 리뷰를 최대한 많이 잡기 위해 Recall 우선

---

### 4. F1 Score

- **사용 사례**: 정밀도와 재현율이 **모두 중요한 경우** (특히 **불균형 데이터**)
- **예시**:
  - Google의 **뉴스 분류 시스템**
    - 잘못된 뉴스 분류도 안 되고, 중요한 뉴스 놓쳐도 안 됨
  - Meta의 **가짜 뉴스 탐지 모델**

---

### 5. AUC-ROC

- **사용 사례**: 다양한 threshold에서 모델의 전체적인 분류 성능 확인
- **예시**:
  - YouTube의 **추천 시스템**
    - 다양한 사용자 행동 예측에서 threshold 조절하여 평가
  - 광고 클릭 예측 모델 (**CTR prediction**)

---

### 요약

| 지표        | 사용 상황                                     | 대표 사례                      |
|-------------|-----------------------------------------------|-------------------------------|
| Accuracy    | 클래스 균형이 적절한 상황에서 전체 성능 측정     | 검색 순위 모델                 |
| Precision   | False Positive가 큰 문제일 때                  | 스팸 필터, 암 진단              |
| Recall      | 놓치면 안 되는 항목이 있을 때                   | 불법 콘텐츠 탐지, 리뷰 조작 탐지 |
| F1 Score    | Precision과 Recall 모두 중요, 불균형 상황에서 사용 | 뉴스 분류, 가짜 뉴스 탐지       |
| AUC-ROC     | 다양한 threshold 기준 성능 평가                 | 추천 시스템, 광고 클릭 예측     |

---

## 회귀 모델 평가 지표 정리

회귀(Regression) 문제에서 모델 성능을 평가할 때 자주 사용되는 주요 지표는 다음과 같습니다.

---

### 1. MAE (Mean Absolute Error, 평균 절대 오차)

- **정의**: 실제 값과 예측 값의 절대값 차이의 평균
- **공식**:  
  MAE = (1 / n) × Σ | yᵢ - ŷᵢ |
- **특징**:
  - 오차의 절대값만 반영 → **직관적**
  - 이상치(Outlier)에 **덜 민감**
- **예시 상황**: 사람이 해석하기 쉬운 지표가 필요할 때

---

### 2. MSE (Mean Squared Error, 평균 제곱 오차)

- **정의**: 실제 값과 예측 값의 차이를 제곱한 값의 평균
- **공식**:  
  MSE = (1 / n) × Σ (yᵢ - ŷᵢ)²
- **특징**:
  - 큰 오차에 **패널티를 크게 부여**
  - 이상치에 **민감**
- **예시 상황**: 오차가 클수록 큰 불이익이 따르는 문제

---

### 3. RMSE (Root Mean Squared Error, 평균 제곱근 오차)

- **정의**: MSE의 제곱근을 취한 값
- **공식**:  
  RMSE = √[(1 / n) × Σ (yᵢ - ŷᵢ)²]
- **특징**:
  - **MSE의 단위 문제를 보완**
  - 오차의 단위를 실제 값과 **같게** 함
- **예시 상황**: 실제 단위와 같은 스케일의 오차를 보고 싶을 때

---

### 4. R² Score (결정계수)

- **정의**: 예측값이 실제값을 얼마나 잘 설명하는지를 나타내는 지표 (0~1 사이)
- **공식**:  
  R^2 = 1 - [ Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)² ]
- **특징**:
  - 1에 가까울수록 좋은 모델
  - 0이면 **무작위 평균 예측 수준**
  - 음수일 수도 있음 → **예측 성능이 평균보다 못함**
- **예시 상황**: 전체 설명력을 파악하고 싶을 때

---

### 요약 비교

| 지표     | 오차의 민감도 | 해석 용이성 | 단위 유지 | 이상치 민감도 |
|----------|----------------|--------------|------------|----------------|
| MAE      | 보통           | 높음         | O          | 낮음           |
| MSE      | 큼             | 낮음         | X          | 높음           |
| RMSE     | 큼             | 보통         | O          | 높음           |
| R² Score | 상대 지표      | 보통         | 비율(%)    | 상황에 따라 다름 |

---

### 참고

Learning Curve 기반 학습 전략 최적화 정리
1. Learning Curve란?
정의: 학습이 진행됨에 따라 모델의 성능(보통 오차 또는 정확도 등)이 훈련 데이터와 검증 데이터에서 어떻게 변화하는지를 시각화한 그래프

축의 구성:

X축: 학습 데이터 수(또는 에포크 수)

Y축: 성능 지표 (오차, 정확도, loss 등)

2. Learning Curve를 통해 확인할 수 있는 것
구분	훈련 오류 (Training Loss)	검증 오류 (Validation Loss)	의미	대처 전략
과소적합 (Underfitting)	높음	높음	모델이 너무 단순	- 더 복잡한 모델 사용
- 더 오래 학습
- 더 많은 특성 사용
과적합 (Overfitting)	낮음	높음	훈련 데이터에 과도하게 적합	- 정규화(L1/L2)
- Dropout
- Early stopping
- 데이터 증강
적절한 학습	낮음	낮음	학습과 일반화가 잘 이루어짐	- 현재 상태 유지 또는 성능 향상을 위한 소폭 튜닝

3. Typical Learning Curve 패턴
과소적합

훈련/검증 에러 모두 높고, 에러가 수렴하지 않음

모델 용량이 부족하거나 충분히 학습하지 못함

과적합

훈련 에러는 낮지만, 검증 에러가 높고 벌어지는 경향

일반화 성능이 낮음

적절한 학습

두 에러가 모두 낮고 일정 수준에서 수렴

4. 학습 전략 최적화 시 고려 사항
데이터 수가 적다면? → 데이터 증강 / 수집

훈련 시간이 짧다면? → 에포크 수 증가, 학습률 조절

학습 곡선이 진동/불안정하다면? → Learning Rate 감소, 배치 정규화

성능이 plateau라면? → 새로운 특성 도입, 모델 구조 변경

5. 관련 평가 지표
학습 곡선과 함께 보는 주요 지표:

Train Loss / Val Loss

Accuracy

Early Stopping 기준 (e.g., patience)

Validation Curve: 하이퍼파라미터별 검증 성능 변화

6. 예시 문제 풀이 전략
질문이 “학습 곡선 상에서 어떤 전략이 적절한가?”라면:

곡선 간 간격이 크면 과적합 → 일반화 기법 필요

둘 다 높으면 과소적합 → 모델 복잡도 증가

둘 다 낮고 안정 → 학습 완료, 추가 조치 불필요

---

### 챕터 7 연관 파일럿 문제
<img width="807" height="733" alt="image" src="https://github.com/user-attachments/assets/7c05d11f-469c-4c6c-b240-579eb26cd5ec" />

<img width="772" height="366" alt="image" src="https://github.com/user-attachments/assets/47a44cbf-4786-471c-b194-a9bfc54f29ff" />

<img width="808" height="656" alt="image" src="https://github.com/user-attachments/assets/2c4694dd-f64b-4678-b1fe-fac8797e91a4" />

<img width="792" height="776" alt="image" src="https://github.com/user-attachments/assets/0727ffcc-3767-42ef-bf47-fb736e38b815" />



