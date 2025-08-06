# Chapter 06. 데이터 전처리하기

## Section 01. 수치형 데이터 정제하기

### 1. 결측치 파악하기

#### 1) 결측치 존재 여부 확인하기

```python
import pandas as pd
import numpy as np

# 예시 데이터프레임 생성
df = pd.DataFrame({
    'A': [1, 2, np.nan],
    'B': [4, np.nan, 6]
})

# 결측치 존재 여부
print(df.isnull().any())
```

#### 2) 결측치 수 확인하기

```python
# 결측치 개수 확인
print(df.isnull().sum())
```

---

### 2. 결측치 처리하기

#### 1) 결측치 삭제하기

```python
# 결측치 포함된 행 삭제
df.dropna()
```

#### 2) 칼럼 제거하기

```python
# 특정 컬럼 제거
df.drop(columns=['A'])
```

#### 3) 결측치 대체하기

| 구분                                       | 설명                                    | 예시 코드                                                                                            |
| ---------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **1. 평균(Mean) 대체**                       | 수치형 데이터에서 흔히 사용되며, 분포가 정규분포에 가까울 때 적절 | `df['A'].fillna(df['A'].mean(), inplace=True)`                                                   |
| **2. 중앙값(Median) 대체**                    | 이상치가 많거나 비대칭 분포일 때 적절                 | `df['A'].fillna(df['A'].median(), inplace=True)`                                                 |
| **3. 최빈값(Mode) 대체**                      | 범주형 변수 또는 discrete한 수치형 변수에 적합        | `df['B'].fillna(df['B'].mode()[0], inplace=True)`                                                |
| **4. 고정값(Constant) 대체**                  | 결측값을 특정한 값으로 일괄 처리할 때 사용              | `df['C'].fillna(0, inplace=True)` 또는 `"Unknown"` 등                                               |
| **5. 선형 보간법(Linear Interpolation)**      | 시간 순서가 있는 시계열 데이터에 적합                 | `df['A'].interpolate(method='linear', inplace=True)`                                             |
| **6. 다항 보간법(Polynomial Interpolation)**  | 비선형 시계열에 적합, `order`에 차수 설정 필요        | `df['A'].interpolate(method='polynomial', order=2)`                                              |
| **7. 전/후 값으로 대체(Forward/Backward Fill)** | 직전/직후의 값을 복사하여 채움. 시계열 데이터에 적합        | `df['A'].fillna(method='ffill', inplace=True)`<br>`df['A'].fillna(method='bfill', inplace=True)` |
| **8. 그룹별 평균/중앙값 대체**                     | 범주형 그룹 단위로 평균/중앙값을 계산하여 결측값 채움        | `df['A'] = df.groupby('Group')['A'].transform(lambda x: x.fillna(x.mean()))`                     |
| **9. 머신러닝 기반 예측 대체**                     | 다른 feature를 이용하여 회귀/분류 모델로 결측값 예측     | `KNNImputer`, `IterativeImputer` 등 사용                                                            |
| **10. 다중 대체(Multiple Imputation)**       | 통계적으로 더 정교한 방법. 여러 번 대체 후 평균 등 계산     | `from fancyimpute import MICE` 등 사용                                                              |

---

### 3. 이상치 파악하기 + 처리하기

### 1. Z-score 방식

#### 개념
Z-score는 각 데이터가 평균에서 얼마나 떨어져 있는지를 표준편차 기준으로 나타낸 값

공식:

```
z = (x - μ) / σ
```

- x: 개별 데이터
- μ: 평균
- σ: 표준편차

Z-score의 절댓값이 3보다 크면 이상치로 간주 (정규분포 가정)

#### 코드 예시

```python
from scipy.stats import zscore

# 수치형 데이터 선택
numeric_df = df.select_dtypes(include='number')

# Z-score 계산
z_scores = zscore(numeric_df)

# 이상치 탐지
outliers = (abs(z_scores) > 3)

# 이상치 포함 행 확인
df[outliers.any(axis=1)]
```

#### 특징
- 정규분포를 가정함
- 평균과 표준편차에 민감하여 극단값이 영향을 줄 수 있음

---

### 2. IQR 방식

#### 개념
IQR(Interquartile Range)은 데이터의 중간 50% 범위를 나타냄

- Q1: 1사분위수 (25%)
- Q3: 3사분위수 (75%)
- IQR = Q3 - Q1

이상치는 다음 범위를 벗어난 값입니다:

```
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

#### 단일 열 코드 예시

```python
# 'A' 열의 이상치 탐지
Q1 = df['A'].quantile(0.25)
Q3 = df['A'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = (df['A'] < lower) | (df['A'] > upper)

# 이상치 행 확인
df[outliers]
```

#### 전체 열 적용 예시

```python
outlier_indices = []

for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_cond = (df[col] < lower) | (df[col] > upper)
    outlier_indices.extend(df[outlier_cond].index)

outlier_indices = list(set(outlier_indices))
df.loc[outlier_indices]
```

#### 특징
- 정규분포 가정이 필요 없음
- 평균과 표준편차에 영향받지 않음
- 극단값에 덜 민감하며 비대칭 분포에도 적합

---

### 5. 구간화 하기 (Binning)

#### 1) 동일 길이로 구간화하기 (`pd.cut`)

#### 개념
- 전체 데이터의 최솟값부터 최댓값까지를 같은 간격으로 나누는 방식입니다.
- 데이터의 분포는 고려하지 않으며, 구간의 길이가 동일합니다.

#### 예시

```python
pd.cut(df['A'], bins=3, labels=['Low', 'Medium', 'High'])
```

예를 들어, df['A'] 값이 0~30이라면 아래처럼 10 단위로 나뉩니다.

| 원래 값 | 구간       | 결과 라벨 |
|---------|------------|-----------|
| 5       | (0, 10]    | Low       |
| 17      | (10, 20]   | Medium    |
| 28      | (20, 30]   | High      |

#### 특징
- 구간의 너비는 일정하지만, 각 구간에 속하는 데이터 수는 다를 수 있습니다.
- 데이터가 특정 구간에 몰릴 수 있습니다.

---

#### 2) 동일 개수로 구간화하기 (`pd.qcut`)

#### 개념
- 데이터를 동일한 개수로 나누는 분위수 기반 구간화 방법입니다.
- 구간의 크기는 다르지만 각 구간에 속한 데이터 수는 동일합니다.

#### 예시

```python
pd.qcut(df['A'], q=3, labels=['Q1', 'Q2', 'Q3'])
```

예를 들어, df['A']의 값이 아래와 같고 총 9개라고 가정하면:

| 원래 값 | 분위수 기준 | 결과 라벨 |
|---------|--------------|------------|
| 5       | 하위 1/3     | Q1         |
| 17      | 중간 1/3     | Q2         |
| 28      | 상위 1/3     | Q3         |

#### 특징
- 데이터 개수를 균등하게 분할합니다.
- 동일한 값이 많은 경우 구간 경계가 겹쳐 오류가 발생할 수 있습니다.

---

#### cut vs qcut 비교 정리

| 항목            | `pd.cut` (동일 길이)         | `pd.qcut` (동일 개수)         |
|------------------|------------------------------|-------------------------------|
| 기준             | 값의 범위 기준               | 분위수 기준                  |
| 구간 길이        | 일정함                        | 유동적                        |
| 구간당 데이터 수 | 불균형 가능                   | 균등하게 분할                |
| 사용 시기        | 일정 구간 기준이 필요한 경우 | 데이터 분포 균형이 중요한 경우 |

---

## Section 02. 범주형 데이터 정제하기

### 1. 레이블 인코딩 (Label Encoding)

#### **개념**
각 범주형 값을 **정수(integer)** 로 매핑하는 방식

예: ['Red', 'Green', 'Blue'] → [2, 1, 0]

#### **특징**
간단하고 저장공간 효율적

서열(순서) 있는 범주형 변수에 적합

**모델이 숫자의 크기에 의미를 부여할 수 있어 주의** 필요
(예: Red(2) > Blue(0)처럼 잘못 해석 가능)

#### 1) 판다스에서 레이블 인코딩하기

```python
df['label'] = df['category'].astype('category').cat.codes
```

#### 2) 사이킷런으로 레이블 인코딩하기

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
```

---

### 2. 원핫 인코딩 (One-Hot Encoding)

#### **개념**
각 범주형 값을 이진 벡터로 변환

값의 수만큼 새로운 열을 만들고, 해당하는 열만 1로 표시

예: ['Red', 'Green', 'Blue'] → [1, 0, 0], [0, 1, 0], [0, 0, 1]

#### **특징**
범주 간 **서열이 없을 때** 사용 (대부분의 범주형 변수)

모델이 잘못된 순서 관계를 학습하지 않도록 방지

차원이 늘어나는 단점 있음 (특히 범주 수 많을 경우)

#### 1) 판다스에서 원핫 인코딩하기

```python
pd.get_dummies(df, columns=['category'])
```

#### 2) 사이킷런으로 원핫 인코딩하기

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['category']])
```

| 구분       | 레이블 인코딩          | 원-핫 인코딩            |
| -------- | ---------------- | ------------------ |
| 값 변환 방식  | 범주 → 정수          | 범주 → 이진 벡터(여러 열)   |
| 서열 정보 유지 | O                | X                  |
| 차원 수 증가  | X (1열 유지)        | O (범주 수만큼 열 증가)    |
| 모델 적합성   | Tree 계열 모델 적합    | 선형 모델/거리 기반 모델에 적합 |
| 주의사항     | 숫자 크기를 순서로 오해 가능 | 차원의 저주 가능성 있음      |


---

## Section 03. 스케일링하기

### 1. 정규화 (Min-Max Scaling)

* 모든 값을 0\~1 사이로 조정
* 공식: (x - min) / (max - min)

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['A']])
```

### 2. 표준화 (Standardization)

* 평균 0, 표준편차 1로 조정
* 공식: (x - 평균) / 표준편차

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_standard = scaler.fit_transform(df[['A']])
```

---

## Section 04. 변수 선택하기 (Feature Selection)

### 1. RFE (Recursive Feature Elimination)

* 모델이 중요하지 않다고 판단한 피처를 반복적으로 제거하여 최적의 피처 subset을 선택
* 보통 성능이 떨어지지 않으면서도 피처 수를 줄일 수 있음

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X, y)
```

### 2. RFE-CV (RFE with Cross-Validation)

* RFE 과정에서 교차 검증을 수행하여 가장 성능이 좋은 피처 조합을 자동으로 선택
* 하이퍼파라미터 튜닝 없이 안정적 선택 가능

```python
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=model, cv=5)
X_rfecv = rfecv.fit_transform(X, y)
```

### 3. UFS (Univariate Feature Selection)

* 각 피처를 독립적으로 통계 검정 후 우수한 피처만 선택
* chi2 (카이제곱), f\_classif 등의 함수 사용 가능

```python
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)
```

---
