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

```python
# 평균으로 대체
mean_A = df['A'].mean()
df['A'].fillna(mean_A, inplace=True)

# 최빈값으로 대체
mode_B = df['B'].mode()[0]
df['B'].fillna(mode_B, inplace=True)
```

---

### 3. 이상치 파악하기

#### 1) z-score로 확인하기

```python
from scipy.stats import zscore
z_scores = zscore(df.select_dtypes(include='number'))
outliers = (abs(z_scores) > 3)
```

#### 2) IQR로 확인하기

```python
Q1 = df['A'].quantile(0.25)
Q3 = df['A'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['A'] < Q1 - 1.5 * IQR) | (df['A'] > Q3 + 1.5 * IQR)
```

---

### 4. 이상치 처리하기

#### 1) 이상치 데이터 삭제하기 (Z-score 기준)

```python
from scipy.stats import zscore
df_z = df[(np.abs(zscore(df.select_dtypes(include='number'))) < 3).all(axis=1)]
```

#### 2) 이상치 데이터 대체하기 (IQR 기준 함수)

```python
def replace_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = series.median()
    return series.apply(lambda x: median if x < lower_bound or x > upper_bound else x)

df['A'] = replace_outliers_iqr(df['A'])
```

---

### 5. 구간화 하기 (Binning)

#### 1) 동일 길이로 구간화하기

```python
pd.cut(df['A'], bins=3, labels=['Low', 'Medium', 'High'])
```

#### 2) 동일 개수로 구간화하기

```python
pd.qcut(df['A'], q=3, labels=['Q1', 'Q2', 'Q3'])
```

---

## Section 02. 범주형 데이터 정제하기

### 1. 레이블 인코딩 (Label Encoding)

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
