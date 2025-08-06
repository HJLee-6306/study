# Chapter 06. 데이터 전처리하기

## Section 01. 수치형 데이터 정제하기

### 1. 결측치 파악하기
- `isnull()` 또는 `isna()` 사용
- `sum()`과 함께 쓰면 각 열의 결측치 개수 확인 가능
```python
df.isnull().sum()
```

### 2. 결측치 처리하기
- 삭제: `dropna()`
- 대체: `fillna()` + 평균, 중앙값, 최빈값 등
```python
df.dropna()
df['column'].fillna(df['column'].mean(), inplace=True)
```

### 3. 이상치 파악하기
- 시각화: boxplot, histogram
- 수치기반 (IQR)
```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['col'] < Q1 - 1.5 * IQR) | (df['col'] > Q3 + 1.5 * IQR)
```

### 4. 이상치 처리하기
- 삭제 또는 대체
```python
df.loc[outliers, 'col'] = df['col'].median()
```

### 5. 구간화하기
- `pd.cut()` 또는 `qcut()` 사용
```python
bins = [0, 30, 60, 100]
labels = ['Low', 'Mid', 'High']
df['binned'] = pd.cut(df['score'], bins=bins, labels=labels)
```

---

## Section 02. 범주형 데이터 정제하기

### 1. 레이블 인코딩
- LabelEncoder 사용
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['cat'])
```

### 2. 원핫 인코딩

#### 1) 판다스
```python
pd.get_dummies(df['cat'])
```

#### 2) 사이킷런
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df[['cat']])
```

---

## Section 03. 스케일링하기

### 1. 정규화
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(df[['col']])
```

### 2. 표준화
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(df[['col']])
```

---

## Section 04. 변수 선택하기

### 1. RFE
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)
```

### 2. RFECV
```python
from sklearn.feature_selection import RFECV
selector = RFECV(estimator=model, cv=5)
selector.fit(X, y)
```

### 3. UFS
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(X, y)
```
