# Chapter 06. 데이터 전처리하기 — 총정리 + 실습 코드(통합본)

본 문서는 KICE AICE Associate 대비용으로, 데이터 전처리 전 범위를 실무 코드와 함께 정리한 자료입니다.  
목차: 수치형 정제 → 범주형 정제 → 스케일링 → 특징 공학 → 변수 선택(UFS/RFE/RFECV 포함) → 파이프라인/검증 → 체크리스트.

---

## 0) 공통 실습 준비(데이터/전처리 뼈대)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 예시 데이터: 사용자 데이터로 교체(data.csv, 'target' 컬럼 존재 가정)
df = pd.read_csv("data.csv")

# 타깃/피처 분리
y = df["target"]
X = df.drop(columns=["target"])

# 컬럼 타입 식별
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# 공통 전처리 파이프라인
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 1) 수치형 데이터 정제하기

### 1-1. 결측치 파악
- 결측 유형 가정(MCAR/MAR/MNAR), 컬럼별 결측 비율/패턴 확인.

```python
missing_count = X.isna().sum().sort_values(ascending=False)
missing_ratio = (X.isna().mean().sort_values(ascending=False) * 100).round(2)
print(missing_count.head(10))
print(missing_ratio.head(10), "%")
```

### 1-2. 결측치 처리(삭제/대치/보간/모델)
- 누수 방지: 전처리는 파이프라인 내부에서 학습셋에만 fit.

```python
from sklearn.impute import KNNImputer
import numpy as np

# ① 행/열 삭제(결측 적고 무작위일 때)
X_drop_row = X.dropna()
X_drop_col = X.dropna(axis=1)

# ② 단순 대치: 공통 전처리(preprocess)에서 수행됨(수치=median, 범주=most_frequent)

# ③ 그룹 기반 대치 예시(원본 df 사용)
if "region" in df.columns and "income" in df.columns:
    df["income"] = df.groupby("region")["income"].transform(
        lambda s: s.fillna(s.median())
    )

# ④ 시계열 보간 예시(시간축이 있고 등간격이 필요할 때)
if "date" in df.columns and "y" in df.columns:
    df_ts = df.sort_values("date").set_index("date")
    df_ts = df_ts.asfreq("D")
    df_ts["y"] = df_ts["y"].interpolate(method="time")

# ⑤ 모델 기반 대치(KNN)
knn_imp = KNNImputer(n_neighbors=5)
if num_cols:
    X[num_cols] = knn_imp.fit_transform(X[num_cols])
```

### 1-3. 이상치 탐지(IQR, Z-Score)
```python
import numpy as np

def iqr_bounds(s: pd.Series, k: float = 1.5):
    Q1, Q3 = s.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return Q1 - k*IQR, Q3 + k*IQR

if num_cols:
    col = num_cols[0]  # 예시 컬럼
    low, high = iqr_bounds(X[col])
    outlier_mask_iqr = (X[col] < low) | (X[col] > high)

    z = (X[col] - X[col].mean()) / X[col].std(ddof=0)
    outlier_mask_z = z.abs() > 3
```

### 1-4. 이상치 처리(제거/완화/변환)
```python
if num_cols:
    col = num_cols[0]
    low, high = iqr_bounds(X[col])
    X_clean = X[~((X[col] < low) | (X[col] > high))].copy()  # 제거
    X["winsor"] = X[col].clip(lower=low, upper=high)        # 윈저라이징
    X["log1p"] = np.log1p(X[col].clip(lower=0))             # 로그 변환(양수)
```

### 1-5. 구간화(Binning)
```python
if "age" in X.columns:
    X["age_bin_equal"] = pd.cut(X["age"], bins=5, labels=False, include_lowest=True)
    X["age_bin_quantile"] = pd.qcut(X["age"], q=4, labels=False, duplicates="drop")
    bins = [0, 19, 29, 39, 49, 120]; labels = ["<=19","20s","30s","40s","50+"]
    X["age_bin_domain"] = pd.cut(X["age"], bins=bins, labels=labels, right=True)
```

---

## 2) 범주형 데이터 정제하기

### 2-1. 레이블 인코딩(순서형)
```python
from sklearn.preprocessing import OrdinalEncoder

if "size" in X.columns:
    enc = OrdinalEncoder(categories=[["small","medium","large"]])
    X["size_ord"] = enc.fit_transform(X[["size"]])
```

### 2-2. 원핫 인코딩(명목형)
```python
# 공통 전처리(preprocess)의 cat_pipe에서 수행됨
from sklearn.preprocessing import OneHotEncoder
```

### 2-3. 고카디널리티(빈도/해싱)
```python
# 빈도 인코딩
if "zipcode" in X.columns:
    freq = X["zipcode"].value_counts()
    X["zipcode_freq"] = X["zipcode"].map(freq)

# 해싱 트릭(문자형 카테고리 → 고정 차원)
from sklearn.feature_extraction import FeatureHasher
if "zipcode" in X.columns:
    fh = FeatureHasher(n_features=64, input_type="string")
    hashed = fh.transform(X["zipcode"].astype(str))
    # hashed는 희소행렬(모델 입력 시 ColumnTransformer 등으로 결합)
```

---

## 3) 스케일링하기

### 3-1. 정규화(Min–Max)
```python
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
if num_cols:
    X[num_cols] = mm.fit_transform(X[num_cols])  # 주의: 실무에선 파이프라인에서 처리
```

### 3-2. 표준화(Standardization)
```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
if num_cols:
    X[num_cols] = sc.fit_transform(X[num_cols])  # 주의: 실무에선 파이프라인에서 처리
```

---

## 4) 특징 공학(Feature Engineering)

### 4-1. 날짜/시간 파생
```python
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
```

### 4-2. 비율·상호작용·롤링 통계
```python
import numpy as np

if set(["price","quantity"]).issubset(X.columns):
    X["price_per_unit"] = X["price"] / (X["quantity"].replace(0, pd.NA))
    X["pv_interaction"] = X["price"] * X["quantity"]

if "date" in df.columns and "sales" in df.columns:
    df = df.sort_values("date")
    df["sales_rolling_mean_7"] = df["sales"].rolling(window=7, min_periods=1).mean()
```

### 4-3. 텍스트 파생(TF–IDF) + 수치/범주 결합
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

if "review" in df.columns:
    text_col = "review"
    pipe_text = Pipeline([
        ("prep", ColumnTransformer([
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ], remainder="drop")),
        ("clf", LogisticRegression(max_iter=200))
    ])
    # pipe_text.fit(X_train, y_train)
```

---

## 5) 변수 선택하기(Feature Selection) — UFS, RFE, RFECV 포함

### 5-0. 개념 비교 요약
| 방법 | 핵심 | 장점 | 단점/주의 | 적합 상황 |
|---|---|---|---|---|
| UFS | 각 특성을 타깃과 단변량으로 점수화 후 상위 k 선택 | 빠름, 1차 스크리닝 | 상호작용 반영 제한 | 차원 축소 초벌 필터 |
| RFE | 모델 중요도/계수로 덜 중요한 특성부터 제거(재귀) | 다변량 효과 반영 | 반복 학습으로 비용 큼 | 모델 기반 선택 |
| RFECV | RFE + 교차검증으로 최적 특성 수 자동 탐색 | 일반화 지표 기반 | 가장 느림 | 최적 특성 수 자동 결정 |

공통 원칙: 데이터 누수 방지(반드시 Pipeline 내부에서 fit), 분류는 StratifiedKFold 권장.

### 5-1. UFS(Univariate Feature Selection)
```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

# 분류(연속형 특성): ANOVA F
pipe_ufs_cls = Pipeline([
    ("prep", preprocess),
    ("ufs", SelectKBest(score_func=f_classif, k=40)),
    ("clf", LogisticRegression(max_iter=1000))
])
pipe_ufs_cls.fit(X_train, y_train)
y_pred = pipe_ufs_cls.predict(X_test)
print("UFS(f_classif) report:\n", classification_report(y_test, y_pred))

# 분류(비음수 특성): chi2  -> 필요 시 MinMaxScaler로 비음수화
# pipe_ufs_chi2 = Pipeline([("prep", preprocess),
#                           ("ufs", SelectKBest(score_func=chi2, k=40)),
#                           ("clf", LogisticRegression(max_iter=1000))])

# 회귀: f_regression
pipe_ufs_reg = Pipeline([
    ("prep", preprocess),
    ("ufs", SelectKBest(score_func=f_regression, k=50)),
    ("reg", Ridge(alpha=1.0, random_state=42))
])
# pipe_ufs_reg.fit(X_train, y_train)
# y_hat = pipe_ufs_reg.predict(X_test)
# print("R2:", r2_score(y_test, y_hat), "MAE:", mean_absolute_error(y_test, y_hat))
```

### 5-2. RFE(Recursive Feature Elimination)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

base_est = LogisticRegression(max_iter=1000, solver="liblinear")

pipe_rfe = Pipeline([
    ("prep", preprocess),
    ("rfe", RFE(estimator=base_est, n_features_to_select=25, step=1)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_rfe.fit(X_train, y_train)
y_pred = pipe_rfe.predict(X_test)
print("RFE(LogReg) accuracy:", accuracy_score(y_test, y_pred))
```

트리 기반 추정기 예시:
```python
from sklearn.ensemble import RandomForestClassifier

est_tree = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
pipe_rfe_tree = Pipeline([
    ("prep", preprocess),
    ("rfe", RFE(estimator=est_tree, n_features_to_select=30, step=1)),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])
# pipe_rfe_tree.fit(X_train, y_train)
```

### 5-3. RFECV(RFE + Cross-Validation)
```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
    step=1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1
)

pipe_rfecv = Pipeline([
    ("prep", preprocess),
    ("rfecv", rfecv),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_rfecv.fit(X_train, y_train)
print("RFECV 최적 특성 수:", pipe_rfecv.named_steps["rfecv"].n_features_)
print("RFECV 선택 마스크 합계:", pipe_rfecv.named_steps["rfecv"].support_.sum())

y_proba = pipe_rfecv.predict_proba(X_test)[:, 1]
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))
```

요점:
- scoring은 문제 특성에 맞게 조정(불균형=roc_auc, average_precision 등).
- 대규모 데이터/특성일수록 시간이 오래 걸림 → step 조정, n_jobs 병렬화, 폴드 축소.

---

## 6) 전처리 파이프라인과 검증(누수 방지)

### 6-1. ColumnTransformer + 선택기 + 모델(분류)
```python
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

pipe_full = Pipeline([
    ("prep", preprocess),
    ("fs", SelectKBest(score_func=f_classif, k=40)),  # UFS→RFE/RFECV 대체 가능
    ("clf", LogisticRegression(max_iter=1000))
])

cv_scores = cross_val_score(pipe_full, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print("CV mean:", cv_scores.mean().round(4), "±", cv_scores.std().round(4))
```

### 6-2. 회귀 문제 예시
```python
from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.metrics import r2_score

pipe_full_reg = Pipeline([
    ("prep", preprocess),
    ("fs", SelectKBest(score_func=f_regression, k=50)),
    ("reg", Ridge(alpha=1.0, random_state=42))
])

# pipe_full_reg.fit(X_train, y_train)
# y_hat = pipe_full_reg.predict(X_test)
# print("R2:", r2_score(y_test, y_hat))
```

---

## 7) 알고리즘-전처리 매핑 요약

| 알고리즘 | 권장 전처리 |
|---|---|
| KNN, SVM, 로지스틱/선형회귀, PCA | 스케일링(표준화), 결측 대치, 원핫/순서 인코딩 |
| 트리·랜덤포레스트·GBM | 결측 대치, 인코딩(원핫/빈도/타깃), 스케일링 영향 적음 |
| 신경망 | 정규화/표준화, 인코딩, 규제(과적합 주의) |

---

## 8) 체크리스트(시험·실무 공통)

- 결측치: 수치=중앙값, 범주=최빈값 기본. 시계열=보간 검토
- 이상치: IQR/Z-Score 경계 계산과 임계 합리화
- 인코딩: 순서형=Ordinal, 명목형=One-Hot. 고카디널리티=빈도/해싱
- 스케일링: 파이프라인 내부에서 fit/transform 분리(누수 방지)
- 특징 공학: 날짜·비율·상호작용·롤링·TF–IDF
- 변수 선택: UFS→RFE→RFECV 순으로 정교화, CV로 성능 검증
- 리포팅: 선택 근거(계수/중요도/점수), CV 성능, 재현성(난수, 파이프라인 저장)
