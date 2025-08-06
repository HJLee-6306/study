# 변수 선택(Feature Selection) — RFE, RFECV, UFS 총정리

> 목적: 예측 성능을 유지·향상하면서 **불필요한 변수 제거** 및 **해석성·학습 효율**을 높이기 위함.  
> 범위: **UFS(Univariate Feature Selection)**, **RFE(Recursive Feature Elimination)**, **RFECV(RFE + Cross‑Validation)** 중심.

---

## 1. 개념 요약

| 방법 | 핵심 아이디어 | 전제/요구사항 | 장점 | 단점/주의 |
|---|---|---|---|---|
| **UFS (단변량 선택)** | 각 변수와 목표값의 **단일 통계적 관련성**을 점수화(예: `f_classif`, `chi2`, `mutual_info_*`) 후 상위 `k`개 선택 | 입력 스케일·분포·음수 값(chi2는 비음수) 고려 | 계산 가볍고 빠름, 초기 스크리닝에 적합 | 변수 간 상호작용 반영 한계, 과소/과대 선택 위험 |
| **RFE** | 선택한 **추정기(estimator)의 중요도/계수**를 근거로 **재귀적으로 덜 중요한 변수 제거** | 추정기가 `coef_` 또는 `feature_importances_` 제공 | 상호작용/다변량 효과 반영 가능 | k 탐색 필요, 계산량 큼(반복 학습), 추정기 선택 민감 |
| **RFECV** | RFE에 **교차검증**을 결합하여 **최적 특성 수를 자동 탐색** | CV 전략·스코어 선택 필요 | 특성 수 자동 결정, 일반화 성능 중심 | 계산량 가장 큼, 대규모 특성/데이터 시 시간 소요 큼 |

---

## 2. 적용 가이드

| 상황 | 권장 접근 | 이유 |
|---|---|---|
| 특성 수가 매우 많고 빠른 1차 거름망 필요 | **UFS(SelectKBest/Percentile)** | 계산 비용이 낮고 노이즈 특성 제거에 유용 |
| 모델이 선형/트리 등 특정 계수를 제공 | **RFE** | 모델 기반 중요도로 다변량 효과 반영 |
| 최적 특성 수를 자동으로 찾고 싶음 | **RFECV** | 교차검증을 통해 일반화 기준으로 선택 |
| 범주형 다수 + 수치형 혼합 | **ColumnTransformer + Pipeline**에서 UFS/RFE/RFECV 결합 | 누수 방지, 처리 일관성, 재현성 |
| 고카디널리티 범주 | 인코딩/해싱/빈도 인코딩 후 선택 | 차원 폭증 억제 후 선택 수행 |

**공통 원칙**  
- **데이터 누수 금지**: 선택기를 **Pipeline** 내부에 두고, 학습 데이터에서만 `fit`.  
- **스케일링 선행 여부**: 선형계열·거리기반 모델은 **표준화** 후 선택이 안정적.  
- **CV 스키마**: 분류는 `StratifiedKFold` 권장, 평가 지표는 문제 특성(불균형 등)에 맞춤.  

---

## 3. 실습 준비

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 예시 데이터 불러오기
df = pd.read_csv("data.csv")  # 사용자 데이터로 교체
y = df["target"]
X = df.drop(columns=["target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 4. UFS (Univariate Feature Selection)

### 4.1 분류(Classification): ANOVA F(연속 X) / chi2(비음수 X)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# 수치/범주 컬럼 식별
num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["int64","float64"]).columns.tolist()

# 파이프라인: 전처리 → UFS → 분류기
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

# ANOVA F: 연속형 수치 특성에 적합
pipe_ufs_f = Pipeline([
    ("prep", preprocess),
    ("ufs", SelectKBest(score_func=f_classif, k=30)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_ufs_f.fit(X_train, y_train)
y_pred = pipe_ufs_f.predict(X_test)
print("UFS(f_classif) report:\n", classification_report(y_test, y_pred))

# chi2: 비음수 특성에 적합(스케일·시프트 필요 시 주의)
pipe_ufs_chi2 = Pipeline([
    ("prep", preprocess),
    ("ufs", SelectKBest(score_func=chi2, k=30)),
    ("clf", LogisticRegression(max_iter=1000))
])

# 주의: chi2는 음수값 불가. 필요하면 MinMaxScaler 등을 사용해 비음수화.
# pipe_ufs_chi2.fit(X_train, y_train)
```

### 4.2 회귀(Regression): f_regression / mutual_info_regression
```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

pipe_ufs_reg = Pipeline([
    ("prep", preprocess),
    ("ufs", SelectKBest(score_func=f_regression, k=40)),
    ("reg", Ridge(alpha=1.0, random_state=42))
])
# pipe_ufs_reg.fit(X_train, y_train)
# y_hat = pipe_ufs_reg.predict(X_test)
# print("R2:", r2_score(y_test, y_hat), "MAE:", mean_absolute_error(y_test, y_hat))
```

**요점**  
- `k`는 CV로 조정하거나, 업무상 해석 가능한 상한(예: 상위 20~50개)을 먼저 적용.  
- 텍스트/희소행렬과 결합 시 `SelectKBest`는 희소 입력 지원 여부를 확인.  

---

## 5. RFE (Recursive Feature Elimination)

### 5.1 분류: Logistic Regression 기반
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

est = LogisticRegression(max_iter=1000, n_jobs=None, solver="liblinear", penalty="l2")

pipe_rfe = Pipeline([
    ("prep", preprocess),
    ("rfe", RFE(estimator=est, n_features_to_select=25, step=1)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe_rfe.fit(X_train, y_train)
y_pred = pipe_rfe.predict(X_test)
print("RFE(LogReg) accuracy:", accuracy_score(y_test, y_pred))
```

### 5.2 트리 기반 추정기 사용 예
```python
from sklearn.ensemble import RandomForestClassifier

est_tree = RandomForestClassifier(
    n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
)

pipe_rfe_tree = Pipeline([
    ("prep", preprocess),
    ("rfe", RFE(estimator=est_tree, n_features_to_select=30, step=1)),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])

# pipe_rfe_tree.fit(X_train, y_train)
```

**요점**  
- `n_features_to_select`는 사전 가설 또는 소규모 탐색으로 결정.  
- `step`은 한 번에 제거할 특성 수(또는 비율). 너무 크면 조잡, 너무 작으면 느림.  
- 추정기 선택이 결과에 큰 영향. 선형/트리 모두 시험해 비교 권장.  

---

## 6. RFECV (RFE + Cross‑Validation)

### 6.1 분류: StratifiedKFold + 최적 특성 수 자동 탐색
```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

base_est = LogisticRegression(max_iter=1000, solver="liblinear")

# 내부 선택(기저)을 위해 하나의 추정기를 사용하고,
# 최종 분류기는 동일/다른 모델로 구성 가능
rfecv = RFECV(
    estimator=base_est,
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
print("RFECV 지원 마스크 크기:", pipe_rfecv.named_steps["rfecv"].support_.sum())

y_proba = pipe_rfecv.predict_proba(X_test)[:, 1]
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))
```

**요점**  
- `scoring`은 불균형 데이터에선 `roc_auc`, `average_precision` 등으로 조정.  
- CV 분할전략에 따라 선택 결과가 변동 가능 → 고정 난수·충분한 폴드 수 권장.  
- 대규모 데이터/특성에서는 시간이 오래 걸리므로 병렬화(`n_jobs`)와 `step` 조정.  

---

## 7. 파이프라인 통합 패턴

### 7.1 ColumnTransformer + 선택기 + 모델(분류)
```python
from sklearn.model_selection import cross_val_score

pipe_full = Pipeline([
    ("prep", preprocess),
    ("fs", SelectKBest(score_func=f_classif, k=40)),  # 또는 RFE/RFECV로 교체
    ("clf", LogisticRegression(max_iter=1000))
])

cv_scores = cross_val_score(pipe_full, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print("CV mean:", cv_scores.mean().round(4), "±", cv_scores.std().round(4))
```

### 7.2 회귀 문제 예시
```python
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

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

## 8. 결과 해석 및 리포팅

| 항목 | 체크리스트 |
|---|---|
| 선택 근거 | 선택기의 점수/중요도/계수 근거 제시 |
| 재현성 | 난수 고정, 파이프라인 저장(`joblib.dump`) |
| 일반화 | CV 성능, 외부 검증셋 성능 동시 보고 |
| 안정성 | 반복 CV/부트스트랩으로 특성 선택의 변동성 확인 |
| 해석 | 선택된 특성의 도메인 타당성 검토(과적합 신호 배제) |

---

## 9. 자주 묻는 질문(FAQ)

1) **UFS만으로 충분한가?**  
- 노이즈 제거의 1차 필터로 유용하지만, **상호작용을 반영하지 못함**. RFE/RFECV 병행 권장.

2) **RFE 추정기는 무엇을 쓰나?**  
- 선형(로지스틱/릿지), 트리(랜덤포레스트/GBM) 등 데이터 특성과 해석 목적에 맞춰 선택.

3) **RFECV가 너무 느릴 때?**  
- `step` 증가, 폴드 축소, 샘플/특성 부분 샘플링, 병렬화(`n_jobs`) 적용.

4) **데이터 누수 예방 핵심?**  
- 선택기와 모든 전처리를 **Pipeline**에 넣어 **훈련 데이터로만 `fit`**.

---

## 10. 권장 기본값 요약(시작점)

| 문제 | 전처리 | 선택기 | 모델 | 스코어 |
|---|---|---|---|---|
| 이진 분류(불균형) | 결측 대치+스케일+원핫 | **RFECV(LogReg, step=1, cv=5)** | LogReg/CatBoost | `roc_auc` |
| 다중 분류 | 결측 대치+스케일+원핫 | **UFS(f_classif, k=30~100)** → RFE 재점검 | LogReg/LightGBM | `accuracy`/`macro_f1` |
| 회귀 | 결측 대치+스케일 | **UFS(f_regression, k=40~100)** → RFE | Ridge/GBM | `r2`/`neg_mae` |

---

## 11. 최소 예제 요약(복사·실행용)

```python
# 1) UFS
pipe = Pipeline([("prep", preprocess),
                 ("ufs", SelectKBest(score_func=f_classif, k=40)),
                 ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)

# 2) RFE
pipe = Pipeline([("prep", preprocess),
                 ("rfe", RFE(estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
                             n_features_to_select=25)),
                 ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)

# 3) RFECV
pipe = Pipeline([("prep", preprocess),
                 ("rfecv", RFECV(estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
                                  step=1, cv=5, scoring="roc_auc", n_jobs=-1)),
                 ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X_train, y_train)
```

---

### 참고
- 모든 코드는 **scikit‑learn** 표준 API 사용. 실제 데이터 스키마에 맞게 컬럼명·스코어·모델을 조정할 것.
- 텍스트·이미지 등 고차원 희소 특성에는 선택기의 입력 형식(밀집/희소) 호환성을 확인할 것.
