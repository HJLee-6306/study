# Chapter 09. 비지도학습으로 AI 모델링하기

## Section 01. 차원 축소

### 1. 주성분 분석 (PCA, Principal Component Analysis)

#### 1) PCA 이해하기
- **목적**: 고차원 데이터를 저차원으로 축소해도 원래 데이터의 **분산(정보)**을 최대한 보존
- 고차원 데이터의 시각화, 노이즈 제거, 계산 효율성 향상
- 핵심 아이디어: 데이터의 **분산이 가장 큰 방향**을 찾고, 그 방향으로 축을 재설정
- **선형 차원 축소 기법**

수학적 개념:
1. 데이터 표준화
2. 공분산 행렬 계산
3. 공분산 행렬의 고유값·고유벡터 계산
4. 고유값이 큰 순서대로 주성분 선택
5. 선택한 주성분으로 데이터 변환

#### 2) PCA 실습

**(1) 합성 데이터 생성하기**
```python
import numpy as np
import pandas as pd

np.random.seed(0)
X = np.random.randn(100, 5)  # 5차원 데이터
df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
```

**(2) 데이터 표준화**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

**(3) PCA 수행**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("설명된 분산 비율:", pca.explained_variance_ratio_)
```

---

### 2. t-SNE (t-distributed Stochastic Neighbor Embedding)

#### 1) t-SNE 이해하기
- **목적**: 고차원 데이터의 **국소적인 구조**를 저차원(2D, 3D)에 보존하며 시각화
- PCA는 전체 분산 최대화에 초점, t-SNE는 가까운 점끼리의 관계를 보존
- **비선형 차원 축소** 기법 → 복잡한 데이터(이미지, 텍스트 임베딩 등)에 적합
- 계산 비용이 크고 하이퍼파라미터(학습률, perplexity)에 민감

#### 2) t-SNE 실습
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:,0], X_tsne[:,1])
plt.title("t-SNE Visualization")
plt.show()
```

---

## Section 02. 군집화

### 1. K-평균 군집화 (K-Means Clustering)

#### 1) 개념
- **비지도 학습**의 대표적인 군집화 알고리즘
- 데이터 포인트를 K개의 클러스터로 나누고, 각 클러스터 중심(centroid)을 반복적으로 업데이트
- 목적: 클러스터 내부의 **제곱 거리 합(SSE)** 최소화

#### 2) 작동 과정
1. K개의 초기 중심 무작위 선택
2. 각 데이터 포인트를 가장 가까운 중심에 할당
3. 각 클러스터의 평균 위치로 중심 재계산
4. 중심 변화가 거의 없을 때까지 반복

#### 3) 실습
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', c='red')
plt.title("K-Means Clustering")
plt.show()
```

---

### 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

#### 1) 개념
- **밀도 기반 군집화**
- 높은 밀도의 데이터 포인트들은 같은 클러스터로 묶음
- 장점: 클러스터 개수를 사전에 지정할 필요 없음, 임의 모양 클러스터 탐지 가능
- 단점: eps, min_samples 하이퍼파라미터에 민감

#### 2) 실습
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.title("DBSCAN Clustering")
plt.show()
```

---

### 3. 고객 세분화 모델 (RFM + K-Means)

#### 1) RFM 분석
- **Recency**: 최근 구매일 (작을수록 우수)
- **Frequency**: 구매 빈도 (클수록 우수)
- **Monetary**: 총 구매 금액 (클수록 우수)

```python
df['R'] = (today - df['last_purchase']).dt.days
df['F'] = df['purchase_count']
df['M'] = df['total_amount']
```

#### 2) RFM 점수화
```python
df['R_score'] = pd.qcut(df['R'], 5, labels=[5,4,3,2,1])
df['F_score'] = pd.qcut(df['F'], 5, labels=[1,2,3,4,5])
df['M_score'] = pd.qcut(df['M'], 5, labels=[1,2,3,4,5])
df['RFM_Score'] = df['R_score'].astype(int) + df['F_score'].astype(int) + df['M_score'].astype(int)
```

#### 3) K-Means를 이용한 고객 군집화
```python
X_rfm = df[['R', 'F', 'M']].values
kmeans = KMeans(n_clusters=4, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_rfm)
```

---

## 💡 시험 포인트 정리
- PCA와 t-SNE의 차이점 (선형/비선형, 목적)
- K-Means의 SSE 최소화 과정
- DBSCAN의 eps, min_samples 의미
- RFM 지표 정의
- 차원 축소와 군집화의 목적 차이
