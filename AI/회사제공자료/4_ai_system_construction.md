# AI 시스템 구축 (심화 강의)

## 목차

- Part 1: ML/AI Pipeline 설계 및 배포
  - MLOps의 철학과 성숙도 모델
- 1.1.1 전통적 소프트웨어 개발 vs ML 시스템 개발
- 1.1.2 MLOps 성숙도 모델
- 1.1.3 MLOps의 핵심 구성요소
- 1.2.1 MLOps 데이터 파이프라인
- 1.2.2 데이터 품질 관리와 검증 자동화
- 1.2.3 MLOps 모델 훈련 파이프라인
- 1.2.4 모델 배포 전략
- 1.3.1 LLM Pipeline의 특수성과 도전과제
- 1.3.2 LLM 서빙 인프라 최적화
- 1.3.3 LLM 배포의 도전과 해결책
- Part 2: AI 시스템 모니터링 및 자동화
  - Model Drift의 유형과 수학적 정의
- 2.1.1 Drift의 유형
- 2.1.2 Drift 탐지 방법론 심층 분석
- 2.1.3 Drift의 시간적 패턴
- 2.2.1 분포 시각화와 해석
- 2.2.2 모델 성능 모니터링 지표
- 2.3.1 Model Drift 개선 방안: 예방적 접근법
- 2.3.2 Model Drift 개선 방안: 반응적 접근법
- Part 3: AI 시스템 최적화
- 3.1.1 Gradient Vanishing과 Exploding
- 3.1.2 활성화 함수 최적화
- 3.1.3 정규화 기법의 진화
- 3.1.4 네트워크 아키텍처 최적화
- 3.2.1 손실함수 그래프의 기하학적 해석
- 3.2.2 최적화 알고리즘의 진화
- 3.3.1 GAN의 게임 이론적 기반
- 3.3.2 GAN 품질 평가 지표
- 3.3.3 GAN 성능 향상 기법

# AI 시스템 구축

"실험실에서 현실로": 프로덕션 레벨의 안정적이고 효율적인 AI 시스템

## Part 1: ML/AI Pipeline 설계 및 배포

### MLOps의 철학과 성숙도 모델

MLOps는 머신러닝(ML)과 운영(Operations)의 합성어로, ML 모델 개발과 배포, 운영을 자동화하고 표준화하는 문화이자 기술입니다.
 전통적인 DevOps와 달리 ML 시스템은 코드, 데이터, 모델이라는 세 가지 핵심 요소가 함께 변화하고 서로에게 영향을 주기 때문에,
 이를 체계적으로 관리하는 MLOps가 필수적입니다.

![MLOps Components](images/MLOps_Components.png)

## 1.1.1 전통적 소프트웨어 개발 vs ML 시스템 개발

#### 근본적 차이점의 이해

- 결정론적 vs 확률적: 전통적 SW는 결정론적 로직, ML 시스템은 확률적 추론에 기반.
- 코드 vs 데이터 중심성: 전통적 SW는 코드가 핵심, ML 시스템은 데이터가 핵심.
- 품질 보증의 복잡성: ML 시스템은 데이터 품질, 모델 성능, 분포 변화 등 다차원 검증 필요.
- 배포 후 동작의 예측성: ML 시스템은 데이터 분포 변화에 따른 성능 변동 가능.

## 1.1.2 MLOps 성숙도 모델

![MLOps Level](images/MLOps_Level.webp)

- Level 0 (Manual):
 모든 프로세스가 수동. 데이터 과학자가 실험, 훈련, 배포를 모두 직접 처리. 재현성이 없고 비효율적이며, 모델과 코드가 분리되어 관리가 어렵습니다.
- Level 1 (ML Pipeline Automation):
 모델 훈련 파이프라인을 자동화하여, 데이터가 업데이트되거나 코드가 변경되면 모델이 자동으로 재훈련되는 연속적 훈련(Continuous Training, CT)을 구현한 단계입니다.
- Level 2 (CI/CD Pipeline Automation):
 완전한 CI/CD 자동화. 코드 변경 시 파이프라인이 자동으로 소스 코드를 빌드, 테스트하고, 데이터와 모델을 검증하며, 최종적으로 프로덕션에 배포하는 가장 성숙한 단계입니다.

## 1.1.3 MLOps의 핵심 구성요소

- 데이터 관리 계층:
 데이터 수집, 저장, 버전 관리(DVC, Git LFS), 스키마 및 품질 검증(Great Expectations)을 담당합니다.
 데이터의 출처와 변경 이력을 추적하는 데이터 혈통(Data Lineage) 관리가 중요합니다.
- 모델 개발 계층:
 실험 관리(MLflow, W&B), 분산 훈련, 하이퍼파라미터 튜닝, 자동화된 평가,
 그리고 최종적으로 승인된 모델을 저장하고 관리하는 모델 레지스트리(Model Registry)를 포함합니다.
- 배포 및 서빙 계층:
 모델을 컨테이너화(Docker)하고, Kubernetes와 같은 오케스트레이션 도구를 사용하여 배포합니다.
 Blue-Green, Canary, Shadow 등 다양한 배포 전략을 통해 안정적인 서비스를 보장합니다.

![MLOps Components Sample](images/MLOps_Components_Sample.webp)

## 1.2.1 MLOps 데이터 파이프라인

#### ETL vs ELT 패턴

![ETL ELT](images/ETL_ELT.png)

- ETL (Extract, Transform, Load): 전통적 접근. Raw Data → Transform → ML-Ready Data → Load → Feature Store. 구조화된 데이터에 효과적.
- ELT (Extract, Load, Transform): 현대적 접근. Raw Data → Load → Data Lake → Transform → ML-Ready Data. 비구조화 데이터 처리에 유리.

#### 데이터 파이프라인의 핵심 원칙

- 재현 가능성 (Reproducibility): 동일한 입력에 대해 동일한 출력 보장, 데이터 혈통 추적.
- 점진적 처리 (Incremental Processing): 전체 데이터 재처리 대신 변경분만 처리, 리소스 효율성 확보.
- 데이터 품질 보증: 스키마 검증 자동화, 이상치 탐지 및 처리.

## 1.2.2 데이터 품질 관리와 검증 자동화

#### 데이터 품질 관리

- 완전성 (Completeness): 필수 데이터의 누락 여부, 결측치 비율 모니터링.
- 정확성 (Accuracy): 데이터 값의 정확성, 범위 검증, 형식 검증.
- 일관성 (Consistency): 데이터 간 논리적 일관성, 중복 데이터 검증.
- 적시성 (Timeliness): 데이터 업데이트 주기, 실시간성 보장.
- 유효성 (Validity): 비즈니스 규칙 준수, 도메인 제약 조건 검증.

#### Great Expectations를 이용한 데이터 검증

```
# Great Expectations 데이터 검증 예시
import great_expectations as ge
import pandas as pd

# 데이터 컨텍스트 생성
context = ge.get_context()

# 검증 스위트 생성
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite=expectation_suite_name
)

# 데이터 품질 검증 규칙 정의
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_between("age", 0, 120)
validator.expect_column_values_to_be_in_set("gender", ["M", "F", "Other"])
validator.expect_table_row_count_to_be_between(1000, 10000)

# 검증 실행
results = validator.validate()
```

## 1.2.3 MLOps 모델 훈련 파이프라인

#### 하이퍼파라미터 최적화 자동화

- 베이지안 최적화: 이전 평가 결과를 바탕으로 다음 탐색할 하이퍼파라미터 조합을 예측하여 효율적으로 최적점을 찾습니다.
- Population-Based Training (PBT): 여러 모델을 병렬로 훈련하며 성능 기반으로 하이퍼파라미터를 동적으로 조정.

![Population-Based Training](images/PBT.jpeg)

#### 분산 훈련 전략

- 데이터 병렬화 (Data Parallelism): 동일 모델을 여러 디바이스에 복제, 배치를 분할하여 병렬 처리.
- 모델 병렬화 (Model Parallelism): 모델을 여러 부분으로 분할, 각 부분을 다른 디바이스에 배치.
- Pipeline 병렬화: 모델을 시간축으로 분할, Micro-batch를 통한 처리량 향상.

## 1.2.4 모델 배포 전략

#### 배포 패턴의 이해

- Blue-Green 배포: 두 개의 동일한 프로덕션 환경 운영, 한쪽에 새 버전 배포 후 트래픽 전환. 빠른 롤백 가능.
- Canary 배포: 트래픽의 일부만 새 버전으로 라우팅, 점진적으로 트래픽 비율 증가. 리스크 최소화.
- A/B 테스트 배포: 사용자 세그먼트별 다른 모델 버전, 통계적 유의성 검증.
- Shadow 배포: 새 모델이 기존 모델과 병렬 실행, 실제 트래픽으로 테스트 (결과는 로깅만). 프로덕션 영향 없이 성능 검증.

![Blue Green Deployment](images/Blue_Green_Deployment.png)

## 1.3.1 LLM Pipeline의 특수성과 도전과제

LLM은 기존 ML 모델과 비교할 수 없는 규모와 복잡성으로 인해 MLOps에 새로운 차원의 과제를 제시합니다.

![LLMOps](images/LLMOps.png)

![LLM Pipeline](images/LLM_Pipeline.png)

## 1.3.2 LLM 서빙 인프라 최적화

거대한 모델 크기로 인한 추론 병목 현상(지연 시간 및 메모리)을 해결하기 위한 특수 기술이 요구됩니다.

#### 최적화 기법

- Dynamic Batching: 가변 길이 시퀀스를 효율적으로 처리하여 패딩을 최소화하고 계산 효율성을 높입니다.
- KV-Cache 최적화:
 이전 토큰의 계산 결과를 캐싱하여 생성 속도를 높이지만, 시퀀스 길이에 따라 메모리 사용량이 급증하여 정교한 관리가 필요합니다.

![KV-Cache Optimization](images/KV_Cache.gif)

![KV-Cache](images/KV_Cache_1.gif)

## 1.3.3 LLM 배포의 도전과 해결책

#### 모델 샤딩과 분산 추론

- Tensor Parallelism in Inference: 가중치 행렬을 여러 GPU에 분산하여 메모리 한계를 극복하고 대규모 모델 서빙을 가능하게 합니다.
- Pipeline Parallelism: 모델의 레이어를 여러 디바이스에 배치하여 파이프라인 형태로 처리, 메모리 효율성 증대.

#### 메모리 관리 및 추론 비용 최적화

- Memory Pool 관리: KV-Cache를 위한 메모리 사전 할당 및 재사용.
- Request Scheduling: 우선순위 기반 요청 처리, 배치 크기 동적 조정.
- Model Quantization for Serving: INT8/INT4 양자화를 통해 메모리 절약 및 추론 가속화.

## Part 2: AI 시스템 모니터링 및 자동화

### Model Drift의 유형과 수학적 정의

Model Drift는 배포된 모델의 성능이 시간이 지남에 따라 저하되는 현상으로, AI 시스템의 수명을 결정하는 핵심 요소입니다.
 Drift는 모델의 잘못이 아니라, 모델이 학습한 세상과 현재 세상이 달라졌기 때문에 발생합니다.

## 2.1.1 Drift의 유형

![Drift Types](images/Drift_Types.png)

- Covariate Shift (공변량 이동):
 입력 데이터(X)의 분포가 변하는 경우. P_train(X) ≠ P_prod(X). (e.g., 새로운 사용자의 유입, 계절에 따른 상품 추천 패턴 변화)
- Prior Probability Shift (사전 확률 이동):
 타겟 변수(Y)의 분포가 변하는 경우. P_train(Y) ≠ P_prod(Y). (e.g., 경제 상황에 따른 사기 거래 비율 자체의 변화)
- Concept Drift (개념 이동):
 데이터와 정답 간의 관계(매핑 함수) 자체가 변하는 가장 심각하고 근본적인 유형. P_train(Y|X) ≠ P_prod(Y|X).
 (e.g., 새로운 경쟁자 등장으로 인한 고객 이탈의 핵심 요인 변화)

## 2.1.2 Drift 탐지 방법론 심층 분석

#### 통계적 검정 기반 방법

- Kolmogorov-Smirnov (KS) Test: 두 연속형 분포의 누적 분포 함수(CDF) 간의 최대 거리를 측정하여 분포의 동일성을 검정합니다.
- Chi-Square Test: 두 범주형 변수의 관측 빈도와 기대 빈도를 비교하여 분포의 동일성을 검정합니다.
- Population Stability Index (PSI):
 변수 분포 변화를 하나의 지표로 정량화하여, 변화의 심각도를 [안정(<0.1), 주의(0.1-0.25), 불안정(>0.25)]으로 직관적으로 파악하게 해줍니다. 금융권에서 널리 사용됩니다.

#### 거리 기반 방법 (고차원 데이터에 효과적)

- Maximum Mean Discrepancy (MMD): 두 분포를 고차원 공간에 매핑하여 평균값의 거리를 측정합니다.
- Wasserstein Distance (Earth Mover's Distance): 한 분포를 다른 분포로 바꾸는 데 필요한 최소 비용(거리)을 측정하여, 분포의 형태까지 고려합니다.

## 2.1.3 Drift의 시간적 패턴

- 급진적 Drift (Abrupt Drift): 특정 시점에서 갑작스러운 변화 (e.g., 시스템 업데이트, 정책 변화). Change Point Detection 기법 활용.
- 점진적 Drift (Gradual Drift): 시간에 따른 서서히 변화 (e.g., 사용자 행동, 시장 트렌드 변화). 추세 분석 및 회귀 기법 활용.
- 순환적 Drift (Recurring Drift): 주기적으로 반복되는 패턴 (e.g., 계절성, 요일별 패턴). 시계열 분석 및 푸리에 변환 활용.

## 2.2.1 분포 시각화와 해석

#### 히스토그램 오버레이 분석

- 기준선(훈련 데이터)과 현재 분포 비교, 모드(peak) 위치 변화, 분포의 폭(variance) 변화, 비대칭성(skewness) 변화 관찰.
- 주요 패턴: 모드 이동 (중심 경향 변화), 분산 증가 (데이터 품질 저하), 다중 모드 출현 (새로운 사용자 세그먼트), 긴 꼬리 형성 (이상치 증가).

#### Q-Q Plot (Quantile-Quantile Plot) 해석

- 정상 상태: 대각선 근처에 점들이 집중, 기울기 1에 가까운 선형 관계. 주어진 두 세트의 데이터가 같은 분포인 상황.
- Drift 상태: S자 곡선 (분산 차이), 평행 이동 (평균 차이), 비선형 패턴 (분포 형태 변화).

## 2.2.2 모델 성능 모니터링 지표

#### 비즈니스 지표와 기술 지표

- 비즈니스 지표: 매출, 전환율, 고객 만족도, 비용 절감 등 비즈니스 가치 직접 측정.
- 기술 지표: 정확도, 재현율, F1-score, AUC 등 모델 성능 측정.
- 시스템 지표: 지연시간, 처리량, 가용성, 리소스 사용률 등 인프라 성능.

#### 실시간 성능 모니터링

- 예측 품질 추적: 예측값과 실제값 비교, 오류 패턴 분석.
- 피드백 루프: 사용자 피드백 수집, 모델 성능 개선 반영.
- 성능 저하 조기 경보: 임계값 기반 알림, 자동 재훈련 트리거.

#### 실무 구현 예시

```
# Prometheus + Grafana 모니터링 설정
from prometheus_client import Counter, Histogram, Gauge
import time
from sklearn.metrics import accuracy_score

# 메트릭 정의
prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

def predict_with_monitoring(features):
    start_time = time.time()
    
    # 예측 수행
    prediction = model.predict(features)
    
    # 메트릭 업데이트
    prediction_counter.inc()
    prediction_latency.observe(time.time() - start_time)
    
    return prediction

# 정확도 업데이트 (배치 처리)
def update_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    model_accuracy.set(accuracy)
```

## 2.3.1 Model Drift 개선 방안: 예방적 접근법

#### 예방적 접근법 (Robust Model Design)

- Domain Adaptation: 훈련 데이터와 실제 데이터의 분포 차이를 모델이 스스로 학습하여 적응하도록 합니다.
 (e.g., Adversarial Training)
- Ensemble: 다양한 시점의 데이터로 학습된 여러 모델을 조합하여, 특정 시점의 변화에 덜 민감하게 만듭니다.
- Online Learning:
 들어오는 데이터 스트림에 대해 모델을 점진적으로 업데이트하여 항상 최신 상태를 유지합니다.
 다만, 과거의 중요한 정보를 잊어버리는 Catastrophic Forgetting 문제가 발생할 수 있습니다.

## 2.3.2 Model Drift 개선 방안: 반응적 접근법

#### 반응적 접근법 (Automated Retraining)

Drift가 탐지되었을 때 자동으로 대응하는 시스템을 구축하는 것이 MLOps의 핵심입니다.

1. 트리거 설정: 성능 지표(e.g., 정확도)나 Drift 지표(e.g., PSI)가 특정 임계값을 넘으면 재훈련 프로세스를 시작합니다.
1. 자동 재훈련:
 최신 데이터를 포함하여 모델을 자동으로 재훈련합니다. (Full Retraining, Incremental Learning, Transfer Learning)
1. A/B 테스트: 재훈련된 모델을 기존 모델과 비교 검증한 후, 성능이 더 좋은 모델로 안전하게 교체합니다.

## Part 3: AI 시스템 최적화

딥러닝 모델의 성능은 아키텍처, 활성화 함수, 정규화 등 다양한 요소의 복잡한 상호작용으로 결정됩니다.

## 3.1.1 Gradient Vanishing과 Exploding

역전파 과정에서 그래디언트가 깊은 층으로 갈수록 급격히 작아지거나(vanishing) 커지는(exploding) 현상으로, 깊은 신경망 학습을 방해하는 핵심 문제입니다.

#### 수학적 원인 분석

- 연쇄 법칙: ∂L/∂w_l = ∂L/∂y_L · ∏(k=l+1 to L) ∂y_k/∂y_{k-1} · ∂y_l/∂w_l
- Vanishing: 가중치가 1보다 작으면 지수적으로 감소 (0.5^10 ≈ 0.001)
- Exploding: 가중치가 1보다 크면 지수적으로 증가 (2^10 ≈ 1024)

#### 활성화 함수별 특성

- Sigmoid: 최대 기울기 0.25, 깊은 망에서 Vanishing 문제 심각
- Tanh: 최대 기울기 1.0, Sigmoid보다 나으나 여전히 문제
- ReLU: 양수 구간에서 기울기 1.0, Vanishing 문제 해결

#### 해결책

- 가중치 초기화: Xavier/Glorot, He 초기화로 적절한 분산 설정
- 정규화: Batch Normalization, Layer Normalization
- Skip Connection: ResNet의 잔차 연결로 그래디언트 직통로 제공

## 3.1.2 활성화 함수 최적화

#### 활성화 함수의 진화

- ReLU (Rectified Linear Unit): f(x) = max(0, x). 양수 구간에서 기울기 1 유지, 계산 복잡도 낮음. Dead ReLU 문제 발생 가능.
- Leaky ReLU: f(x) = max(αx, x), α=0.01. Dead ReLU 문제 부분적 해결, 음수 영역에서도 작은 기울기 유지.
- PReLU (Parametric ReLU): α를 학습 가능한 파라미터로 설정, 데이터에 맞는 최적의 음수 기울기 학습.
- ELU (Exponential Linear Unit): f(x) = x if x > 0 else α(e^x - 1). 평균 0에 가까운 출력, 부드러운 음수 영역.
- Swish/SiLU: f(x) = x · σ(x). 부드러운 비단조 함수, 실험적으로 우수한 성능.
- GELU (Gaussian Error Linear Unit): f(x) = x · Φ(x). Transformer에서 널리 사용, 더 부드러운 특성.

#### 활성화 함수 선택 가이드

- 일반적인 경우: ReLU 또는 Leaky ReLU
- 깊은 네트워크: ELU, Swish
- Transformer: GELU
- 생성 모델: Leaky ReLU, Swish

#### 실무 구현 예시

```
# PyTorch 활성화 함수 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomActivation(nn.Module):
    def __init__(self, activation_type='relu', alpha=0.01):
        super().__init__()
        self.activation_type = activation_type
        self.alpha = alpha
    
    def forward(self, x):
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'leaky_relu':
            return F.leaky_relu(x, self.alpha)
        elif self.activation_type == 'elu':
            return F.elu(x)
        elif self.activation_type == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation_type == 'gelu':
            return F.gelu(x)
        else:
            return F.relu(x)  # 기본값
```

## 3.1.3 정규화 기법의 진화

#### 정규화의 목적과 효과

- 내부 공변량 이동(Internal Covariate Shift) 해결: 레이어별 입력 분포 변화로 인한 학습 불안정성 완화.
- 과적합 방지: 모델의 복잡도 제한, 일반화 성능 향상.
- 학습 속도 향상: 더 높은 학습률 사용 가능, 빠른 수렴.

#### 정규화 기법 비교

- Batch Normalization: 배치 단위로 정규화, 배치 크기에 민감, CNN에서 효과적.
- Layer Normalization: 레이어 내에서 정규화, 배치 크기 무관, Transformer에서 표준.
- Group Normalization: 채널 그룹별 정규화, 작은 배치 크기에서 효과적.
- Instance Normalization: 각 인스턴스별 정규화, 스타일 전송에서 효과적.

#### 가중치 정규화

- L1 정규화 (Lasso): 희소성 유도, 특성 선택 효과.
- L2 정규화 (Ridge): 가중치 크기 제한, 과적합 방지.
- Elastic Net: L1 + L2 조합, 희소성과 정규화 균형.

#### 실무 구현 예시

```
# PyTorch 정규화 기법 구현
import torch
import torch.nn as nn

class NormalizedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 배치 정규화
            nn.ReLU(),
            nn.Dropout(0.3),  # 드롭아웃
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 레이어 정규화
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, output_size)
        )
        
        # L2 정규화 적용
        self.l2_reg = 0.01
    
    def forward(self, x):
        return self.layers(x)
    
    def l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss
```

## 3.1.4 네트워크 아키텍처 최적화

#### 아키텍처 설계 원칙

- 깊이 vs 너비의 균형: 깊은 네트워크는 복잡한 특성 학습, 너비는 병렬 처리 및 그래디언트 소실 완화.
- 효율성 vs 표현력: 매개변수 효율성과 모델 표현력의 최적 균형점 탐색.
- 확장성: 데이터와 컴퓨팅 리소스 증가에 따른 체계적 확장 가능성.

#### 현대적 아키텍처 패턴

- ResNet: Skip connection으로 그래디언트 직통로 제공, 깊은 네트워크에서도 안정적 학습.
- DenseNet: 모든 이전 레이어와 연결, 특성 재사용을 통한 매개변수 효율성.
- EfficientNet: Compound Scaling으로 깊이, 너비, 해상도를 균형있게 조정.
- Vision Transformer: Attention 메커니즘을 이미지에 적용, 패치 기반 처리.

#### 실무 구현 예시

```
# ResNet Block 구현
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Skip connection
        out = nn.ReLU()(out)
        
        return out
```

## 3.2.1 손실함수 그래프의 기하학적 해석

#### Loss Landscape의 특성

- Local Minima vs Global Minimum: 고차원 공간에서는 대부분 안장점(saddle point). SGD의 확률적 특성이 탈출 도움.
- Sharp vs Flat Minima: Flat minima가 일반화 성능에 유리. SAM (Sharpness-Aware Minimization) 기법.
- Loss Surface의 복잡성: 고차원 공간에서의 지형학적 특성, 경로 의존성.

#### 학습률과 수렴성

- Learning Rate Scheduling:
 Step Decay, Cosine Annealing, Exponential Decay 등 학습률을 동적으로 조절하여 안정적인 수렴 유도.
- Adaptive Learning Rate: Adam, AdaGrad, RMSprop 등 특성별 학습률을 조정하여 최적화 효율 증대.
- Warmup과 Cooldown: 학습 초기 안정화와 후기 정교화를 위한 전략.

#### 실무 구현 예시

```
# PyTorch 학습률 스케줄러 구현
import torch.optim as optim

# 모델과 옵티마이저 설정
# model = YourModel()  # 실제 모델 클래스로 대체
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 다양한 스케줄러
# Step Decay
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# One Cycle Policy
# scheduler3 = optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=0.01, epochs=100, steps_per_epoch=len(train_loader)
# )

# 학습 루프에서 사용
# for epoch in range(num_epochs):
#     for batch in train_loader:  # train_loader는 실제 데이터 로더로 대체
#         # 학습 단계
#         loss = train_step(batch)
#         loss.backward()
#         optimizer.step()
#         
#         # 스케줄러 업데이트 (One Cycle의 경우)
#         scheduler3.step()
#     
#     # 에포크 단위 스케줄러 업데이트
#     scheduler1.step()  # Step Decay
#     scheduler2.step()  # Cosine Annealing
```

## 3.2.2 최적화 알고리즘의 진화

#### 옵티마이저의 발전사

- SGD (Stochastic Gradient Descent): 기본적인 확률적 경사하강법, 단순하지만 수렴이 느림.
- Momentum: 이전 그래디언트의 방향성을 고려하여 진동 감소, 빠른 수렴.
- AdaGrad: 특성별 학습률 조정, 희소 데이터에 효과적.
- RMSprop: AdaGrad의 개선 버전, 지수이동평균 사용.
- Adam: Momentum + RMSprop 조합, 가장 널리 사용되는 옵티마이저.
- AdamW: Adam + Weight Decay 분리, 더 나은 정규화 효과.

#### 옵티마이저 선택 가이드

- 일반적인 경우: Adam 또는 AdamW
- 컴퓨터 비전: SGD with Momentum
- 자연어 처리: AdamW
- 생성 모델: Adam, RMSprop

#### 실무 구현 예시

```
# 다양한 옵티마이저 비교
import torch
import torch.optim as optim

# 모델 정의
# model = YourModel()  # 실제 모델 클래스로 대체

# 다양한 옵티마이저
optimizers = {
    'sgd': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'adam': optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)),
    'adamw': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    'rmsprop': optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
}

# 옵티마이저 선택
optimizer = optimizers['adamw']

# 학습 루프
# for epoch in range(num_epochs):
#     for batch in train_loader:  # train_loader는 실제 데이터 로더로 대체
#         optimizer.zero_grad()
#         loss = model(batch)
#         loss.backward()
#         
#         # 그래디언트 클리핑 (RNN에서 유용)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         
#         optimizer.step()
```

## 3.3.1 GAN의 게임 이론적 기반

GAN은 생성자(Generator)와 판별자(Discriminator)가 서로 경쟁하며 학습하는 미니맥스 게임(Minimax Game) 구조를 가집니다.

```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1-D(G(z)))]
```

- Nash 균형:
 경쟁자의 대응에 따라 각자 제일 합리적인 선택을 하여, 생성자와 판별자 서로가 더 이상 전략을 바꿀 이유가 없는 상태. 이론적으로 Generator가 실제 데이터 분포를 완벽 모사.
- 훈련의 불안정성: 실제로는 균형점 찾기 어려워 훈련이 매우 불안정하고 어렵습니다.

## 3.3.2 GAN 품질 평가 지표

- IS (Inception Score): 생성된 이미지의 품질과 다양성을 평가. 높을수록 좋은 성능.
- FID (Fréchet Inception Distance):
 실제 이미지 분포와 생성된 이미지 분포 간의 거리를 측정하여, 인간의 시각적 평가와 더 높은 상관관계를 보이는 신뢰도 높은 지표. 낮을수록 좋은 성능.

## 3.3.3 GAN 성능 향상 기법

- WGAN (Wasserstein GAN): 판별자를 1-Lipschitz 함수로 제약하고 Earth Mover's Distance를 손실 함수로 사용하여, 훈련 안정성을 획기적으로 개선했습니다.
- Progressive GAN: 저해상도 이미지 생성부터 시작하여 점진적으로 층을 추가하며 고해상도 이미지를 안정적으로 학습하는 전략.
- StyleGAN: 스타일과 콘텐츠를 분리하여 제어하는 혁신적인 생성자 구조를 통해, 생성된 이미지의 품질과 제어 가능성을 한 차원 높였습니다.
- Self-Attention GAN: Attention 메커니즘을 활용하여 장거리 의존성을 모델링하고 구조적 일관성을 향상.
