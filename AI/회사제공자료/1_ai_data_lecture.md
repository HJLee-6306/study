# AI 데이터에 대한 이해 (심화 강의)

## 목차

- Part 1: AI 학습용 데이터 수집
  - 데이터 수집의 중요성과 철학
- 1.1.1 데이터 중심 AI 패러다임 전환
- 1.1.2 "Garbage In, Garbage Out"
- 1.2.1 웹 크롤링 : 대규모 데이터 수집
- 1.2.2 웹 크롤링 : 윤리적 및 기술적 이슈
- 1.2.3 API 활용 : 구조화된 데이터 수집
- 1.2.4 크라우드소싱을 통한 데이터 수집
- 1.2.5 시뮬레이션 및 합성 데이터 생성
- 1.2.6 IoT 센서 데이터 수집의 특수성
- 1.3.1 수동 라벨링
- 1.3.2 자동 라벨링과 약한 지도학습
- 1.3.3 반자동 라벨링과 Active Learning
- 1.3.4 Self-Supervised Learning
- 1.4.1 개인정보 보호의 다층적 접근
- 1.4.2 편향성 방지를 위한 체계적 접근
- 1.4.3 데이터 거버넌스 및 품질 관리 체계
- Part 2: 데이터 정제
  - 결측치 처리 이해
- 2.1.1 결측치 발생 메커니즘 분석
- 2.1.2 결측치 처리 방법의 선택 기준
- 2.1.3 이상치 탐지 및 처리
- 2.1.4 결측치 처리 라이브러리 활용 가이드
- 2.1.5 결측치 처리 성능 최적화와 베스트 프랙티스
- 2.2.1 시계열 데이터 처리의 고유 특성
- 2.2.2 이미지 데이터 처리의 심화 이해
- 2.2.3 텍스트 데이터 처리의 언어학적 접근
- 2.3.1 DAE (Denoising Autoencoder)
- 2.3.2 DAE : 노이즈 타입별 특성과 효과
- 2.3.3 DAE : 응용 분야별 특화 전략
- Part 3: 데이터 증강
  - 데이터 부족 문제
- 3.1.1 Long-tail 분포의 이해와 대응
- 3.1.2 도메인별 데이터 부족의 특수성
- 3.2.1 클래스 불균형 문제의 심화 분석
- 3.3.1 기본 데이터 증강 기법
- 3.4.1 Mixup 기법
- 3.4.2 도메인별 Mixup 적용의 특수성
- 3.5.1 생성 모델 기반 증강
- 3.5.2 데이터 증강의 윤리적 이슈와 한계

# AI에 사용되는 데이터에 대한 이해

데이터 중심 AI 접근법: "데이터가 모델을 만든다"

AI 학습용 데이터의 수집, 정제, 증강

## Part 1: AI 학습용 데이터 수집

### 데이터 수집의 중요성과 철학

- AI 개발의 핵심 패러다임 변화: 모델 중심에서 데이터 중심으로
- "데이터가 모델을 만든다": 고품질 데이터의 확보와 체계적 관리의 중요성 증대

## 1.1.1 데이터 중심 AI 패러다임 전환

현대 AI 개발에서 가장 중요한 변화 중 하나는 모델 중심 접근법에서 데이터 중심 접근법으로의 전환입니다.
 전통적으로는 더 복잡하고 정교한 모델 아키텍처를 설계하는 것이 성능 향상의 핵심이었다면, 현재는 고품질 데이터의 확보와 체계적인 관리가 더욱 중요해졌습니다.

#### 데이터 중심 AI의 핵심 원칙

- 일관성 (Consistency):
 같은 입력에 대해 항상 같은 라벨이 부여되어야 합니다. (e.g., 동일한 이미지에 대해 항상 동일한 객체 라벨)
- 정확성 (Accuracy):
 라벨이 실제 정답(Ground Truth)과 일치해야 합니다. (e.g., 고양이 이미지에 '고양이' 라벨)
- 대표성 (Representativeness):
 데이터가 실제 운영 환경의 다양성과 분포를 충실히 반영해야 합니다. (e.g., 다양한 조명, 각도, 배경의 이미지 포함)
- 완전성 (Completeness):
 모델이 학습해야 할 모든 중요한 케이스(일반적인 경우부터 엣지 케이스까지)를 포함해야 합니다. (e.g., 자율주행 데이터에 희귀한 교통 상황 포함)

## 1.1.2 "Garbage In, Garbage Out"

이 원칙은 단순히 나쁜 데이터가 나쁜 결과를 낳는다는 표면적 의미를 넘어, AI 시스템의 근본적 특성을 설명합니다.
 AI 모델은 인간처럼 상식적 추론을 통해 데이터의 오류를 보정할 수 없으며, 오히려 데이터의 패턴을 그대로 학습하고 증폭시킵니다.

#### 데이터 품질이 모델 성능에 미치는 구체적 영향

- 편향 증폭 (Bias Amplification):
 데이터에 존재하는 사회적, 역사적 편견(e.g., 특정 성별, 인종에 대한 편견)이 모델의 예측 편향으로 직결되어 차별적인 결과를 초래합니다.
- 일반화 실패 (Generalization Failure):
 편협한 데이터로 학습된 모델은 실제 환경에 적용될 때 성능이 급격히 저하되는 한계를 보입니다.
- 신뢰성 저하:
 일관성 없는 라벨링이나 노이즈가 많은 데이터는 모델 예측의 불안정성을 높여, 시스템의 신뢰도를 떨어뜨립니다.
- 공정성 문제:
 특정 집단에 편향된 데이터로 학습된 모델은 해당 집단에 대해 불공정한 결과를 도출할 수 있습니다.

## 1.2.1 웹 크롤링 : 대규모 데이터 수집

웹 크롤링은 인터넷이라는 거대한 정보 저장소에서 구조화되지 않은 데이터를 체계적으로 수집하는 방법입니다.
 이는 전통적인 설문조사나 실험 기반 데이터 수집과 달리, 자연스럽게 생성된 데이터(naturally occurring data)를 활용할 수 있다는 장점이 있습니다.

#### 크롤링 대상별 특성과 전략

- 정적 웹페이지 크롤링:
 HTML 구조가 고정적이어서 파싱이 상대적으로 용이합니다. 뉴스 기사, 블로그 포스트, 제품 정보 등에 적합하며,
 `BeautifulSoup`BeautifulSoup, `Scrapy`Scrapy 같은 라이브러리가 주로 사용됩니다. 페이지 구조 변경에 대한 모니터링과 적응이 필요합니다.
- 동적 웹페이지 크롤링:
 JavaScript로 콘텐츠가 동적으로 생성되므로, 브라우저 엔진 시뮬레이션(Headless Browser)을 통한 렌더링 후 데이터 추출이 필요합니다.
 소셜 미디어, 전자상거래 사이트 등에서 흔하며, `Selenium`Selenium, `Playwright`Playwright 같은 도구가 사용됩니다.

## 1.2.2 웹 크롤링 : 윤리적 및 기술적 이슈

웹 크롤링은 방대한 데이터를 얻을 수 있지만, 법적, 윤리적, 기술적 제약사항을 반드시 준수해야 합니다.

- 로봇 배제 표준 (robots.txt) 준수:
 웹사이트 운영자의 크롤링 정책을 존중하고, 허용된 영역과 금지된 영역을 구분해야 합니다. 크롤링 빈도 제한도 준수해야 합니다.
- 서버 부하 관리:
 과도한 요청은 대상 서버에 심각한 부하를 주어 서비스를 방해할 수 있습니다. 요청 간격 조절(`time.sleep()`time.sleep()),
 동시 연결 수 제한, 피크 시간대 크롤링 회피 등을 통해 서버 부하를 최소화해야 합니다.
- 법적 제약사항:
저작권법: 크롤링한 콘텐츠의 상업적 이용 시 저작권 침해 문제가 발생할 수 있습니다.
개인정보보호법: 개인 식별이 가능한 정보(PII)를 수집할 경우 법적 문제가 발생할 수 있으므로, 익명화/가명화 처리가 필수적입니다.
웹사이트 이용약관: 각 웹사이트의 이용약관을 검토하여 크롤링이 허용되는지 확인해야 합니다.

## 1.2.3 API 활용 : 구조화된 데이터 수집

API(Application Programming Interface)를 활용한 데이터 수집은 웹 크롤링과 달리, 제공자가 의도적으로 구조화하여 제공하는 데이터를 얻는 방법입니다.

#### API 기반 데이터 수집의 장점과 한계

- 장점:
데이터 품질: 제공자가 검증한 구조화된 데이터로 품질이 높습니다.
안정성: 정해진 스키마와 형식으로 일관된 데이터가 제공되어 파싱 오류가 적습니다.
효율성: 필요한 데이터만 선택적으로 요청할 수 있어 효율적입니다.
법적 안정성: 명시적 허가 하에 데이터를 사용하므로 법적 리스크가 낮습니다.
- 한계:
접근 제한: 제공자가 허용하는 범위 내에서만 접근 가능합니다.
비용: 대규모 데이터 요청 시 높은 비용이 발생할 수 있습니다.
의존성: 제공자의 정책 변경(API 변경, 중단)에 따라 영향을 받습니다.

## 1.2.4 크라우드소싱을 통한 데이터 수집

크라우드소싱은 대규모의 분산된 작업자들을 활용하여 데이터를 수집하고 라벨링하는 방법입니다.
 Amazon Mechanical Turk, CrowdFlower, Labelbox 등의 플랫폼이 널리 사용됩니다.

#### 크라우드소싱의 장단점

- 장점: 대규모 데이터 수집 가능, 다양한 배경의 작업자 활용, 비용 효율성
- 단점: 작업자 품질 관리의 어려움, 일관성 없는 결과, 작업자 동기 부여 문제

#### 품질 관리 전략

- 작업자 선별: 사전 테스트를 통한 작업자 품질 평가, 신뢰도 기반 작업자 선별
- 중복 작업: 같은 작업을 여러 작업자에게 할당하여 일관성 검증
- 품질 검증: Golden Standard 데이터를 통한 작업자 성능 모니터링

## 1.2.5 시뮬레이션 및 합성 데이터 생성

실제 데이터 수집이 어려운 경우, 시뮬레이션을 통해 합성 데이터를 생성하는 방법이 있습니다. 이는 특히 자율주행, 로봇공학, 의료 영상 등에서 유용합니다.

#### 시뮬레이션 기반 데이터 생성

- 물리 기반 시뮬레이션: Unity, Unreal Engine, CARLA 등을 활용한 가상 환경에서의 데이터 생성
- 수학적 모델링: 확률 분포, 통계적 모델을 기반으로 한 데이터 생성
- 도메인 지식 활용: 전문가 지식을 바탕으로 한 현실적인 시나리오 생성

#### 합성 데이터의 장단점

- 장점: 비용 효율성, 완전한 제어 가능, 희귀한 시나리오 생성 가능
- 단점: 현실과의 차이(Reality Gap), 도메인 적응 문제, 검증의 어려움

## 1.2.6 IoT 센서 데이터 수집의 특수성

센서 데이터는 물리적 세계와 디지털 세계를 연결하는 중요한 매개체입니다.
 인간의 주관적 판단이 개입되지 않은 객관적 측정값이라는 장점이 있지만, 동시에 측정 환경과 센서 특성에 따른 고유한 노이즈와 편향을 가집니다.

#### 센서 데이터의 고유 특성

- 고빈도 생성: 초당 수십~수천 개의 데이터 포인트 생성, 실시간 처리 요구
- 측정 불확실성: 센서 오차, 환경 영향(온도, 습도, 진동), 보정 필요성
- 다차원성: 시간, 공간, 물리적 특성의 복합적 관계
- 시공간적 상관관계: 위치와 시간에 따른 데이터의 의존성

#### 센서 데이터 처리의 핵심 과제

- 신호 처리: 노이즈 필터링, 신호 증폭, 주파수 분석
- 센서 융합: 다중 센서 데이터의 통합 및 보완
- 실시간 처리: 지연 시간 최소화를 위한 스트리밍 처리
- 장애 대응: 센서 고장, 통신 장애, 배터리 부족 등

#### 실제 적용 예시

- 자율주행: LiDAR, 카메라, 레이더, GPS 센서 데이터 통합
- 스마트 시티: 교통, 환경, 에너지 센서 네트워크
- 의료 IoT: 웨어러블 디바이스의 생체 신호 모니터링

## 1.3.1 수동 라벨링

수집된 원시 데이터는 AI 모델이 학습할 수 있도록 의미 있는 라벨(Label)이 부여되어야 합니다.
 라벨링은 AI 학습의 핵심 단계이며, 라벨의 품질이 모델 성능에 직접적인 영향을 미칩니다.

수동 라벨링 (Human Annotation)은 인간의 인지 능력과 상식적 추론을 활용하여 복잡하고 미묘한 판단이 필요한 태스크에서 높은 품질의 라벨을 생성하는 방법입니다.
 특히 창의성, 감정, 윤리적 판단이 필요한 영역에서는 현재로서는 대체 불가능한 방법입니다.

#### 라벨링 작업의 유형

- 분류 라벨링: 이미지 분류, 텍스트 분류, 감정 분석 등
- 객체 검출: 바운딩 박스, 세그멘테이션 마스크 생성
- 시퀀스 라벨링: 개체명 인식(NER), 품사 태깅(POS tagging)
- 관계 라벨링: 엔티티 간 관계, 문장 간 관계 등

#### 품질 관리 체계

- 라벨러 간 일치도(IAA): Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha 등으로 측정
- 가이드라인 개발: 명확하고 일관된 라벨링 기준 수립
- 품질 검증: Golden Standard 데이터를 통한 정확도 측정
- 지속적 개선: 피드백 루프를 통한 라벨링 프로세스 개선

#### 실무 도구

- 라벨링 플랫폼: Labelbox, Supervisely, CVAT, LabelImg
- 품질 관리: Label Studio, Prodigy, Snorkel

## 1.3.2 자동 라벨링과 약한 지도학습

대규모 데이터셋을 수동으로 라벨링하는 것은 시간과 비용이 많이 듭니다.
 약한 지도학습(Weak Supervision)은 완벽하지 않은 라벨링 정보를 활용하여 모델을 학습시키는 패러다임입니다.

#### 약한 지도학습의 주요 기법

- 휴리스틱 규칙 기반 라벨링:
 도메인 전문가의 지식을 명시적인 규칙으로 변환하여 데이터를 라벨링합니다. (e.g., 특정 키워드가 포함되면 긍정으로 분류)
- 거리 함수 기반 라벨링:
 이미 라벨링된 데이터와의 유사도(임베딩 공간에서의 근접성)를 기반으로 새로운 데이터의 라벨을 추정합니다.
- 기존 모델 활용: 사전 훈련된 모델의 예측 결과를 새로운 데이터의 라벨로 활용합니다.
- 크라우드소싱 집계: 다중 작업자의 라벨링 결과를 통계적으로 집계하여 신뢰도 높은 라벨 생성

#### Snorkel 프레임워크

- 라벨링 함수(LF) 정의: 다양한 소스의 라벨링 함수를 프로그래밍 방식으로 정의
- 확률적 모델: LF들의 결과를 확률적으로 결합하여 최종 라벨 결정
- 신뢰도 학습: 각 LF의 정확도와 커버리지를 자동으로 학습
- 생산성 향상: 수동 라벨링 대비 10-100배 빠른 라벨링 속도

#### 실무 적용 예시

- 의료 영상: 기존 진단 보고서를 기반으로 한 자동 라벨링
- 금융 문서: 규제 문서의 키워드 기반 자동 분류
- 소셜 미디어: 감정 분석을 위한 자동 라벨링

## 1.3.3 반자동 라벨링과 Active Learning

Active Learning은 기계학습에서 가장 정보가 많은 샘플을 선택적으로 라벨링함으로써 효율적인 학습을 추구하는 방법론입니다.
 이는 모든 데이터가 동등한 학습 가치를 가지지 않는다는 관찰에서 출발합니다.

![Active Learning](images/Active_Learning.png)

#### Active Learning의 핵심 전략

- 불확실성 기반 샘플 선택:
 모델이 가장 확신하지 못하는(예측 확률이 낮은) 샘플을 우선적으로 라벨링 요청합니다. (e.g., 엔트로피 기반 불확실성 측정, Query by Committee)
- 다양성 기반 샘플 선택:
 전체 데이터 분포를 고려하여, 아직 모델이 잘 학습하지 못한 영역이나 데이터 공간의 다양한 부분을 대표하는 샘플을 선택합니다.
 (e.g., 클러스터링 후 각 클러스터에서 샘플 선택)
- 예상 모델 변화 기반 샘플 선택:
 새로운 라벨이 모델의 가중치나 성능에 가장 큰 영향을 미칠 것으로 예상되는 샘플을 선택합니다.

#### 불확실성 측정 방법

- 엔트로피 기반: 예측 확률 분포의 엔트로피가 높은 샘플 선택
- Margin 기반: 가장 확실한 클래스와 두 번째 확실한 클래스 간의 확률 차이가 작은 샘플 선택
- Least Confidence: 최대 확률이 낮은 샘플 선택
- Query by Committee: 여러 모델의 예측이 일치하지 않는 샘플 선택

#### 실무 도구 및 프레임워크

- Python 라이브러리: modAL, scikit-learn의 active learning 구현
- 전문 플랫폼: Label Studio, Prodigy, LightTag
- 클라우드 서비스: AWS SageMaker Ground Truth, Google Cloud AutoML

## 1.3.4 Self-Supervised Learning

Self-supervised learning(자기지도학습)은 unlabelled dataset으로부터 좋은 representation을 얻고자하는 학습방식으로 representation learning의 일종입니다.
 unsupervised learning이라고 볼수도 있지만 최근에는 self-supervised learning이라고 많이 부르고 있습니다.
 그 이유는 label(y) 없이 input(x) 내에서 target으로 쓰일만 한 것을 정해서, 즉 self로 task를 정해서 supervision방식으로 모델을 학습하기 때문입니다.

self-supervised learning의 task를 pretext task(= 일부러 어떤 구실을 만들어서 푸는 문제)라고 부르고
 pretext task를 학습한 모델은 downstream task에 transfer하여 사용할 수 있습니다.
 self-supervised learning의 목적은 downstream task를 잘푸는 것이기 때문에 기존의 unsupervised learning과 다르게 downsream task의 성능으로 모델을 평가합니다.

![Self-Supervised Learning](images/Self_Supervised_Learning.png)

- Self-prediction (자기 예측):
 개별 샘플 내에서 데이터의 일부를 이용해 나머지를 예측하는 task를 말합니다. 예를 들어 time-series의 경우 next time step을 예측하는 방식이 대표적입니다.
- Contrastive Learning (대조 학습):
 각 데이터 포인트를 고유한 클래스로 간주하고, 같은 인스턴스의 다른 변형(Augmented View)은 가깝게, 다른 인스턴스는 멀게 학습하여(즉, 관계를 학습하여) 의미 있는 표현을 획득합니다.

이러한 방법론들은 대규모 비라벨링 데이터로부터 강력한 특징 표현을 학습하여, 다운스트림 태스크의 성능을 크게 향상시킵니다.

## 1.4.1 개인정보 보호의 다층적 접근

개인정보 보호는 단순히 개인 식별 정보를 제거하는 것을 넘어서, 재식별 가능성과 추론 공격에 대한 방어를 포함하는 종합적 접근이 필요합니다.

- 익명화 (Anonymization):
 개인 식별이 영구적으로 불가능하도록 데이터를 처리합니다. (e.g., k-anonymity, l-diversity, t-closeness 등의 기법)
- 가명화 (Pseudonymization):
 추가 정보 없이는 개인 식별이 어렵도록 처리하지만, 필요시 재식별이 가능한 가역적 처리입니다.
- 차등 프라이버시 (Differential Privacy):
 데이터에 수학적으로 보장된 노이즈를 추가하여, 개별 데이터 포인트의 존재 여부가 분석 결과에 미치는 영향을 최소화합니다.

## 1.4.2 편향성 방지를 위한 체계적 접근

데이터에 내재된 편향은 AI 모델의 예측에 직접적인 영향을 미쳐 불공정한 결과를 초래할 수 있습니다. 따라서 편향성을 식별하고 완화하기 위한 체계적인 접근이 필수적입니다.

#### 데이터 편향의 유형

- 표본 편향 (Sampling Bias): 모집단을 제대로 대표하지 못하는 데이터 수집으로 발생 (e.g., 특정 인구 집단만 과도하게 포함)
- 측정 편향 (Measurement Bias): 데이터 측정 방식의 오류나 불일치로 발생 (e.g., 센서 오작동, 주관적 평가 기준)
- 역사적 편향 (Historical Bias): 과거 사회적, 문화적 편견이 데이터에 반영되어 발생 (e.g., 과거 채용 데이터에 성차별적 요소 포함)

#### 편향성 완화 전략

- 다양성 확보: 인구통계학적, 지리적, 문화적 다양성을 확보하여 데이터셋의 대표성을 높입니다.
- 공정성 지표 개발 및 모니터링: 그룹별 성능 차이(e.g., Equal Opportunity, Demographic Parity)를 측정하고 지속적으로 모니터링하여 불공정성을 식별합니다.
- 데이터 재균형: 오버샘플링, 언더샘플링, 합성 데이터 생성 등을 통해 소수 그룹의 데이터를 보강합니다.

## 1.4.3 데이터 거버넌스 및 품질 관리 체계

데이터 거버넌스는 조직 내 데이터의 수집, 저장, 사용, 공유에 대한 정책과 절차를 정의하는 체계입니다. AI 프로젝트의 성공을 위해서는 체계적인 데이터 품질 관리가 필수적입니다.

#### 데이터 거버넌스의 핵심 요소

- 데이터 카탈로그: 조직 내 모든 데이터셋의 메타데이터 관리, 데이터 계보(Lineage) 추적
- 데이터 품질 표준: 정확성, 완전성, 일관성, 적시성, 유효성 기준 정의
- 접근 제어: 역할 기반 접근 제어(RBAC), 데이터 분류 및 보안 정책
- 규정 준수: GDPR, CCPA, HIPAA 등 관련 법규 준수

#### 데이터 품질 관리 도구

- 데이터 프로파일링: Great Expectations, Pandas Profiling, Data Quality
- 데이터 검증: TensorFlow Data Validation, Deequ, Great Expectations
- 데이터 모니터링: Evidently AI, WhyLabs, Fiddler
- 데이터 카탈로그: Apache Atlas, DataHub, Amundsen

#### 데이터 라이프사이클 관리

- 수집 단계: 데이터 소스 검증, 스키마 정의, 품질 체크포인트
- 처리 단계: 변환 로직 검증, 데이터 파이프라인 모니터링
- 저장 단계: 백업 및 복구, 버전 관리, 아카이빙 정책
- 사용 단계: 접근 로그, 사용 패턴 분석, 성능 모니터링

## Part 2: 데이터 정제

### 결측치 처리 이해

결측치(Missing Values)는 데이터 분석 및 AI 모델 학습에 심각한 문제를 야기할 수 있습니다. 결측치의 발생 메커니즘을 이해하는 것은 적절한 처리 방법을 선택하는 데 핵심적입니다.

## 2.1.1 결측치 발생 메커니즘 분석

#### 결측치 유형별 발생 원인과 특성

- MCAR (Missing Completely At Random):
 결측이 관찰된 변수나 관찰되지 않은 변수와 전혀 관련이 없이 완전히 무작위로 발생하는 경우입니다. 이는 가장 이상적인 결측 상황이지만 실제로는 매우 드물게 발생합니다.
 (e.g., 설문조사에서 무작위로 선택된 질문 생략, 기계적 오류로 인한 데이터 손실)
- MAR (Missing At Random):
 결측이 관찰된 다른 변수들과는 관련이 있지만, 결측된 변수 자체의 값과는 관련이 없는 경우입니다. 이는 조건부 무작위성을 의미하며, 적절한 변수들을 조건으로 주면 무작위성을 확보할 수 있습니다.
 (e.g., 나이가 높을수록 소득 정보 제공을 꺼리는 경우)
- MNAR (Missing Not At Random):
 결측이 결측된 변수 자체의 값과 관련이 있는 경우입니다. 이는 가장 복잡하고 처리하기 어려운 결측 유형으로, 도메인 지식과 추가적인 가정이 필요합니다.
 (e.g., 소득이 높을수록 소득 정보를 숨기려는 경향)

#### 결측치 탐지 및 분석 방법

- 시각적 탐지: Missingno 라이브러리를 활용한 결측치 패턴 시각화
- 통계적 분석: Little's MCAR 테스트, 패턴 분석
- 도메인 지식 활용: 비즈니스 로직을 통한 결측 원인 추정

#### 실무 도구

- Python 라이브러리: missingno, pandas, numpy
- 전문 도구: R의 mice 패키지, SPSS의 Missing Values Analysis

## 2.1.2 결측치 처리 방법의 선택 기준

#### 삭제 방법의 장단점과 적용 조건

- 완전 케이스 분석 (Complete Case Analysis):
 결측치가 있는 모든 행을 삭제하는 가장 간단한 방법입니다. MCAR 가정이 성립하고 결측률이 5% 미만이며 샘플 크기가 충분히 클 때만 제한적으로 사용 가능합니다.
 정보 손실이 크고 편향 가능성이 높습니다.
- 리스트와이즈 삭제 (Listwise Deletion): 결측치가 있는 모든 관측치를 분석에서 제외. 구현이 간단하나 데이터 손실이 큼.
- 페어와이즈 삭제 (Pairwise Deletion):
 특정 분석에 필요한 변수에 결측치가 있는 경우에만 해당 관측치를 제외. 데이터 손실을 줄일 수 있으나, 분석마다 다른 샘플 크기로 인해 복잡성 증가.

#### 대체 방법의 이론적 근거와 실용적 고려사항

- 단순 대체법의 한계:
 평균/중앙값 대체는 분산을 과소추정하고 분포 형태를 왜곡하며, 변수 간 상관관계를 약화시킵니다. 회귀 대체는 예측 모델의 정확성에 의존하며 과적합 위험이 있습니다.
- 다중 대체법 (Multiple Imputation)의 우수성:
 결측으로 인한 불확실성을 명시적으로 모델링하는 가장 이론적으로 우수하고 권장되는 방법입니다. 여러 번의 대체를 통해 변동성을 반영하고,
 Rubin's Rules를 통해 통계적 추론을 수행합니다. (e.g., MICE, Amelia)

#### 고급 대체 방법

- KNN 대체: 유사한 특성을 가진 관측치들의 평균으로 대체
- 랜덤 포레스트 대체: 트리 기반 모델을 활용한 대체
- 딥러닝 기반 대체: Autoencoder, GAN을 활용한 대체
- 베이지안 대체: 불확실성을 확률적으로 모델링

#### 실무 구현 예시

```
# Python 코드 예시
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
import missingno as msno
import matplotlib.pyplot as plt

# 1. 결측치 탐지 및 시각화
def analyze_missing_data(df):
    """결측치 패턴 분석"""
    # 결측치 개수 확인
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # missingno를 이용한 시각화
    msno.matrix(df, figsize=(10, 6))
    plt.title('결측치 패턴 시각화')
    plt.show()
    
    # 결측치 히트맵
    msno.heatmap(df, figsize=(10, 6))
    plt.title('결측치 상관관계 히트맵')
    plt.show()
    
    return missing_counts, missing_percentages

# 2. 기본 결측치 처리 방법들
def basic_missing_handling(df):
    """기본적인 결측치 처리 방법들"""
    
    # dropna() - 결측치가 있는 행/열 삭제
    df_dropna = df.dropna()  # 모든 결측치가 있는 행 삭제
    df_dropna_subset = df.dropna(subset=['column1', 'column2'])  # 특정 컬럼만 고려
    df_dropna_how = df.dropna(how='all')  # 모든 값이 결측인 행만 삭제
    df_dropna_thresh = df.dropna(thresh=len(df.columns)-2)  # 최소 2개 컬럼에 값이 있는 행만 유지
    
    # fillna() - 결측치 대체
    df_fill_mean = df.fillna(df.mean())  # 평균값으로 대체
    df_fill_median = df.fillna(df.median())  # 중앙값으로 대체
    df_fill_mode = df.fillna(df.mode().iloc[0])  # 최빈값으로 대체
    df_fill_ffill = df.fillna(method='ffill')  # 이전 값으로 대체 (forward fill)
    df_fill_bfill = df.fillna(method='bfill')  # 다음 값으로 대체 (backward fill)
    df_fill_interpolate = df.interpolate()  # 선형 보간법
    
    return {
        'dropna': df_dropna,
        'dropna_subset': df_dropna_subset,
        'fill_mean': df_fill_mean,
        'fill_median': df_fill_median,
        'fill_mode': df_fill_mode,
        'fill_ffill': df_fill_ffill,
        'fill_bfill': df_fill_bfill,
        'interpolate': df_fill_interpolate
    }

# 3. 고급 결측치 처리 방법들
def advanced_missing_handling(df):
    """고급 결측치 처리 방법들"""
    
    # SimpleImputer를 이용한 체계적 대체
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_median = SimpleImputer(strategy='median')
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')
    imputer_constant = SimpleImputer(strategy='constant', fill_value=0)
    
    # KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    df_knn_imputed = pd.DataFrame(
        knn_imputer.fit_transform(df), 
        columns=df.columns,
        index=df.index
    )
    
    # Iterative Imputer (MICE 방식)
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    iterative_imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100),
        max_iter=10,
        random_state=42
    )
    df_iterative_imputed = pd.DataFrame(
        iterative_imputer.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    return {
        'knn_imputed': df_knn_imputed,
        'iterative_imputed': df_iterative_imputed
    }

# 4. 시계열 데이터의 결측치 처리
def time_series_missing_handling(df, time_column):
    """시계열 데이터 전용 결측치 처리"""
    
    # 시간 순서대로 정렬
    df_sorted = df.sort_values(time_column)
    
    # 시계열 특화 보간법
    df_linear_interp = df_sorted.interpolate(method='linear')
    df_time_interp = df_sorted.interpolate(method='time')
    df_polynomial_interp = df_sorted.interpolate(method='polynomial', order=2)
    df_spline_interp = df_sorted.interpolate(method='spline', order=3)
    
    # 이동평균을 이용한 대체
    df_rolling_mean = df_sorted.fillna(df_sorted.rolling(window=3, center=True).mean())
    
    return {
        'linear_interp': df_linear_interp,
        'time_interp': df_time_interp,
        'polynomial_interp': df_polynomial_interp,
        'spline_interp': df_spline_interp,
        'rolling_mean': df_rolling_mean
    }

# 5. 결측치 처리 성능 평가
def evaluate_imputation_performance(original_df, imputed_df, test_mask):
    """결측치 대체 성능 평가"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    # 테스트 마스크를 이용해 실제 값과 예측 값 비교
    original_values = original_df[test_mask]
    imputed_values = imputed_df[test_mask]
    
    mse = mean_squared_error(original_values, imputed_values)
    mae = mean_absolute_error(original_values, imputed_values)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': np.sqrt(mse)
    }

# 6. 실무 활용 예시
def practical_missing_data_workflow(df):
    """실무에서 사용할 수 있는 결측치 처리 워크플로우"""
    
    # 1단계: 결측치 분석
    missing_counts, missing_percentages = analyze_missing_data(df)
    
    # 2단계: 결측률에 따른 처리 전략 선택
    high_missing_cols = missing_percentages[missing_percentages > 50].index
    low_missing_cols = missing_percentages[missing_percentages <= 5].index
    medium_missing_cols = missing_percentages[
        (missing_percentages > 5) & (missing_percentages <= 50)
    ].index
    
    # 3단계: 컬럼별 적절한 처리 방법 적용
    df_processed = df.copy()
    
    # 높은 결측률 컬럼은 삭제
    df_processed = df_processed.drop(columns=high_missing_cols)
    
    # 낮은 결측률 컬럼은 단순 대체
    for col in low_missing_cols:
        if df_processed[col].dtype in ['int64', 'float64']:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0])
    
    # 중간 결측률 컬럼은 고급 대체
    if len(medium_missing_cols) > 0:
        from sklearn.impute import KNNImputer
        knn_imputer = KNNImputer(n_neighbors=5)
        df_processed[medium_missing_cols] = knn_imputer.fit_transform(
            df_processed[medium_missing_cols]
        )
    
    return df_processed

# 사용 예시
# df = pd.read_csv('your_data.csv')
# df_cleaned = practical_missing_data_workflow(df)
```

## 2.1.3 이상치 탐지 및 처리

이상치(Outlier)는 데이터 분포에서 비정상적으로 벗어난 관측치로, 모델 성능에 부정적 영향을 미칠 수 있습니다.
 이상치의 정확한 식별과 적절한 처리는 데이터 품질 향상의 핵심입니다.

#### 이상치 탐지 방법

- 통계적 방법: Z-score, IQR(Interquartile Range), Modified Z-score
- 거리 기반 방법: Local Outlier Factor(LOF), Isolation Forest
- 밀도 기반 방법: DBSCAN, OPTICS
- 딥러닝 기반 방법: Autoencoder, One-Class SVM

#### 이상치 처리 전략

- 제거: 명확한 오류인 경우 제거
- 대체: 이상치를 적절한 값으로 대체
- 캡핑: 이상치를 임계값으로 제한
- 분리 모델링: 이상치를 별도로 모델링

#### 실무 구현 예시

```
# Python 코드 예시
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

# Z-score 기반 이상치 탐지
# data는 pandas DataFrame 또는 numpy array 형태의 데이터
z_scores = np.abs(stats.zscore(data))
outliers = (z_scores > 3).any(axis=1)

# IQR 기반 이상치 탐지
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(data) == -1
```

## 2.1.4 결측치 처리 라이브러리 활용 가이드

#### 주요 Python 라이브러리별 특징과 활용법

- Pandas: 기본적인 결측치 처리 기능 제공 (dropna, fillna, interpolate)
- NumPy: 수치 계산과 결측치 마스킹 기능
- Scikit-learn: 체계적인 결측치 대체 알고리즘 (SimpleImputer, KNNImputer, IterativeImputer)
- Missingno: 결측치 패턴 시각화 및 분석
- SciPy: 통계적 보간법과 고급 수치 계산

#### 라이브러리별 장단점 비교

라이브러리 | 장점 | 단점 | 적합한 상황
---|---|---|---
Pandas | 사용법 간단, 빠른 처리 | 고급 알고리즘 부족 | 간단한 결측치 처리
Scikit-learn | 다양한 알고리즘, 일관된 API | 메모리 사용량 높음 | 복잡한 결측치 대체
Missingno | 직관적인 시각화 | 처리 기능 없음 | 결측치 패턴 분석

## 2.1.5 결측치 처리 성능 최적화와 베스트 프랙티스

#### 성능 최적화 기법

- 메모리 효율성: 대용량 데이터에서 chunk 단위 처리, 적절한 데이터 타입 사용
- 계산 효율성: 벡터화 연산 활용, 병렬 처리 적용
- 정확성 향상: 교차 검증을 통한 대체 방법 선택, 도메인 지식 활용

#### 실무 베스트 프랙티스

- 결측치 원인 분석: 데이터 수집 과정에서의 결측 원인 파악
- 적절한 방법 선택: 데이터 특성과 결측 패턴에 따른 방법 선택
- 결과 검증: 대체 결과의 통계적 검증 및 도메인 전문가 검토
- 문서화: 결측치 처리 과정과 방법의 상세한 기록

#### 자주 발생하는 실수와 주의사항

- 무분별한 삭제: 정보 손실을 고려하지 않은 과도한 행/열 삭제
- 부적절한 대체: 데이터 특성을 고려하지 않은 단순 평균 대체
- 검증 부족: 대체 결과의 품질 검증 과정 생략
- 일관성 부족: 훈련/테스트 데이터에 다른 처리 방법 적용

## 2.2.1 시계열 데이터 처리의 고유 특성

시간적 순서가 내재된 정보를 포함하며, 이러한 시간적 의존성은 전통적인 독립 동일 분포(i.i.d) 가정을 위반합니다.
 따라서 시계열 고유의 특성을 고려한 전처리가 필수적입니다.

#### 시계열 데이터의 특성

- 자기상관성 (Autocorrelation): 과거 값들과 현재 값 간의 선형 관계
- 계절성 (Seasonality): 규칙적으로 반복되는 패턴
- 추세 (Trend): 장기적인 증가 또는 감소 패턴
- 정상성 (Stationarity) 확보의 중요성: 평균, 분산, 공분산이 시간에 무관하게 일정한 상태

#### 시계열 전처리 기법

- 차분(Differencing): 추세 제거를 위한 차분 적용
- 로그 변환: 분산 안정화 및 정규성 확보
- 이동평균: 노이즈 제거 및 추세 추출
- 계절성 제거: 계절적 차분, 계절적 조정

#### 시계열 데이터의 결측치 처리

- 시간적 특성을 고려한 대체: 이전/이후 값 활용, 계절성 패턴 반영
- 보간법 활용: 선형 보간, 다항식 보간, 스플라인 보간
- 이동평균 활용: 노이즈 제거와 결측치 대체 동시 수행
- 예측 모델 활용: ARIMA, Prophet 등을 이용한 결측치 예측

#### 실무 구현 예시

```
# Python 코드 예시
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# 1. 시계열 데이터 결측치 처리 함수들
def handle_timeseries_missing_data(df, time_column, value_column):
    """시계열 데이터 전용 결측치 처리"""
    
    # 시간 순서대로 정렬
    df_sorted = df.sort_values(time_column).copy()
    df_sorted.set_index(time_column, inplace=True)
    
    # 다양한 보간법 적용
    df_linear = df_sorted[value_column].interpolate(method='linear')
    df_time = df_sorted[value_column].interpolate(method='time')
    df_polynomial = df_sorted[value_column].interpolate(method='polynomial', order=2)
    df_spline = df_sorted[value_column].interpolate(method='spline', order=3)
    
    # 이동평균을 이용한 대체
    df_rolling_mean = df_sorted[value_column].fillna(
        df_sorted[value_column].rolling(window=7, center=True).mean()
    )
    
    # 계절성을 고려한 대체 (월별 평균)
    df_seasonal = df_sorted[value_column].fillna(
        df_sorted[value_column].groupby(df_sorted.index.month).transform('mean')
    )
    
    return {
        'linear': df_linear,
        'time': df_time,
        'polynomial': df_polynomial,
        'spline': df_spline,
        'rolling_mean': df_rolling_mean,
        'seasonal': df_seasonal
    }

# 2. ARIMA 모델을 이용한 결측치 예측
def impute_with_arima(df, time_column, value_column, missing_indices):
    """ARIMA 모델을 이용한 결측치 대체"""
    
    df_sorted = df.sort_values(time_column).copy()
    df_sorted.set_index(time_column, inplace=True)
    
    # 결측치가 없는 부분으로 ARIMA 모델 훈련
    train_data = df_sorted[value_column].dropna()
    
    # ARIMA 모델 적합 (p=1, d=1, q=1)
    model = ARIMA(train_data, order=(1, 1, 1))
    fitted_model = model.fit()
    
    # 결측치 위치 예측
    predictions = fitted_model.forecast(steps=len(missing_indices))
    
    # 예측값으로 결측치 대체
    df_imputed = df_sorted.copy()
    for i, idx in enumerate(missing_indices):
        df_imputed.loc[idx, value_column] = predictions[i]
    
    return df_imputed

# 3. Prophet을 이용한 결측치 처리
def impute_with_prophet(df, time_column, value_column):
    """Prophet 모델을 이용한 결측치 대체"""
    
    # Prophet용 데이터 형식으로 변환
    prophet_df = df[[time_column, value_column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # 결측치가 없는 데이터로 모델 훈련
    train_df = prophet_df.dropna()
    
    # Prophet 모델 적합
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train_df)
    
    # 전체 기간에 대한 예측
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    
    # 예측값으로 결측치 대체
    df_imputed = df.copy()
    df_imputed[value_column] = df_imputed[value_column].fillna(
        forecast['yhat'].values
    )
    
    return df_imputed

# 4. 정상성 검정
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] < 0.05

# 5. 차분을 통한 정상성 확보
def make_stationary(timeseries):
    diff_series = timeseries.diff().dropna()
    return diff_series

# 6. 계절성 분해
def decompose_timeseries(timeseries, period=12):
    """시계열 분해"""
    decomposition = seasonal_decompose(timeseries, period=period)
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }

# 7. 시계열 결측치 처리 성능 평가
def evaluate_timeseries_imputation(original_series, imputed_series, missing_mask):
    """시계열 결측치 대체 성능 평가"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    original_values = original_series[missing_mask]
    imputed_values = imputed_series[missing_mask]
    
    mse = mean_squared_error(original_values, imputed_values)
    mae = mean_absolute_error(original_values, imputed_values)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': np.sqrt(mse),
        'MAPE': np.mean(np.abs((original_values - imputed_values) / original_values)) * 100
    }

# 사용 예시
# df = pd.read_csv('timeseries_data.csv')
# df['date'] = pd.to_datetime(df['date'])
# 
# # 결측치 처리
# imputed_data = handle_timeseries_missing_data(df, 'date', 'value')
# 
# # ARIMA 모델을 이용한 대체
# missing_indices = df[df['value'].isnull()].index
# df_arima_imputed = impute_with_arima(df, 'date', 'value', missing_indices)
# 
# # Prophet 모델을 이용한 대체
# df_prophet_imputed = impute_with_prophet(df, 'date', 'value')
```

## 2.2.2 이미지 데이터 처리의 심화 이해

이미지 데이터는 픽셀 단위의 고차원 정보를 포함하며, 각 픽셀 간의 공간적 관계가 의미를 결정하는 중요한 요소입니다.
 이러한 특성은 이미지 전처리에서 특별한 고려사항을 요구합니다.

#### 이미지 전처리의 핵심 요소

- 공간적 일관성 유지: 픽셀 간의 상대적 위치 관계, 객체의 형태와 구조 정보, 텍스처와 패턴의 연속성을 보존
- 크기 조정 (Resizing): 모델 입력 요구사항에 맞춘 이미지 크기 조정
- 채널 처리: 색상 공간 변환 및 채널별 정규화
- 노이즈 제거: 이미지 품질 향상을 위한 필터링

#### 크기 조정 전략

- 보간 방법: 최근접 이웃(Nearest Neighbor), 이중선형(Bilinear), 이중삼차(Bicubic), Lanczos
- 종횡비 처리: 패딩(Padding), 크롭(Cropping), 비균등 스케일링
- 해상도 선택: 계산 효율성과 정보 보존의 트레이드오프

#### 색상 공간 및 정규화

- 색상 공간 변환: RGB, HSV, LAB, Grayscale, YUV
- 채널별 정규화: 각 채널의 분포 차이 보정
- ImageNet 정규화: 전이 학습 시 도메인 간 분포 차이 완화

#### 실무 구현 예시

```
# Python 코드 예시
import cv2
import numpy as np
from PIL import Image
import albumentations as A

# OpenCV를 이용한 이미지 전처리
def preprocess_image_cv2(image_path, target_size=(224, 224)):
    # image_path는 이미지 파일 경로
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

# Albumentations를 이용한 고급 전처리
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
])
```

## 2.2.3 텍스트 데이터 처리의 언어학적 접근

텍스트 데이터는 인간의 언어 사용 패턴을 반영하며, 문법적 구조와 의미적 관계가 복합적으로 얽혀있습니다.
 따라서 단순한 통계적 처리를 넘어서 언어학적 원리를 고려한 전처리가 필요합니다.

#### 토큰화 (Tokenization)의 언어학적 의미

- 형태소 단위 토큰화: 언어의 최소 의미 단위(어근, 접사) 분리. 한국어처럼 교착어에 중요
- 서브워드 토큰화 (Subword Tokenization):
 미등록어(OOV, Out-Of-Vocabulary) 문제를 해결하고 형태학적 유사성을 포착하기 위해 단어를 더 작은 단위(서브워드)로 분리
 (e.g., BPE, WordPiece, SentencePiece)
- 문자 단위 토큰화: 문자 수준에서의 토큰화로 언어 독립적 처리 가능

#### 텍스트 정제 및 정규화

- 정규화: 대소문자 통일, 숫자 정규화, 특수문자 처리
- 불용어 제거: 의미가 없는 단어들의 제거
- 어간 추출 및 표제어 추출: 단어의 기본 형태로 변환
- 오타 교정: 자동 오타 검출 및 수정

#### 특수 토큰 및 시퀀스 관리

- 특수 토큰: [CLS], [SEP], [PAD], [MASK] 등 Transformer 모델용 토큰
- 시퀀스 길이 관리: 패딩, 트렁케이션, 슬라이딩 윈도우
- 어휘 사전 관리: 빈도 기반 어휘 사전 구축, 미등록어 처리

#### 실무 구현 예시

```
# Python 코드 예시
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import transformers

# 기본 텍스트 정제
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# 토큰화
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# 불용어 제거
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return filtered_tokens

# BERT 토크나이저 사용
# text는 전처리할 텍스트 문자열
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, padding='max_length', truncation=True, max_length=512)
```

## 2.3.1 DAE (Denoising Autoencoder)

DAE는 단순히 노이즈를 제거하는 도구를 넘어, 데이터의 본질적 구조를 학습하는 표현 학습(Representation Learning) 방법입니다.
 노이즈가 추가된 입력에서 원본을 복원하는 과정에서 모델은 데이터의 중요한 특징과 불필요한 변동을 구분하는 능력을 획득합니다.

![DAE](images/DAE.jpeg)

- 매니폴드 학습 (Manifold Learning) 관점:
 고차원 데이터가 실제로는 저차원 매니폴드 상에 존재한다는 가정 하에, 노이즈 제거 과정을 통해 이 매니폴드 구조를 학습합니다.
- 정화 (Regularization) 효과:
 노이즈 주입은 모델이 과적합되는 것을 방지하고 일반화 성능을 향상시키는 일종의 정규화 역할을 합니다.

## 2.3.2 DAE : 노이즈 타입별 특성과 효과

![DAE Noise](images/DAE_Noise.png)

- 가우시안 노이즈: 연속적이고 부드러운 특성을 가지며, 이미지나 음성 신호에 흔히 사용됩니다.
- 드롭아웃 노이즈: 신경망의 과적합 방지를 위해 개발되었지만, DAE에서는 구조적 노이즈로 활용되어 모델의 강건성을 높입니다.
- 마스킹 노이즈:
 입력의 일부를 완전히 제거하여 정보 손실과 복원에 대한 학습을 유도합니다. (e.g., BERT의 Masked Language Model)

응용: 이미지 복원, 음성 신호 개선, 텍스트 데이터의 오타 교정 등 다양한 분야에 활용됩니다.

## 2.3.3 DAE : 응용 분야별 특화 전략

#### 의료 영상에서의 Denoising Autoencoder

- 의료 영상 노이즈의 특성: 양자 노이즈, 스캐터링, 아티팩트 등.
- 의료 특화 손실 함수: 진단적으로 중요한 영역에 가중치 부여, 조직 경계 및 미세 구조 보존 강조.

#### 자연어 처리에서의 텍스트 정제

- 텍스트 노이즈의 유형: 오타, 문법 오류, 비표준 표현.
- 맥락 보존 전략: 문장 수준 의미 일관성 유지, 장거리 의존성 보존.

## Part 3: 데이터 증강

### 데이터 부족 문제

데이터 부족은 단순히 데이터의 절대적 수량이 적은 양적 부족뿐만 아니라,
 실제 환경의 다양성을 반영하지 못하는 질적 부족이 더 큰 문제입니다.
 이 두 측면은 서로 다른 접근법을 요구하며, 종종 동시에 해결해야 하는 복합적 문제입니다.

#### 데이터 부족의 유형

- 양적 부족: 통계적 유의성 확보의 어려움, 모델의 일반화 성능 저하, 과적합 위험 증가
- 질적 부족: 실제 환경의 다양성 미반영, 엣지 케이스 누락, 편향된 학습으로 인한 강건성 부족
- 분포 불균형: 특정 클래스나 조건의 데이터 과소/과다 표현

#### 데이터 부족의 영향

- 모델 성능: 정확도, 재현율, F1-score 등의 성능 지표 저하
- 일반화 능력: 새로운 데이터에 대한 예측 성능 부족
- 편향 문제: 특정 패턴에 과도하게 의존하는 모델
- 신뢰성: 예측 결과의 불확실성 증가

#### 해결 방향

- 데이터 증강: 기존 데이터를 변형하여 새로운 샘플 생성
- 전이학습: 사전 훈련된 모델을 활용한 지식 전이
- 적은 샘플 학습: Few-shot, Zero-shot 학습 기법
- 합성 데이터: 시뮬레이션을 통한 데이터 생성

## 3.1.1 Long-tail 분포의 이해와 대응

현실 세계의 많은 데이터는 소수의 일반적 사례가 대부분을 차지하고, 다수의 희귀 사례가 꼬리 부분(Long-tail)을 형성하는 분포를 따릅니다.
 이는 희귀 사례에 대한 학습을 어렵게 만듭니다.

#### Long-tail 분포의 특징

- 파레토 법칙(80-20 규칙): 소수의 인기 항목이 대부분의 관심을 차지
- 헤비 테일: 극값 이벤트의 빈도가 정규분포보다 높음
- 지수적 감소: 빈도가 지수적으로 감소하는 패턴
- 집중도: 상위 20%가 전체의 80%를 차지

#### Long-tail 분포의 영향

- 학습 편향: 다수 클래스에 편향된 학습
- 성능 저하: 소수 클래스에 대한 낮은 성능
- 일반화 실패: 희귀 케이스에 대한 예측 실패
- 평가 왜곡: 전체 정확도는 높지만 실제 성능은 낮음

#### 대응 전략

- 계층적 샘플링: 클래스별 균형을 맞춘 샘플링
- 가중치 기반 학습: 소수 클래스에 높은 가중치 부여
- 멀티스케일 접근: 다양한 스케일에서의 특징 학습
- 앙상블 방법: 여러 모델의 조합으로 성능 향상

## 3.1.2 도메인별 데이터 부족의 특수성

#### 의료 데이터 부족의 구조적 원인

- 환자 프라이버시와 규제: HIPAA 등 엄격한 개인 건강 정보 보호 규제로 데이터 공유 및 활용에 제약.
- 희귀 질환의 본질적 희소성: 환자 수가 적어 데이터 확보가 어려움.
- 임상 시험의 제약사항: 엄격한 선정 기준, 윤리적 고려사항으로 인한 데이터 제한.

#### 금융 데이터의 특수한 제약

- 시장 효율성 가설: 유용한 정보가 빠르게 가격에 반영되어 데이터의 시간적 가치 감소.
- 규제 환경과 컴플라이언스: 금융 감독 기관의 엄격한 규제, 고객 정보 보호 의무.
- 시스템적 리스크와 극값 이벤트: 금융 위기 같은 희귀한 극값 이벤트의 예측 어려움.

## 3.2.1 클래스 불균형 문제의 심화 분석

클래스 불균형은 특정 클래스(소수 클래스)의 데이터 수가 다른 클래스(다수 클래스)에 비해 현저히 적은 경우를 말합니다.
 대부분의 기계학습 알고리즘은 전체 정확도를 최대화하도록 설계되어 있어, 불균형 데이터에서는 다수 클래스에 편향된 예측을 하게 됩니다.

#### 클래스 불균형의 영향

- 학습 편향: 다수 클래스에 편향된 모델 학습
- 성능 왜곡: 정확도는 높지만 실제 성능은 낮음
- 평가 지표 문제: 단순 정확도로는 실제 성능 파악 어려움
- 실무 적용 실패: 실제 환경에서의 성능 저하

#### 불균형 해결 방법의 이론적 기반

- 샘플링 기법:
언더샘플링: 다수 클래스의 데이터 수를 줄입니다. 정보 손실 위험이 있습니다.
오버샘플링: 소수 클래스의 데이터 수를 늘립니다. 단순 복제는 과적합을 유발할 수 있습니다.
SMOTE (Synthetic Minority Oversampling Technique):
 소수 클래스 데이터들을 연결하는 특징 공간 상의 직선상에 새로운 합성 데이터를 생성하는 가장 대표적인 오버샘플링 기법입니다.
 이는 과적합 위험을 줄이면서 소수 클래스의 다양성을 확보합니다.
- 코스트 센서티브 학습 (Cost-Sensitive Learning):
 모델의 손실 함수에서 소수 클래스의 오분류에 더 큰 페널티(비용)를 부과하여, 모델이 소수 클래스에 더 집중하도록 학습을 유도합니다.

#### 실무 구현 예시

```
# Python 코드 예시
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

# SMOTE 적용
# X는 특성 데이터, y는 라벨 데이터
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# 코스트 센서티브 학습
model = RandomForestClassifier(class_weight=class_weight_dict)
```

## 3.3.1 기본 데이터 증강 기법

데이터 증강은 기존 데이터를 변형하여 새로운 샘플을 생성하는 기법입니다.
 이를 통해 데이터의 다양성을 증가시키고 모델의 일반화 성능을 향상시킬 수 있습니다.

#### 이미지 데이터 증강

- 기하학적 변환: 회전, 이동, 스케일링, 뒤집기
- 색상 변환: 밝기, 대비, 채도, 색조 조정
- 노이즈 추가: 가우시안 노이즈, 소금-후추 노이즈
- 블러 및 샤프닝: 가우시안 블러, 언샤프 마스킹

#### 텍스트 데이터 증강

- 동의어 치환: WordNet, BERT 기반 동의어 찾기
- 문장 구조 변환: 능동태-수동태 변환, 문장 순서 변경
- 백번역: 번역 후 다시 원문으로 번역
- EDA (Easy Data Augmentation): 삭제, 삽입, 교체, 교환

#### 실무 구현 예시

```
# Python 코드 예시 (이미지 증강)
import albumentations as A
from PIL import Image
import numpy as np

# Albumentations를 이용한 이미지 증강
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])
```

## 3.4.1 Mixup 기법

Mixup은 두 데이터 포인트와 그 라벨을 동일한 가중치로 선형 결합하여 새로운 학습 데이터를 생성하는 기법입니다.
 이는 특성 공간과 라벨 공간에서의 볼록 결합(Convex Combination)을 의미합니다.

#### Mixup의 수학적 기반

```
# Mixup의 수학적 기반
new_x = λ * x_i + (1 - λ) * x_j
new_y = λ * y_i + (1 - λ) * y_j

# λ는 베타 분포 Beta(α, α)에서 샘플링되며, α 값에 따라 혼합 강도가 조절됩니다.
```

#### Mixup의 핵심 특징

- 결정 경계의 부드러움:
 Mixup은 데이터 분포의 볼록 껍질(Convex Hull) 내에서 새로운 샘플을 생성하여, 모델이 데이터 간의 선형적인 관계를 학습하도록 유도합니다.
 이는 결정 경계를 부드럽게 만들고, 과적합을 방지하며, 모델의 일반화 성능과 강건성 향상에 크게 기여합니다.
- 정칙화 효과:
 노이즈 주입과 유사하게, 모델이 훈련 데이터에 과도하게 의존하는 것을 막아 일반화 성능을 높이는 정칙화 효과를 가집니다.
- 라벨 스무딩: 하드 라벨 대신 소프트 라벨을 사용하여 모델의 확신도를 조절
- 도메인 적응: 도메인 간 차이를 완화하는 효과

#### Mixup의 장단점

- 장점: 과적합 방지, 일반화 성능 향상, 강건성 증가
- 단점: 의미적으로 부자연스러운 샘플 생성, 계산 오버헤드

#### 실무 구현 예시

```
# Python 코드 예시
import numpy as np
import torch
from scipy.stats import beta

def mixup_data(x, y, alpha=0.2):
    # x는 입력 데이터 (torch.Tensor), y는 라벨 데이터
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

## 3.4.2 도메인별 Mixup 적용의 특수성

#### Word-level Mixup의 언어학적 고려사항

- 의미적 일관성 유지: 단어 임베딩 공간에서의 의미적 거리 고려, 문법적 범주 일치성 확보.
- 구문적 구조 보존: 문장의 구문 분석 결과 활용, 종속 관계와 수식 관계 고려.

#### Sentence-level Mixup의 담화 분석

- 담화 응집성 (Discourse Coherence): 문장 간 논리적 연결성 유지, 주제 일관성과 정보 흐름 보존.
- 화용론적 측면: 발화 의도와 맥락 정보 보존, 언어행위론적 특성 고려.

## 3.5.1 생성 모델 기반 증강

단순 변환을 넘어, 실제 데이터와 구별하기 어려운 새로운 데이터를 '창조'합니다.
 특히 조건부 생성 모델(Conditional GAN)을 활용하면,
 부족한 소수 클래스의 데이터만 선택적으로 생성하여 데이터 불균형 문제를 효과적으로 해결할 수 있습니다.

- GAN (Generative Adversarial Networks): 생성자와 판별자의 경쟁을 통해 실제와 유사한 데이터를 생성.
- Diffusion Models: 노이즈를 점진적으로 제거하여 데이터를 생성하는 방식. 고품질 이미지 생성에 뛰어난 성능.

## 3.5.2 데이터 증강의 윤리적 이슈와 한계

데이터 증강은 AI 모델의 성능을 향상시키는 강력한 도구이지만, 동시에 여러 윤리적 문제와 기술적 한계를 가지고 있습니다.
 이러한 이슈들을 인식하고 적절히 대응하는 것이 중요합니다.

#### 윤리적 이슈

- 편향 증폭: 기존 데이터의 편향이 증강 과정에서 증폭될 수 있음
- 개인정보 보호: 개인 식별 가능한 정보가 증강 과정에서 노출될 위험
- 저작권 문제: 원본 데이터의 저작권이 증강된 데이터에 미치는 영향
- 투명성 부족: 증강 과정의 추적 가능성과 설명 가능성 부족

#### 기술적 한계

- 도메인 간 차이: 증강된 데이터와 실제 환경 간의 차이
- 의미적 일관성: 증강 과정에서 원본 데이터의 의미가 왜곡될 가능성
- 과적합 위험: 과도한 증강으로 인한 모델의 과적합
- 계산 비용: 대규모 증강 시 발생하는 계산 오버헤드

#### 대응 방안

- 편향 모니터링: 증강 전후의 데이터 분포 변화 추적
- 품질 검증: 증강된 데이터의 품질을 지속적으로 검증
- 윤리적 가이드라인: 증강 과정에서의 윤리적 원칙 수립
- 투명성 확보: 증강 방법과 과정의 문서화 및 공개

#### 미래 발전 방향

- 도메인 적응: 실제 환경에 더 가까운 증강 기법 개발
- 자동화: 증강 정책의 자동 최적화
- 윤리적 AI: 윤리적 고려사항을 내장한 증강 기법
- 효율성 향상: 계산 효율성을 고려한 증강 방법
