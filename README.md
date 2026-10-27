# 지하철 승·하차 수요 예측 대시보드

선택한 날짜·시각을 입력하면 지하철 역 1104의 향후 5·15·30·60분 승차량과 하차량을 예측하는 프로젝트다. 전처리·모델 학습 노트북과 저장된 LightGBM 모델을 직접 호출하는 Flask 대시보드로 구성된다.

## 현재 추론 앱의 동작

`날짜·시각 선택 → 달력 특성 계산 → 8개 모델 추론 → 음수·누적값 보정 → 표 표시`

- 시간, 요일, 월, 공휴일, 명절, 연휴 인접일, 추세 등 달력 특성 생성
- 승차·하차와 4개 예측 구간을 조합한 8개 모델 사용
- 운행창 밖의 시각 입력 거부
- 예측 누적값이 5분 ≤ 15분 ≤ 30분 ≤ 60분이 되도록 보정
- 모델 입력 특성을 웹 화면에서 확인

현재 `inference/app.py`는 실시간 이용량 lag 값을 입력받지 않는다. 과거 `모델설계문서.md`의 lag 특성 기반 설명과 구분하고, 배포된 `models_v3.pkl` 및 추론 코드의 입력 구성을 기준으로 사용한다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `metro_usage_202607071800.csv` | 원본 이용 자료 |
| `eda/데이터탐색.ipynb` | 분포·시간 패턴 탐색 |
| `preprocessing/전처리.ipynb` | 시계열 격자·타깃·특성 구성 |
| `train.csv`, `val.csv`, `test.csv` | 시간순 분할 자료 |
| `training/모델학습_lightbgm.ipynb` | 모델 학습 |
| `training/성능측정_lightbgm.ipynb` | 평가 |
| `models/models_v3.pkl` | 현재 앱에서 사용하는 모델 묶음 |
| `inference/app.py` | Flask 화면과 추론 API |

## 실행 방법

저장소 루트의 Windows PowerShell에서 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install flask pandas numpy lightgbm scikit-learn joblib holidays
.\.venv\Scripts\python.exe inference/app.py
```

`http://localhost:5000`에 접속해 날짜·시각을 선택하고 입력 특성과 예측 결과를 확인한다. 앱 시작 시 저장된 모델을 읽으므로 모델 파일과 학습 당시 패키지의 호환성이 필요하다.

API는 `/api/meta`, `/api/features?ts=...`, `/api/predict?ts=...`로 구분되어 있다. 운행창 검사는 05:00 이후 또는 00:00~00:10을 허용한다.

## 재학습과 결과 해석

재학습은 EDA → 전처리 → 모델 학습 → 평가 노트북 순서로 진행한다. 일부 노트북의 절대 경로인 `OUT_DIR` 등을 현재 저장소 위치로 맞춰야 한다. 재학습 후에는 모델의 특성 목록과 `make_features()`의 출력이 일치하는지 확인한다.

대상은 단일 역이며 예측은 달력 패턴에 따른 수요 추정이다. 행사·운행 장애·실시간 승객 변화까지 직접 반영하지 않는다. 과거 설계 문서의 성능표를 현재 v3 모델의 성능으로 옮겨 쓰지 않으며, 해당 모델에 대응하는 평가 결과를 따로 확인한다.

---

[GitHub 저장소](https://github.com/pranpreten/subway_inference) · [문서 작성 기준 코드](https://github.com/pranpreten/subway_inference/tree/97706362c23f3608dc67cf7420e78f3544d1b5a0)
