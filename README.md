# 프로젝터 인터랙티브 공 게임

카메라로 빨간색 물체를 인식하여 프로젝터 화면에 투영된 공을 튕기는 인터랙티브 게임입니다.
색상 감지 방식으로 **HSV 기반**과 **ML(딥러닝) 기반** 두 가지 버전을 제공합니다.

---

## 게임 종류

### 🏐 배구 게임 (`Game_HSV.py`) — HSV 기반
- 화면 중앙에 네트가 있는 **2인용 배구 게임**
- 빨간 물체로 공을 쳐서 상대 코트에 떨어뜨리면 득점
- OpenCV HSV 색상 범위로 빨간색 물체를 감지
- CUDA GPU 가속 지원

### 🏀 농구 게임 (`Game_ML.py`) — ML 기반
- 떨어지는 농구공을 빨간 물체로 쳐서 점수를 올리는 **1인용 게임**
- 학습된 PyTorch 모델로 색상을 감지

---

## 시스템 구성

```
카메라 → 색상 감지 → 프로젝터 좌표 변환 → 충돌 판정 → 프로젝터 출력
```

| 파일 | 설명 |
|------|------|
| `calibration.py` | 체스보드 기반 카메라-프로젝터 호모그래피 캘리브레이션 |
| `run_calibration.py` | 캘리브레이션 실행 스크립트 |
| `train_color.py` | 색상 분류 모델 학습 (7가지 색상) |
| `color_detect.py` | 학습된 모델로 실시간 색상 감지 |
| `Game_HSV.py` | HSV 기반 배구 게임 |
| `Game_ML.py` | ML 기반 농구 게임 |

---

## 설치 방법

```bash
pip install opencv-python numpy torch torchvision
```

GPU 가속을 사용하려면 CUDA 버전의 PyTorch 설치:
```bash
# PyTorch 공식 사이트(https://pytorch.org)에서 환경에 맞는 명령어 확인
```

---

## 실행 순서

### 1단계: 캘리브레이션
프로젝터와 카메라의 좌표를 맞추는 작업입니다.

1. `chessboard.jpg`를 프로젝터 화면에 출력
2. 카메라가 체스보드를 인식하면 자동으로 호모그래피 행렬 저장

```bash
python run_calibration.py
```

- 5프레임 연속 체스보드 인식 성공 시 `calibration_matrix.npy` 자동 저장
- `q` 키로 종료

### 2단계: 색상 모델 학습 (ML 버전 사용 시)

```bash
python train_color.py
```

- `dataset/ColorClassification/` 폴더의 이미지로 학습
- 7가지 색상 분류: Red, Blue, Green, Black, White, Brown, Violet
- 학습 완료 후 `color_model.pt` 저장

### 3단계: 게임 실행

**배구 게임 (HSV 기반)**
```bash
python Game_HSV.py
```

**농구 게임 (ML 기반)**
```bash
python Game_ML.py
```

- `q` 키로 종료

---

## 조작 방법

- **빨간색 물체** (장갑, 공 등)를 카메라 앞에서 움직여 공을 타격
- 공이 바닥에 떨어지면 자동으로 리스폰

---

## 프로젝트 구조

```
PROJECT5/
├── calibration.py          # 캘리브레이션 모듈
├── run_calibration.py      # 캘리브레이션 실행
├── train_color.py          # 색상 모델 학습
├── color_detect.py         # ML 색상 감지
├── Game_HSV.py             # 배구 게임 (HSV)
├── Game_ML.py              # 농구 게임 (ML)
├── chessboard.jpg          # 캘리브레이션용 체스보드 이미지
├── calibration_matrix.npy  # 캘리브레이션 결과 (자동 생성)
├── color_model.pt          # 학습된 색상 모델 (자동 생성)
└── game_image/
    ├── basketball.png
    ├── volleyball.png
    └── net.png
```

---

## 기술 스택

- **Python**
- **OpenCV** — 카메라 입력, 색상 감지 (HSV), 호모그래피
- **PyTorch** — 색상 분류 신경망 학습 및 추론
- **NumPy** — 행렬 연산
