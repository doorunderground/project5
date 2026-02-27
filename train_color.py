#%%
  
# 색상 분류 모델을 학습
# 7가지 색상(빨강, 파랑, 초록, 검정, 흰색, 갈색, 보라)
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = r'C:\\PROJECT5\dataset\\ColorClassification'
IMG_SIZE = 64

COLORS = ['red', 'Blue', 'Green', 'Black', 'White', 'Brown', 'Violet']
COLOR_LABELS = {}
for i in range(len(COLORS)):
    COLOR_LABELS[COLORS[i]] = i
    

#%%
# ── 2. 이미지 로드 ────────────────────────────────
def load_images():
    X, y = [], []
    for color in COLORS:
        folder = os.path.join(DATA_DIR, color)
        if not os.path.exists(folder):
            print(f"[경고] 폴더 없음: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{color}: {len(files)}개")
        # jpg/png 파일만 골라서 읽기
        
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            # 이미지를 숫자 배열로 읽음
            
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img.flatten() / 255.0)
            # 픽셀 값은 0~255임
            # 신경망은 0~1 사이의 값을 훨씬 잘 학습함
            # 정규화
            # 이미지 1장을 숫자 벡터로 만들어서 데이터셋에 추가
            
            y.append(COLOR_LABELS[color])
            #정답 레이블
            # red->0 , Blue->1, Green->2
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = load_images()
print(f"\n원본 데이터 shape : {X.shape}")   # (96, 12288)
print(f"레이블 shape      : {y.shape}")    # (96,)


#%%
# 데이터가 적어서 노이즈를 추가해서 샘플 수를 인위적으로 늘림

def augment(X, y, times=10):
    X_aug, y_aug = [X], [y] #리스트 안에 원본을 담고
    for _ in range(times):
        noise = np.random.normal(0, 0.03, X.shape).astype(np.float32)
        X_aug.append(np.clip(X + noise, 0, 1))
        y_aug.append(y)
    X_aug = np.concatenate(X_aug)
    y_aug = np.concatenate(y_aug)
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]

X, y = augment(X, y, times=10)
print(f"증강 후 데이터 shape: {X.shape}")  # (1056, 12288)

# 원본 96
# 증강 후(times=10) 96x11 = 1056개


#%%
# ── 4. 학습 / 테스트 분리 & DataLoader ───────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dl = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=16, shuffle=True
)
test_dl = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=16
)

print(f"학습 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")


#%%
# ── 5. 모델 정의 (Sleep Detector와 동일 구조) ─────────
class ColorNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 256), #12288개 픽셀을 256개 특징으로 압축
            nn.ReLU(),  #음수 -> 0으로 처리
            nn.Linear(256, 256), #256 -> 256 (패턴 더 학습)
            nn.ReLU(),
            nn.Linear(256, num_classes), # 256 -> 7가지 색상
        )
    def forward(self, x):
        return self.net(x)

model     = ColorNet(num_classes=len(COLORS))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()   # 다중 분류 → CrossEntropy (Sleep Detector는 BCE)

print(model)

#입력: 빨간 이미지
#출력: [4.2, -1.3, 0.5, -2.1, 0.1, -0.8, -1.5]
#      red  Blue  Green Black White Brown Violet
#→ argmax → 0 → 'red' 로 분류


#%%
# ── 6. 학습 루프 ──────────────────────────────────
EPOCHS = 100
print("학습 시작...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        pred = model(xb) #예측
        loss = loss_fn(pred, yb) #얼마나 틀렸나
        optimizer.zero_grad() # 기울기 초기화
        loss.backward() #어떤 가중치가 문제인지 계산
        optimizer.step() #가중치 수정
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                correct += (model(xb).argmax(1) == yb).sum().item()
        acc = correct / len(y_test)
        print(f"Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_dl):.4f} | Test Acc: {acc:.4f}")



#%%
# ── 7. 모델 저장 ──────────────────────────────────
torch.save(model.state_dict(), r'C:\PROJECT5\color_model.pt')
print("저장 완료: color_model.pt")