#%%
# ── 1. 라이브러리 임포트 & 설정 ───────────────────
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 64
COLORS   = ['red', 'Blue', 'Green', 'Black', 'White', 'Brown', 'Violet']
MIN_AREA = 500


#%%
# ── 2. 모델 정의 (train_color.py와 동일해야 함) ────
class ColorNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)


#%%
# ── 3. 모델 로드 ──────────────────────────────────
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = ColorNet(num_classes=len(COLORS))
_model.load_state_dict(torch.load(r'C:\PROJECT5\color_model.pt', map_location=_device))
_model.to(_device)
_model.eval()
print(f"색상 감지 모델 로드 완료 | 클래스: {COLORS} | 디바이스: {_device}")


#%%
# ── 4. 색상 감지 함수 ─────────────────────────────
def detect_color(frame, target='red', tile=32, stride=16):
    """
    학습된 모델로 특정 색상 물체 감지
    - target : 감지할 색상 ('red', 'Blue', 'Green' 등)
    - tile   : 분석할 패치 크기 (px)
    - stride : 슬라이딩 간격 (작을수록 정밀, 느림)
    반환: [(cx, cy, contour), ...] — game.py detect_red()와 동일한 형식
    """
    h, w = frame.shape[:2]
    target_idx = COLORS.index(target)
    mask = np.zeros((h, w), dtype=np.uint8)

    # ── BGR→RGB 변환을 루프 밖에서 한 번만 ──
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── numpy→tensor (1, 3, H, W) ──
    frame_t = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # ── Unfold: 슬라이딩 윈도우 패치를 한 번에 추출 ──
    # 결과: (1, 3*tile*tile, N)  N = 패치 개수
    patches = F.unfold(frame_t, kernel_size=tile, stride=stride)
    N  = patches.shape[2]
    nx = (w - tile) // stride + 1  # x 방향 패치 수

    if N == 0:
        return []

    # ── (N, 3, tile, tile) 로 변환 후 IMG_SIZE로 일괄 리사이즈 ──
    patches = patches.squeeze(0).permute(1, 0).reshape(N, 3, tile, tile)
    if tile != IMG_SIZE:
        patches = F.interpolate(patches, size=(IMG_SIZE, IMG_SIZE),
                                mode='bilinear', align_corners=False)

    # ── 모델 추론 ──
    patches_flat = patches.reshape(N, -1).to(_device)
    with torch.no_grad():
        probs = F.softmax(_model(patches_flat), dim=1)
        conf, preds = probs.max(1)
        conf  = conf.cpu().numpy()
        preds = preds.cpu().numpy()

    # ── 히트 패치 좌표 계산 & 마스크 생성 ──
    # Unfold 순서: y(행) 우선, x(열) 나중  →  i = yi*nx + xi
    hit = (preds == target_idx) & (conf > 0.9)
    hit_idx = np.where(hit)[0]
    xs = (hit_idx % nx) * stride
    ys = (hit_idx // nx) * stride
    for x, y in zip(xs, ys):
        mask[y:y+tile, x:x+tile] = 255

    # 윤곽선 → 무게중심 (game.py detect_red()와 동일)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy, cnt))
    return centers

# %%
