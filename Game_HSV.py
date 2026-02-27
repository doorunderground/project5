import cv2
import numpy as np
import random

# 1. 환경 설정
PROJ_W, PROJ_H = 1280, 720
calibration_file = "calibration_matrix.npy"

# 2. 변환 행렬 로드
try:
    H = np.load(calibration_file)
except Exception:
    print("캘리브레이션 파일 없음 또는 손상 → 단위 행렬로 대체")
    H = np.eye(3, dtype=np.float32)

# --- 게임 변수 설정 ---
ball_pos = [random.randint(100, PROJ_W-100), 50]
ball_vel = [random.choice([-5, 5]), 2]
gravity = 0.5
ball_radius = 50
score_left = 0
score_right = 0
hit_effect = 0
hit_cooldown = 0

def load_basketball(radius):
    """농구공 이미지 로드 및 리사이즈 (BGRA)"""
    img = cv2.imread("game_image/volleyball.png", cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("game_image/volleyball.png 파일을 찾을 수 없습니다.")
    size = radius * 2
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def overlay_image(bg, img_bgra, cx, cy):
    """BGRA 이미지를 배경에 알파 블렌딩으로 합성"""
    h, w = img_bgra.shape[:2]
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = x1 + w, y1 + h
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(bg.shape[1], x2), min(bg.shape[0], y2)
    if bx1 >= bx2 or by1 >= by2:
        return
    sx1, sy1 = bx1 - x1, by1 - y1
    src = img_bgra[sy1:sy1 + (by2 - by1), sx1:sx1 + (bx2 - bx1)]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    dst = bg[by1:by2, bx1:bx2].astype(np.float32)
    bg[by1:by2, bx1:bx2] = (dst * (1 - alpha) + src[:, :, :3] * alpha).astype(np.uint8)

basketball_img = load_basketball(ball_radius)

# 네트 이미지 로드 (하단 중앙 고정)
_net_raw = cv2.imread("game_image/net.png", cv2.IMREAD_UNCHANGED)
if _net_raw is None:
    raise FileNotFoundError("game_image/net.png 파일을 찾을 수 없습니다.")
if _net_raw.shape[2] == 3:
    _net_raw = cv2.cvtColor(_net_raw, cv2.COLOR_BGR2BGRA)
_net_h = 250  # 네트 높이(px), 필요시 조절
_net_scale = _net_h / _net_raw.shape[0]
_net_w = int(_net_raw.shape[1] * _net_scale)
net_img = cv2.resize(_net_raw, (_net_w, _net_h), interpolation=cv2.INTER_AREA)
NET_CX = PROJ_W // 2
NET_CY = PROJ_H - _net_h // 2  # 하단에 딱 붙도록
NET_TOP_Y = PROJ_H - _net_h    # 네트 꼭대기 Y 좌표

# 빨간색 HSV 범위 (빨강은 색상환 양 끝에 걸쳐 있어 두 범위 필요)
RED_LOWER1 = np.array([  0, 100, 100])  # H: 0~10 (하단 빨강)
RED_UPPER1 = np.array([ 10, 255, 255])
RED_LOWER2 = np.array([160, 100, 100])  # H: 160~180 (상단 빨강)
RED_UPPER2 = np.array([180, 255, 255])

MIN_AREA = 500  # 너무 작은 노이즈 무시

# GPU(CUDA) 자동 감지 및 필터 사전 생성
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _erode_filter  = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE,  cv2.CV_8UC1, _kernel, iterations=2)
    _dilate_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, _kernel, iterations=2)
    print("GPU(CUDA) 가속 활성화")
else:
    print("CPU 모드로 실행 (CUDA 없음 또는 미지원 OpenCV)")

def detect_red(frame):
    """빨간색 물체의 중심 좌표 목록 반환. 없으면 빈 리스트."""
    if USE_CUDA:
        # cvtColor → GPU, inRange → CPU(미지원), erode/dilate → GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV).download()
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    if USE_CUDA:
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(mask)
        gpu_mask = _erode_filter.apply(gpu_mask)
        gpu_mask = _dilate_filter.apply(gpu_mask)
        mask = gpu_mask.download()
    else:
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

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

def cam_to_proj(cx, cy):
    """카메라 좌표 → 프로젝터 좌표 변환"""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return transformed[0][0]

def reset_ball(side):
    """공을 지정된 side('left'/'right') 위쪽에서 다시 시작"""
    if side == 'left':
        ball_pos[0] = random.randint(100, NET_CX - 100)
    else:
        ball_pos[0] = random.randint(NET_CX + 100, PROJ_W - 100)
    ball_pos[1] = -ball_radius
    ball_vel[0] = random.choice([-5, 5])
    ball_vel[1] = 2

def main():
    global ball_pos, ball_vel, score_left, score_right, hit_effect, hit_cooldown

    cv2.namedWindow("Projector_Window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Projector_Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- 게임 로직 업데이트 ---
            ball_vel[1] += gravity
            ball_pos[0] += ball_vel[0]
            ball_pos[1] += ball_vel[1]

            # 좌우 벽 반사
            if ball_pos[0] < ball_radius:
                ball_vel[0] = abs(ball_vel[0])
            elif ball_pos[0] > PROJ_W - ball_radius:
                ball_vel[0] = -abs(ball_vel[0])

            # 네트 꼭대기 충돌 (위에서 내려올 때)
            if (ball_pos[1] + ball_radius >= NET_TOP_Y and
                    abs(ball_pos[0] - NET_CX) <= _net_w // 2):
                ball_vel[1] = -abs(ball_vel[1]) * 0.8
                ball_pos[1] = NET_TOP_Y - ball_radius

            # 네트 수직 장벽 (네트 옆면 통과 방지)
            if ball_pos[1] + ball_radius > NET_TOP_Y:
                if (ball_pos[0] - ball_radius < NET_CX < ball_pos[0] + ball_radius):
                    ball_vel[0] *= -1
                    if ball_vel[0] > 0:
                        ball_pos[0] = NET_CX + ball_radius
                    else:
                        ball_pos[0] = NET_CX - ball_radius

            # 공이 바닥 아래로 떨어지면 → 점수 처리
            if ball_pos[1] - ball_radius > PROJ_H:
                if ball_pos[0] < NET_CX:
                    score_right += 1
                    print(f"오른쪽 득점! L:{score_left} R:{score_right}")
                    reset_ball('right')
                else:
                    score_left += 1
                    print(f"왼쪽 득점! L:{score_left} R:{score_right}")
                    reset_ball('left')

            if hit_cooldown > 0:
                hit_cooldown -= 1

            # --- 빨간색 물체 감지 및 충돌 판정 ---
            obj_proj_positions = []
            detected = detect_red(frame)

            for (cx, cy, cnt) in detected:
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

                tx, ty = cam_to_proj(cx, cy)
                obj_proj_positions.append((int(tx), int(ty)))

                cv2.putText(frame, f"({int(tx)},{int(ty)})", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if hit_cooldown == 0:
                    dist = np.sqrt((tx - ball_pos[0])**2 + (ty - ball_pos[1])**2)
                    if dist < ball_radius + 40:
                        ball_vel[1] = -15
                        ball_vel[0] = random.randint(-10, 10)
                        hit_effect = 15
                        hit_cooldown = 20
                        print("HIT!")

            # --- 프로젝터 출력화면 그리기 ---
            display = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
            cv2.rectangle(display, (0, 0), (PROJ_W - 1, PROJ_H - 1), (255, 255, 255), 3)

            for hx, hy in obj_proj_positions:
                if 0 <= hx < PROJ_W and 0 <= hy < PROJ_H:
                    cv2.circle(display, (hx, hy), 45, (0, 200, 255), 3)
                    cv2.circle(display, (hx, hy), 6, (0, 200, 255), -1)

            overlay_image(display, basketball_img, int(ball_pos[0]), int(ball_pos[1]))
            overlay_image(display, net_img, NET_CX, NET_CY)

            if hit_effect > 0:
                cv2.putText(display, "HIT", (int(ball_pos[0]) - 40, int(ball_pos[1]) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                hit_effect -= 1

            # 피카츄 배구 스타일 점수 표시
            cv2.putText(display, str(score_left),  (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
            cv2.putText(display, str(score_right), (PROJ_W - 130, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)

            cv2.imshow("Camera_Feed", frame)
            cv2.imshow("Projector_Window", display)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
