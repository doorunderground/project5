import cv2
import numpy as np
import random
from color_detect import detect_color

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
score = 0
hit_effect = 0
hit_cooldown = 0

def load_basketball(radius):
    """농구공 이미지 로드 및 리사이즈 (BGRA)"""
    img = cv2.imread("game_image/basketball.png", cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("game_image/basketball.png 파일을 찾을 수 없습니다.")
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


def cam_to_proj(cx, cy):
    """카메라 좌표 → 프로젝터 좌표 변환"""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return transformed[0][0]

def main():
    global ball_pos, ball_vel, score, hit_effect, hit_cooldown

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

            if ball_pos[0] < ball_radius or ball_pos[0] > PROJ_W - ball_radius:
                ball_vel[0] *= -1

            if ball_pos[1] > PROJ_H + ball_radius:
                ball_pos[0] = random.randint(100, PROJ_W - 100)
                ball_pos[1] = -ball_radius
                ball_vel[0] = random.choice([-5, 5])
                ball_vel[1] = 2
                print("MISS - 다시 시작!")

            if hit_cooldown > 0:
                hit_cooldown -= 1

            # --- 빨간색 물체 감지 및 충돌 판정 ---
            obj_proj_positions = []
            detected = detect_color(frame, target='red', stride=16)

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
                        score += 1
                        hit_effect = 15
                        hit_cooldown = 20
                        print(f"HIT! 현재 점수: {score}")

            # --- 프로젝터 출력화면 그리기 ---
            display = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)
            cv2.rectangle(display, (0, 0), (PROJ_W - 1, PROJ_H - 1), (255, 255, 255), 3)

            for hx, hy in obj_proj_positions:
                if 0 <= hx < PROJ_W and 0 <= hy < PROJ_H:
                    cv2.circle(display, (hx, hy), 45, (0, 200, 255), 3)
                    cv2.circle(display, (hx, hy), 6, (0, 200, 255), -1)

            overlay_image(display, basketball_img, int(ball_pos[0]), int(ball_pos[1]))

            if hit_effect > 0:
                cv2.putText(display, "HIT!", (int(ball_pos[0]) - 40, int(ball_pos[1]) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                hit_effect -= 1

            cv2.putText(display, f"Score: {score}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

            cv2.imshow("Camera_Feed", frame)
            cv2.imshow("Projector_Window", display)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
