import cv2
import numpy as np
import calibration

def main():
    # 1. 빔프로젝터에 체스보드 띄우기 (전체 화면)
    screen_image = cv2.imread('chessboard.jpg')
    cv2.namedWindow('Projector View', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Projector View', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Projector View', screen_image)
    cv2.waitKey(1000) # 프로젝터에 화면이 뜰 시간 확보

    # 2. 카메라 연결
    cap = cv2.VideoCapture(0)

    stable_count = 0
    REQUIRED_STABLE = 5  # 5프레임 연속 성공해야 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success = calibration.init(frame)

        if success:
            stable_count += 1
            cv2.putText(frame, f"Stable: {stable_count}/{REQUIRED_STABLE}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            stable_count = 0
            cv2.putText(frame, "Detecting...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Camera View', frame)

        if stable_count >= REQUIRED_STABLE:
            print("Callibration Success")
            np.save('calibration_matrix.npy', calibration.mat)
            break

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()