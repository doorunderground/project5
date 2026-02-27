#%%
import cv2
import numpy as np

# chessboard.jpg를 프로젝터(빔) 해상도 (1280x720)으로 resize
# 그 이미지에서 체스보드 코너 위치를 픽셀 좌표로 추출 할거임

PROJ_W, PROJ_H = 1280, 720
board_size = (7, 5)
screen_image = cv2.imread('chessboard.jpg')
screen_image_proj = cv2.resize(screen_image, (PROJ_W, PROJ_H))
screen_ret, screen_corners = cv2.findChessboardCorners(screen_image_proj, board_size)

#board_size = (7, 5)
#screen_image = cv2.imread('chessboard.jpg')
#screen_ret, screen_corners = cv2.findChessboardCorners(screen_image, board_size)

if not screen_ret:
    raise RuntimeError("No image")


# 2. 카메라 촬영 후 Homography 계산 - init()함수
# 카메라가 본 벽(체스판) 좌표를 -> 프로젝터(스크린) 좌표로 바꿔주는 변환 행렬
mat = np.eye(3, dtype=np.float32)
# [1 0 0
# 0 1 0
# 0 0 1]

def init(camera_image):
    global mat
    cam_ret, cam_corners = cv2.findChessboardCorners(camera_image, board_size)
    # 카메라로 촬영한 이미지에서 체스보드 코너 35개 (7x5) 검출
    '''
    카메라가 본 체스보드:                   프로젝터 이미지의 체스보드:
    (기울어지고 왜곡됨)                         (정면, 반듯함)

     *  *  *  *  *  *  *                    *  *  *  *  *  *  *
       *  *  *  *  *  *  *                  *  *  *  *  *  *  *
          *  *  *  *  *  *  *               *  *  *  *  *  *  *
             *  *  *  *  *  *  *            *  *  *  *  *  *  *
                *  *  *  *  *  *  *         *  *  *  *  *  *  *
    cam_corner는 벽이 비스듬하면 찌그러진 상태 그대로 좌표가 나옴
    '''
    
    if not cam_ret:
        return False
    
    # Homography
    mat, _ = cv2.findHomography(cam_corners.reshape(-1, 2), screen_corners.reshape(-1, 2))
    # (35, 1, 2) -> (35, 2)    / 2 = {x1,y1}좌표
    return True


# 카메라 좌표 -> 프로젝터 좌표로 바꾸는 함수
def trans(cam_pos, cam_pos2 = None):
    print("MAT", mat, cam_pos, cam_pos2)
    if cam_pos2 is not None:
        cam_pos = (cam_pos, cam_pos2)
    
    p = np.array(cam_pos).reshape(1, 2)
    # (320, 240) -> [[320, 240]]
    p = np.concatenate([p, np.ones((1, 1))], axis=1)
    # Homomgraphy는 3x3 행렬
    # (x,y)에 바로 못 곱함
    # (x,y) -> (x, y, 1)
    
    d = (p @ mat.T).flatten()
    # [x  y  1] x [3 x 3]행렬
    # [u', v', w']
    
    return d[0]/d[2],  d[1]/d[2]
    
'''
    만약 손이 카메라 화면에서
    (400, 300)에 있다면? -> (850, 420)으로 바꿔줘야함
    
    카메라에서 (400, 300)에 위치한 손은
    프로젝터 화면에서 (850, 420) 위치에 해당한다.
'''
    
    
# 테스트용 촬영
# 카메라가 본 찌그러진 벽 화면을
# 프로젝터 좌표계 기준으로 반듯하게 펴는 것
def test():
    camera_image = cv2.imread('test_camera.jpg')
    init(camera_image)

    dst = cv2.warpPerspective(camera_image, mat, (1920, 1080))
    cv2.imshow('dst', dst)
    cv2.waitKey(0)


# %%
