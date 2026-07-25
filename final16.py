import cv2
import time

# ==========================================
# [설정] 장면 재생 후 휴식시간 (초 단위)
# ==========================================
REST_DURATION = 2.0  # 원하시는 휴식시간(초)을 적어주세요 (예: 2초, 3초)

last_played_time = 0
is_resting = False

# 카메라 설정 (사용 중인 인덱스로 설정)
cap = cv2.VideoCapture(0)

print("🚀 final16.py 실행 시작...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # ----------------------------------------------------
        # 1. 휴식시간(쿨타임) 체크 및 버퍼 비우기
        # ----------------------------------------------------
        if is_resting:
            # 설정한 휴식시간이 지났는지 확인
            if current_time - last_played_time >= REST_DURATION:
                is_resting = False
                print("🔄 휴식시간 종료! 다시 감지 시작")
            else:
                # 휴식시간 중일 때는 분석을 안 하고 프레임만 흘려보냄 (밀림 방지)
                continue

        # ----------------------------------------------------
        # 2. 장면/객체 분석 및 재생 로직
        # ----------------------------------------------------
        # (기존 조건문 위치)
        scene_detected = True  # 예시: 장면이 감지되었다고 가정

        if scene_detected:
            print("🔊 장면 재생!")
            
            # [재생 로직 실행]
            # play_audio() 또는 오디오/영상 재생 함수
            
            # 재생 완료 후 휴식시간 상태로 전환
            last_played_time = time.time()
            is_resting = True

        # CPU 과점유 방지용 아주 미세한 대기 (0.001초)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()