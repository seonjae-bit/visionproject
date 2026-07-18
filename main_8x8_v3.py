import cv2
import numpy as np
import sounddevice as sd

# ==========================
# 설정
# ==========================

WIDTH = 8
HEIGHT = 8

FRAME_TIME = 0.5              # 한 장면 재생 시간
SILENCE_RATIO = 0.10          # 장면 끝 무음 비율

FS = 44100

MIN_FREQ = 131.0
MAX_FREQ = 2093.0

FADE_MS = 2

OUTPUT_DEVICE = 1

# ==========================
# 자동 계산
# ==========================

COLUMN_TIME = FRAME_TIME / WIDTH
SILENCE_TIME = FRAME_TIME * SILENCE_RATIO

COLUMN_SAMPLES = int(FS * COLUMN_TIME)
SILENCE_SAMPLES = int(FS * SILENCE_TIME)

# ==========================
# 사인파 미리 계산
# ==========================

t = np.arange(COLUMN_SAMPLES) / FS

freqs = np.linspace(
    MIN_FREQ,
    MAX_FREQ,
    HEIGHT
)

waves = np.zeros(
    (HEIGHT, COLUMN_SAMPLES),
    dtype=np.float32
)

for i, f in enumerate(freqs):
    waves[i] = np.sin(2 * np.pi * f * t)

# ==========================
# Fade In / Fade Out
# ==========================

fade_samples = max(
    1,
    int(FS * FADE_MS / 1000)
)

fade = np.ones(
    COLUMN_SAMPLES,
    dtype=np.float32
)

fade[:fade_samples] = np.linspace(
    0,
    1,
    fade_samples
)

fade[-fade_samples:] = np.linspace(
    1,
    0,
    fade_samples
)

waves *= fade

# ==========================
# 무음 미리 생성
# ==========================

silence = np.zeros(
    SILENCE_SAMPLES,
    dtype=np.float32
)

# ==========================
# 오디오 출력 장치
# ==========================

sd.default.device = (None, OUTPUT_DEVICE)

# ==========================
# 카메라
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("Press Q to quit")
# ==========================
# 메인 루프
# ==========================

while True:

    # 8개의 열을 순서대로 재생
    for column in range(WIDTH):

        # --------------------------
        # 최신 카메라 프레임 읽기
        # --------------------------

        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        # --------------------------
        # 흑백 변환
        # --------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # --------------------------
        # 8×8 축소
        # --------------------------

        small = cv2.resize(
            gray,
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        # --------------------------
        # 현재 열만 추출
        # --------------------------

        brightness = (
            small[:, column].astype(np.float32)
            / 255.0
        )

        # 밝기²
        brightness = brightness ** 2

        # --------------------------
        # 컬럼 오디오 생성
        # --------------------------

        column_audio = np.zeros(
            COLUMN_SAMPLES,
            dtype=np.float32
        )

        for y in range(HEIGHT):

            column_audio += (
                waves[y]
                * brightness[y]
            )

        # --------------------------
        # 볼륨 정규화
        # --------------------------

        peak = np.max(
            np.abs(column_audio)
        )

        if peak > 1e-6:

            column_audio *= (
                0.9 / peak
            )
        # --------------------------
        # 컬럼 재생
        # --------------------------

        sd.play(
            column_audio,
            FS
        )

        # 컬럼 재생이 끝날 때까지 대기
        sd.wait()

    # ==========================
    # 프레임 종료(10% 무음)
    # ==========================

    sd.play(
        silence,
        FS
    )

    sd.wait()

    # ==========================
    # 화면 출력
    # ==========================

    cv2.imshow(
        "8x8",
        cv2.resize(
            small,
            (320, 320),
            interpolation=cv2.INTER_NEAREST
        )
    )

    # ==========================
    # 종료
    # ==========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
# ==========================
# 프로그램 종료
# ==========================

sd.stop()

cap.release()

cv2.destroyAllWindows()

print("프로그램 종료")