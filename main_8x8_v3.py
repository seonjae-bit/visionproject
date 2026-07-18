import cv2
import numpy as np
import sounddevice as sd

# =========================
# 영상 / 소리 기본 설정
# =========================
WIDTH = 64
HEIGHT = 64

FS = 44100

# 한 프레임을 가로로 스캔하는 시간
COLUMN_TIME = 0.5 / WIDTH   # 전체 프레임 길이 = 0.5초

# 주파수 범위 (행 방향 매핑)
MIN_FREQ = 131
MAX_FREQ = 4186

# 프레임 사이 무음 구간 (딜레이 역할)
SILENCE_TIME = COLUMN_TIME * 5
silence = np.zeros(int(FS * SILENCE_TIME), dtype=np.float32)

# =========================
# 카메라 초기화
# =========================
cap = cv2.VideoCapture(0)

# =========================
# 주파수 테이블 (행 → 음높이)
# =========================
freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)

t = np.linspace(
    0,
    COLUMN_TIME,
    int(FS * COLUMN_TIME),
    endpoint=False
)

# 각 주파수별 사인파 미리 생성 (속도 최적화)
wave_bank = np.array([
    np.sin(2 * np.pi * f * t)
    for f in freqs
], dtype=np.float32)

# =========================
# 오디오 스트림 (실시간 출력)
# =========================
stream = sd.OutputStream(
    samplerate=FS,
    channels=1,
    blocksize=2048,
    dtype='float32'
)
stream.start()

# =========================
# 메인 루프
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # 1. 그레이스케일 변환
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 2. 64×64 축소
    # -------------------------
    small = cv2.resize(
        gray,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # -------------------------
    # 3. 화면 표시용 확대
    # -------------------------
    enlarged = cv2.resize(
        small,
        (400, 400),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow("Grayscale 64x64", enlarged)

    # =========================
    # 4. 이미지 → 소리 변환
    # =========================
    full_sound = []

    for x in range(WIDTH):

        column = small[:, x]

        # 0~1 정규화
        brightness = column.astype(np.float32) / 255.0

        # 밝기 강조 (비선형)
        brightness = brightness ** 2

        # 행 방향 주파수 합성
        sound = np.sum(
            wave_bank * brightness[:, None],
            axis=0
        )

        # 전체 음량 조절
        sound *= 0.05

        # 클릭 노이즈 방지 (fade in/out)
        fade_len = min(int(0.01 * FS), len(sound) // 2)

        if fade_len > 0:
            fade = np.linspace(0, 1, fade_len, dtype=np.float32)

            sound[:fade_len] *= fade
            sound[-fade_len:] *= fade[::-1]

        full_sound.append(sound)

    # 열 방향 연결 (좌→우 스캔)
    full_sound = np.concatenate(full_sound)

    # 프레임 간 구분용 무음 추가
    full_sound = np.concatenate([full_sound, silence])

    # =========================
    # 5. 오디오 출력 (스트리밍)
    # =========================
    stream.write(full_sound.astype(np.float32))

    # ESC 종료
    if cv2.waitKey(1) == 27:
        break

# =========================
# 정리
# =========================
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()