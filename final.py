import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# =====================================================================
# ⚙️ 기본 설정
# =====================================================================
WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.6       # 전체 스캔 시간 0.6초
FS = 44100
MIN_FREQ = 220.0
MAX_FREQ = 880.0
TOTAL_SAMPLES = int(FS * FRAME_TIME)
DELAY_TIME = 0.06

# 32개 주파수 정의
freqs = np.geomspace(MIN_FREQ, MAX_FREQ, HEIGHT)
freq_weights = np.linspace(0.8, 1.0, HEIGHT, dtype=np.float32)

# 🚀 [사용자 정의 x축] 전체 시간을 사용자 수식 스케일(x)로 변환
# 1개 열당 x가 10만큼 증가한다고 볼 때 (32개 열 = 0 ~ 320 스케일)
# x=9 ~ 11 구간이 정확히 열 경계선(10) 전후 10% 구간이 됨
X_MAX = WIDTH * 10.0
x = np.linspace(0, X_MAX, TOTAL_SAMPLES, dtype=np.float32)

# 고정 주파수 사인파 sin(2*pi*f*t)
t = np.arange(TOTAL_SAMPLES) / FS
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 32개 열의 중심 x 좌표 (5, 15, 25, ..., 315)
col_x_centers = (np.arange(WIDTH) * 10.0) + 5.0

# 스피커 팝 노이즈 방지용 3ms 극소 페이드
FADE_SAMPLES = int(FS * 0.003)
fade_envelope = np.ones(TOTAL_SAMPLES, dtype=np.float32)
fade_envelope[:FADE_SAMPLES] = np.linspace(0, 1, FADE_SAMPLES)
fade_envelope[-FADE_SAMPLES:] = np.linspace(1, 0, FADE_SAMPLES)


class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): raise RuntimeError("Camera open failed")
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        self.ok, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok: self.ok, self.frame = ok, frame
            else: time.sleep(0.01)

    def read(self): return self.ok, self.frame
    def release(self): self.running = False; self.cap.release()


cam = CameraStream()
sd.default.device = None
print("Vision.py Running (Time-Axis Formula Applied)... Press Ctrl+C to quit.")

try:
    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        small_flipped = np.flipud(small)
        
        # 0 ~ 9 단계 양자화 진폭 (32x32)
        small_step = (small_flipped / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        amps_grid = (small_step / 9.0) * freq_weights[:, None]

        # -----------------------------------------------------------------
        # 🚀 사용자 수식 반영:
        # 시간 x축 상에서 1열(a) -> 2열(b) 경계 구간 동안
        # f(x) = ((b-a)/2)*(x-10) + ((b-a)/2) + a 수식으로 연속 연결
        # -----------------------------------------------------------------
        f_x_matrix = np.zeros((HEIGHT, TOTAL_SAMPLES), dtype=np.float32)
        
        for h in range(HEIGHT):
            # np.interp는 점과 점 사이를 직선 f(x) = m*x + n 으로 연결하므로
            # 사용자님의 구간별 직선 수식 f(x)와 수학적으로 완벽히 동일합니다.
            f_x_matrix[h] = np.interp(x, col_x_centers, amps_grid[h])

        # y = f(x) * sin(2 * pi * f * t)
        combined_wave = np.sum(base_waves * f_x_matrix, axis=0)

        # 양 끝단 팝 노이즈 방지 페이드 & 음량 정규화
        combined_wave *= fade_envelope
        combined_wave /= (HEIGHT * 0.4)

        # 재생
        sd.play(combined_wave, FS)
        sd.wait()
        
        time.sleep(DELAY_TIME)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    sd.stop()
    cam.release()
    print("Program terminated successfully.")