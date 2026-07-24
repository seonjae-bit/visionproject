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
FRAME_TIME = 0.6
FS = 44100
MIN_FREQ = 220.0       # 220Hz (1옥타브 라 / 화면 맨 아래)
MAX_FREQ = 880.0       # 880Hz (3옥타브 라 / 화면 맨 위)
TOTAL_SAMPLES = int(FS * FRAME_TIME)
DELAY_TIME = 0.06      # 휴식 시간

# 1. 0.6초 전체 시간 축
t = np.arange(TOTAL_SAMPLES) / FS

# 2. 🚀 [주파수 차원 정렬]
# index 0: 저음(220Hz) -> index 31: 고음(880Hz)
freqs = np.geomspace(MIN_FREQ, MAX_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 저음 감쇄 보정 가중치
freq_weights = np.linspace(0.8, 1.0, HEIGHT, dtype=np.float32)

# 3. 열(Column) 보간 가중치 행렬
col_centers = (np.arange(WIDTH) + 0.5) * (TOTAL_SAMPLES / WIDTH)
sample_indices = np.arange(TOTAL_SAMPLES)

col_weights = np.array([
    np.interp(sample_indices, col_centers, (np.arange(WIDTH) == c).astype(float))
    for c in range(WIDTH)
], dtype=np.float32)

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
print("Vision.py Running (Fixed Pitch Sweep Issue)... Press Ctrl+C to quit.")

try:
    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 🚀 [핵심 수정] 이미지의 상하(Y축) 반전 처리
        # 이미지 0행(맨 위) -> 고음(880Hz / index 31)
        # 이미지 31행(맨 아래) -> 저음(220Hz / index 0)
        small_flipped = np.flipud(small)
        
        small_step = (small_flipped / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        
        # (32, 32) 진폭 행렬
        amps = (small_step / 9.0) * freq_weights[:, None]

        # (32, 32) @ (32, TOTAL_SAMPLES) -> (32, TOTAL_SAMPLES)
        smooth_amps = np.dot(amps, col_weights)
        
        # 전체 32개 주파수 파형 합성
        combined_wave = np.sum(base_waves * smooth_amps, axis=0)
        
        # 3ms 극소 페이드 및 음량 스케일링
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