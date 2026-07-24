import cv2
import numpy as np
import sounddevice as sd
import threading
import time

WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.6       # 스캔 오디오 길이 0.6초
FS = 44100
MIN_FREQ = 220.0
MAX_FREQ = 880.0
TOTAL_SAMPLES = int(FS * FRAME_TIME)
DELAY_TIME = 0.06      # 순수 기다리는 시간 (time.sleep)

t = np.arange(TOTAL_SAMPLES) / FS
freqs = np.geomspace(MAX_FREQ, MIN_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)
freq_weights = np.linspace(1.0, 0.8, HEIGHT, dtype=np.float32)

# 삼각형 보간 축
col_centers = (np.arange(WIDTH) + 0.5) * (TOTAL_SAMPLES / WIDTH)
sample_indices = np.arange(TOTAL_SAMPLES)

col_weights = np.array([
    np.interp(sample_indices, col_centers, (np.arange(WIDTH) == c).astype(float))
    for c in range(WIDTH)
], dtype=np.float32)

# 🚀 스피커 팍! 튀는 것만 방지하는 3ms (132개 샘플) 극소 페이드
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

try:
    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        small_step = (small / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        
        amps_per_col = (small_step / 9.0) * freq_weights[:, None]

        # 삼각형 보간
        smooth_amps = np.dot(amps_per_col, col_weights)
        combined_wave = np.sum(base_waves * smooth_amps, axis=0)
        
        # 3ms 극소 페이드 적용 (스피커 충격음 방지)
        combined_wave *= fade_envelope
        combined_wave /= (HEIGHT * 0.5)

        # 1. 0.6초 오디오 재생
        sd.play(combined_wave, FS)
        sd.wait()
        
        # 2. 0.06초 순수 기다리기 (오디오 없음)
        time.sleep(DELAY_TIME)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    sd.stop()
    cam.release()