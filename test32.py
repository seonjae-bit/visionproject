
import cv2
import numpy as np
import sounddevice as sd
import threading
import time

WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.5
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0
COL_TIME = FRAME_TIME / WIDTH
FADE_MS = 1

samples = int(FS * COL_TIME)
t = np.arange(samples) / FS

freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)

fade = max(1, int(FS * FADE_MS / 1000))
env = np.ones(samples, np.float32)
fi = np.linspace(0, 1, fade)
fo = np.linspace(1, 0, fade)
env[:fade] = fi
env[-fade:] = fo
waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)
waves *= env

# =========================================================
# 💡 [저음 청각 보정 가중치 추가]
# 고음(맨 위, idx=0) = 1.0 (100%)
# 저음(맨 아래, idx=31) = 0.8 (80%, 즉 20% 감소)
# =========================================================
freq_weights = np.linspace(1.0, 0.8, HEIGHT, dtype=np.float32)

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
            
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)

        self.ok, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.ok = ok
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ok, self.frame

    def release(self):
        self.running = False
        self.cap.release()

cam = CameraStream()
print("Headless Scan Sonification Running... Press Ctrl+C to quit.")

sd.default.device = (None, 1)

try:
    while True:
        ok, frame = cam.read()
        if not ok: 
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 어두운 노이즈 보정
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 0 ~ 9 단계 (10단계 양자화)
        small_step = (small / 28.44).astype(np.int32)
        small_step = np.clip(small_step, 0, 9)
        
        audio = np.empty(samples * WIDTH, dtype=np.float32)
        
        for c in range(WIDTH):
            # 0~9 단계 진폭 변환
            amp = (small_step[:, c].astype(np.float32) / 9.0) ** 2
            
            # [수정] 저음 20% 감쇄 가중치(freq_weights) 적용
            amp *= freq_weights
            
            col = np.sum(waves * amp[:, None], axis=0)
            m = np.max(np.abs(col))
            if m > 1e-6: 
                col *= 0.9 / m
            audio[c * samples:(c + 1) * samples] = col
            
        # [모니터 제거] cv2.imshow 및 cv2.waitKey 완전히 제거
        sd.play(audio, FS)
        sd.wait()

except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected. Stopping...")

finally:
    sd.stop()
    cam.release()
    print("Program terminated successfully.")