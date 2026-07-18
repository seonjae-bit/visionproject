import cv2
import numpy as np
import sounddevice as sd
import threading
import time

WIDTH = 32
HEIGHT = 32
# [수정] 반응 속도를 높이기 위해 전체 재생 시간을 0.4초로 단축
FRAME_TIME = 0.8
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0

# 0.4초에 맞춘 총 샘플 수 및 기본 오디오 타임라인 자동 변경
total_samples = int(FS * FRAME_TIME)
t = np.arange(total_samples) / FS
freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
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
print("Press q to quit.")

while True:
    ok, frame = cam.read()
    if not ok: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    THRESHOLD_LEVEL = 15
    small[small < THRESHOLD_LEVEL] = 0
    
    max_pixel_val = np.max(small) / 255.0
    amp_matrix = (small.astype(np.float32) / 255.0) ** 2
    
    x_old = np.linspace(0, 1, WIDTH)
    x_new = np.linspace(0, 1, total_samples)
    
    smooth_amps = np.zeros((HEIGHT, total_samples), dtype=np.float32)
    for r in range(HEIGHT):
        smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
    
    audio_matrix = base_waves * smooth_amps
    audio = np.sum(audio_matrix, axis=0)
    
    m = np.max(np.abs(audio))
    avg_brightness = np.mean(small)
    
    if avg_brightness < 1.0 or m < 1e-4:
        audio = np.zeros_like(audio)
    else:
        audio *= (0.9 / m) * max_pixel_val
        
    cv2.imshow("Sonification", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
    sd.default.device = (None, 1)
    sd.play(audio, FS)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sd.stop()
        break
    sd.wait()

cam.release()
cv2.destroyAllWindows()