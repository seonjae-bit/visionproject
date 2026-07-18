import cv2
import numpy as np
import sounddevice as sd
import threading
import time

WIDTH = 32
HEIGHT = 32
FRAME_TIME = 1.0
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0

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
    
    # [개선 1] 카메라 자체의 자글자글한 고주파 노이즈를 뭉개주는 블러 필터 적용
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    # [개선 2] 암부 노이즈 차단 (밝기 0~255 중 15 이하는 완전한 어둠으로 처리)
    # 인간의 눈으로 식별하기 힘든 미세한 흔들림을 강제로 0으로 만듭니다.
    THRESHOLD_LEVEL = 15
    small[small < THRESHOLD_LEVEL] = 0
    
    amp_matrix = (small.astype(np.float32) / 255.0) ** 2
    
    x_old = np.linspace(0, 1, WIDTH)
    x_new = np.linspace(0, 1, total_samples)
    
    smooth_amps = np.zeros((HEIGHT, total_samples), dtype=np.float32)
    for r in range(HEIGHT):
        smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
    
    audio_matrix = base_waves * smooth_amps
    audio = np.sum(audio_matrix, axis=0)
    
    # [개선 3] 볼륨 정규화 조건 강화
    # 픽셀들의 평균 밝기가 너무 낮으면 노이즈를 키우지 않고 완전 음소거합니다.
    m = np.max(np.abs(audio))
    avg_brightness = np.mean(small)
    
    if avg_brightness < 1.0 or m < 1e-4:
        audio = np.zeros_like(audio)  # 완전한 침묵
    else:
        audio *= (0.9 / m)  # 의미 있는 소리가 있을 때만 정상 증폭
        
    cv2.imshow("Sonification", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
    sd.default.device = (None, 1)
    sd.play(audio, FS)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sd.stop()
        break
    sd.wait()

cam.release()
cv2.destroyAllWindows()