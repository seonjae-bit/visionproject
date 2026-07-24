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
FRAME_TIME = 0.6       # 재생 시간 0.6초
FS = 44100
MIN_FREQ = 220.0       # 220Hz (1옥타브 라 / A3)
MAX_FREQ = 880.0       # 880Hz (3옥타브 라 / A5)
TOTAL_SAMPLES = int(FS * FRAME_TIME)

# 휴식 시간 (0.06초)
DELAY_TIME = FRAME_TIME * 0.10

# 1. 0.6초 전체 통째로 흐르는 시간(t) 축 생성
t = np.arange(TOTAL_SAMPLES) / FS

# 2. 220Hz ~ 880Hz 지수(Logarithmic) 스케일 배치
freqs = np.geomspace(MAX_FREQ, MIN_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 저음 20% 감쇄 가중치
freq_weights = np.linspace(1.0, 0.8, HEIGHT, dtype=np.float32)

# 3. 🚀 [범인 잡기] 박스형/사다리꼴 보간 가중치 행렬 생성 (1열~32열 평평 유지)
SAMPLES_PER_COL = TOTAL_SAMPLES / WIDTH
RAMP_SAMPLES = int(SAMPLES_PER_COL * 0.1)  # 경계면 10%만 부드럽게 경사 처리

col_weights = np.zeros((WIDTH, TOTAL_SAMPLES), dtype=np.float32)
for c in range(WIDTH):
    start = int(c * SAMPLES_PER_COL)
    end = int((c + 1) * SAMPLES_PER_COL)
    
    # 박스 구간 1.0 채우기
    col_weights[c, start:end] = 1.0
    
    # 앞뒤 경계 10% 사다리꼴(Ramp) 페이드
    if start > 0:
        ramp_in = np.linspace(0, 1, RAMP_SAMPLES)
        col_weights[c, start:start+RAMP_SAMPLES] = ramp_in
    if end < TOTAL_SAMPLES:
        ramp_out = np.linspace(1, 0, RAMP_SAMPLES)
        col_weights[c, end-RAMP_SAMPLES:end] = ramp_out


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
print("Vision.py Running (Flat Trapezoid Ramp, 220~880Hz)... Press Ctrl+C to quit.")

sd.default.device = None

# 전체 스캔 시작/끝에 3ms짜리 극소 페이드로 클릭음 방지
edge_fade = int(FS * 0.003)
fade_in_out = np.ones(TOTAL_SAMPLES, dtype=np.float32)
fade_in_out[:edge_fade] = np.linspace(0, 1, edge_fade)
fade_in_out[-edge_fade:] = np.linspace(1, 0, edge_fade)

try:
    while True:
        ok, frame = cam.read()
        if not ok: 
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0  # 노이즈 제거
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 0 ~ 9 단계 양자화
        small_step = (small / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        
        amps_per_col = small_step / 9.0
        amps_per_col *= freq_weights[:, None]

        # 사다리꼴 가중치 곱
        smooth_amps = np.dot(amps_per_col, col_weights)

        # 파형 합성
        combined_wave = np.sum(base_waves * smooth_amps, axis=0)
        combined_wave *= fade_in_out  # 양 끝 3ms 아주 살짝 페이드
        
        # 절대적인 고정 스케일링 (정규화 펌핑 현상 완전 제거)
        combined_wave = combined_wave / (HEIGHT * 0.5)

        # 오디오 재생
        sd.play(combined_wave, FS)
        sd.wait()
        
        time.sleep(DELAY_TIME)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    sd.stop()
    cam.release()
    print("Program terminated successfully.")