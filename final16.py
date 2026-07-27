import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# =====================================================================
# ⚙️ [동기식 위상 제어] 시스템 설정
# =====================================================================
WIDTH = 16
HEIGHT = 16
FS = 44100
DELAY_TIME = 0.01  # 지연 시간 최소화

# 1. Base Frequency n 설정 (30.0Hz -> 약 1.6초 스캔)
N_BASE = 30.0                     
T_UNIT = 1.0 / N_BASE             

# 2. 16개 주파수 채널 설정
m_multipliers = np.arange(4, 4 + HEIGHT, dtype=np.int32)
freqs = m_multipliers * N_BASE    

# 3. 열(Column)당 시간 구조 및 샘플 수
SAMPLES_MAIN = int(FS * (2 * T_UNIT))
SAMPLES_TRANS = int(FS * (1 * T_UNIT))
SAMPLES_PER_COL = SAMPLES_MAIN + SAMPLES_TRANS
TOTAL_SAMPLES = SAMPLES_PER_COL * WIDTH

# 4. [최적화 1] 기본 파형 사전 생성
t_col = np.arange(SAMPLES_PER_COL, dtype=np.float32) / FS
base_waves_col = np.array([np.sin(2 * np.pi * f * t_col) for f in freqs], dtype=np.float32)

# 5. [최적화 2] 코사인 보간 가중치 곡선 사전 생성 (루프 밖으로 이동)
t_ramp = np.linspace(0.0, 1.0, SAMPLES_TRANS, dtype=np.float32)
ramp_up = ((1.0 - np.cos(t_ramp * np.pi)) / 2.0).astype(np.float32)
# BroadCast를 위한 형태 미리 변환 (1, SAMPLES_TRANS)
ramp_up_2d = ramp_up[None, :]

freq_weights = np.linspace(0.8, 1.0, HEIGHT, dtype=np.float32)[:, None]

# 6. [최적화 3] 페이드 엔벨롭 및 전체 출력 버퍼 미리 생성
FADE_SAMPLES = int(FS * 0.003)
fade_envelope = np.ones(TOTAL_SAMPLES, dtype=np.float32)
fade_envelope[:FADE_SAMPLES] = np.linspace(0, 1, FADE_SAMPLES, dtype=np.float32)
fade_envelope[-FADE_SAMPLES:] = np.linspace(1, 0, FADE_SAMPLES, dtype=np.float32)

# 최종 합성 오디오 버퍼 메모리 미리 할당
combined_wave = np.zeros(TOTAL_SAMPLES, dtype=np.float32)


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
print(f"Vision.py Running (Optimized Cosine Natural Interpolation)...")

try:
    # 루프 시작 전 엔벨롭 전체 배열 미리 할당
    col_envelope = np.zeros((HEIGHT, SAMPLES_PER_COL), dtype=np.float32)

    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        small_flipped = np.flipud(small)
        
        # 0 ~ 9 단계 양자화 진폭
        small_step = (small_flipped / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        amps_grid = (small_step / 9.0) * freq_weights

        # 벡터화 연산을 위해 포문 내 연산 최적화
        for c in range(WIDTH):
            a = amps_grid[:, c:c+1]                            # (HEIGHT, 1)
            b = amps_grid[:, (c + 1) % WIDTH : (c + 1) % WIDTH + 1] # (HEIGHT, 1)
            
            # 메인 구간 (a 고정)
            col_envelope[:, :SAMPLES_MAIN] = a
            
            # 경계 B(x) 구간 (코사인 보간)
            col_envelope[:, SAMPLES_MAIN:] = a + (b - a) * ramp_up_2d
            
            # 파형 합성 및 슬라이스에 즉시 삽입 (메모리 재할당 방지)
            start_idx = c * SAMPLES_PER_COL
            end_idx = start_idx + SAMPLES_PER_COL
            combined_wave[start_idx:end_idx] = np.sum(base_waves_col * col_envelope, axis=0)

        # 전체 파형 후처리
        combined_wave *= fade_envelope
        combined_wave /= (HEIGHT * 0.4)

        # 재생 (비동기로 빠르고 부드럽게 재생)
        sd.play(combined_wave, FS)
        sd.wait()
        
        time.sleep(DELAY_TIME)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    sd.stop()
    cam.release()
    print("Program terminated successfully.")