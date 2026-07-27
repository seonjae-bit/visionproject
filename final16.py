import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# =====================================================================
# ⚙️ [동기식 위상 제어] n = 60Hz 시스템 설정
# =====================================================================
WIDTH = 16
HEIGHT = 16
FS = 44100
DELAY_TIME = 0.05

# 1. Base Frequency n = 60Hz 설정
N_BASE = 30.0                    # n = 60
T_UNIT = 1.0 / N_BASE             # 1/n 초 = 약 0.01667초 (16.67ms)

# 2. 16개 주파수 채널 설정 (n의 정수 배수: 4배수 ~ 19배수)
m_multipliers = np.arange(4, 4 + HEIGHT, dtype=np.int32)
freqs = m_multipliers * N_BASE    # 모든 주파수가 n=60Hz의 완벽한 정수 배수

# 3. 열(Column)당 시간 구조 설계
# [메인 구간]: 2 * T_UNIT (33.33ms, 진폭 a 고정)
# [경계 B(x) 구간]: 1 * T_UNIT (16.67ms, 진폭 a -> b 코사인 보간)
SAMPLES_MAIN = int(FS * (2 * T_UNIT))
SAMPLES_TRANS = int(FS * (1 * T_UNIT))
SAMPLES_PER_COL = SAMPLES_MAIN + SAMPLES_TRANS

TOTAL_SAMPLES = SAMPLES_PER_COL * WIDTH

# 4. 각 주파수별 기본 파형 미리 생성 (위상 연속성 100% 보장)
t_col = np.arange(SAMPLES_PER_COL) / FS
base_waves_col = np.array([np.sin(2 * np.pi * f * t_col) for f in freqs], dtype=np.float32)

# =====================================================================
# 🌊 [자연 보간 핵심 수정] B(x) 경계선 보간용 코사인 가중치 곡선 (0.0 -> 1.0)
# =====================================================================
# t: 0.0부터 1.0까지 변화하는 시간 비율
t_ramp = np.linspace(0.0, 1.0, SAMPLES_TRANS, dtype=np.float32)

# 코사인 보간 공식 적용: 양 끝점에서 기울기(1차 미분)가 0이 됨
ramp_up = ((1.0 - np.cos(t_ramp * np.pi)) / 2.0).astype(np.float32)

freq_weights = np.linspace(0.8, 1.0, HEIGHT, dtype=np.float32)

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
print(f"Vision.py Running (Cosine-Interpolated Natural Interpolation System: n={N_BASE}Hz)...")

try:
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
        amps_grid = (small_step / 9.0) * freq_weights[:, None]

        col_audio_list = []

        for c in range(WIDTH):
            a = amps_grid[:, c]                               # 현재 열 진폭
            b = amps_grid[:, (c + 1) % WIDTH]                 # 다음 열 진폭
            
            # -------------------------------------------------------------
            # 🚀 [자연 보간 알고리즘 - Cosine Version]
            # 1) 메인 구간: 진폭 a 고정 (y = a * sin)
            # 2) 경계 B(x) 구간: 코사인 곡선을 통해 진폭 a -> b로 완만하게 보간
            #    (접선의 기울기가 0으로 수렴하여 전 구간 미분 가능)
            # -------------------------------------------------------------
            col_envelope = np.zeros((HEIGHT, SAMPLES_PER_COL), dtype=np.float32)
            
            # 메인 구간 (a 고정)
            col_envelope[:, :SAMPLES_MAIN] = a[:, None]
            
            # 경계 B(x) 구간 (a -> b 코사인 보간)
            b_x = a[:, None] + (b - a)[:, None] * ramp_up[None, :]
            col_envelope[:, SAMPLES_MAIN:] = b_x
            
            # 파형 합성: Envelope * 고정 주파수 사인파
            col_wave = np.sum(base_waves_col * col_envelope, axis=0)
            col_audio_list.append(col_wave)

        # 전체 파형 통합
        combined_wave = np.concatenate(col_audio_list)
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