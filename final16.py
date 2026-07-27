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

# 1. 음높이(주파수) 기준은 원래대로 60Hz 복구!
N_BASE = 60.0                     # 240Hz ~ 1140Hz 원래 음높이
T_UNIT = 1.0 / N_BASE             # 1/60 초 (약 16.67ms)

# 2. 재생 시간 배율 (이 값으로 속도만 조절합니다)
# TIME_SCALE = 1 -> 0.8초 스캔 (기존)
# TIME_SCALE = 2 -> 1.6초 스캔 (추천: 원래 음높이 + 느긋한 재생)
# TIME_SCALE = 3 -> 2.4초 스캔
TIME_SCALE = 2                    

m_multipliers = np.arange(4, 4 + HEIGHT, dtype=np.int32)
freqs = m_multipliers * N_BASE    # 주파수 범위 원래대로 복구!

# 3. 시간 배율(TIME_SCALE)을 곱해 재생 시간만 확대
SAMPLES_MAIN = int(FS * (2 * TIME_SCALE * T_UNIT))
SAMPLES_TRANS = int(FS * (1 * TIME_SCALE * T_UNIT))
SAMPLES_PER_COL = SAMPLES_MAIN + SAMPLES_TRANS
TOTAL_SAMPLES = SAMPLES_PER_COL * WIDTH

# 기본 파형 사전 생성
t_col = np.arange(SAMPLES_PER_COL, dtype=np.float32) / FS
base_waves_col = np.array([np.sin(2 * np.pi * f * t_col) for f in freqs], dtype=np.float32)

# 🌊 [코사인 보간 곡선 사전 생성]
t_ramp = np.linspace(0.0, 1.0, SAMPLES_TRANS, dtype=np.float32)
ramp_up_2d = (((1.0 - np.cos(t_ramp * np.pi)) / 2.0).astype(np.float32))[None, :]
freq_weights = np.linspace(0.8, 1.0, HEIGHT, dtype=np.float32)[:, None]

# 페이드 엔벨롭
FADE_SAMPLES = int(FS * 0.003)
fade_envelope = np.ones(TOTAL_SAMPLES, dtype=np.float32)
fade_envelope[:FADE_SAMPLES] = np.linspace(0, 1, FADE_SAMPLES, dtype=np.float32)
fade_envelope[-FADE_SAMPLES:] = np.linspace(1, 0, FADE_SAMPLES, dtype=np.float32)

# 🔔 [장면 시작 알림 삐- 소리 설정]
BEEP_FREQ = 1000.0                # 1000Hz 삐소리
BEEP_DUR = 0.03                   # 30ms
BEEP_SAMPLES = int(FS * BEEP_DUR)

t_beep = np.arange(BEEP_SAMPLES, dtype=np.float32) / FS
beep_wave = (0.35 * np.sin(2 * np.pi * BEEP_FREQ * t_beep)).astype(np.float32)

beep_fade = int(FS * 0.003)
beep_wave[:beep_fade] *= np.linspace(0, 1, beep_fade, dtype=np.float32)
beep_wave[-beep_fade:] *= np.linspace(1, 0, beep_fade, dtype=np.float32)

# 오디오 버퍼
latest_wave = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
wave_lock = threading.Lock()
sample_ptr = 0  


def audio_callback(outdata, frames, time_info, status):
    global sample_ptr, latest_wave
    
    with wave_lock:
        chunk_end = sample_ptr + frames
        if chunk_end <= TOTAL_SAMPLES:
            outdata[:, 0] = latest_wave[sample_ptr:chunk_end]
            sample_ptr = chunk_end % TOTAL_SAMPLES
        else:
            first_part = TOTAL_SAMPLES - sample_ptr
            second_part = frames - first_part
            outdata[:first_part, 0] = latest_wave[sample_ptr:]
            outdata[first_part:, 0] = latest_wave[:second_part]
            sample_ptr = second_part


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

print(f"Vision.py Running (Pitch Fixed: N_BASE={N_BASE}Hz, Time Scale={TIME_SCALE}x)...")

stream = sd.OutputStream(channels=1, samplerate=FS, callback=audio_callback)
stream.start()

try:
    col_envelope = np.zeros((HEIGHT, SAMPLES_PER_COL), dtype=np.float32)
    temp_wave = np.zeros(TOTAL_SAMPLES, dtype=np.float32)

    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        small_flipped = np.flipud(small)
        
        small_step = (small_flipped / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        amps_grid = (small_step / 9.0) * freq_weights

        for c in range(WIDTH):
            a = amps_grid[:, c:c+1]
            b = amps_grid[:, (c + 1) % WIDTH : (c + 1) % WIDTH + 1]
            
            col_envelope[:, :SAMPLES_MAIN] = a
            col_envelope[:, SAMPLES_MAIN:] = a + (b - a) * ramp_up_2d
            
            start_idx = c * SAMPLES_PER_COL
            end_idx = start_idx + SAMPLES_PER_COL
            temp_wave[start_idx:end_idx] = np.sum(base_waves_col * col_envelope, axis=0)

        temp_wave *= fade_envelope
        temp_wave /= (HEIGHT * 0.4)

        # 시작 알림 삐- 소리 추가
        temp_wave[:BEEP_SAMPLES] += beep_wave

        with wave_lock:
            latest_wave[:] = temp_wave

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop()
    stream.close()
    cam.release()
    print("Program terminated successfully.")