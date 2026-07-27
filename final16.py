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

N_BASE = 30.0                     # 30Hz -> 1.6초 1회 스캔
T_UNIT = 1.0 / N_BASE             

m_multipliers = np.arange(4, 4 + HEIGHT, dtype=np.int32)
freqs = m_multipliers * N_BASE    

SAMPLES_MAIN = int(FS * (2 * T_UNIT))
SAMPLES_TRANS = int(FS * (1 * T_UNIT))
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

# =====================================================================
# 🔔 [장면 시작 알림 삐- 소리 설정]
# =====================================================================
BEEP_FREQ = 1000.0                # 삐소리 주파수 (1000Hz)
BEEP_DUR = 0.03                   # 30ms (0.03초 동안 지속)
BEEP_SAMPLES = int(FS * BEEP_DUR)

t_beep = np.arange(BEEP_SAMPLES, dtype=np.float32) / FS
beep_wave = (0.35 * np.sin(2 * np.pi * BEEP_FREQ * t_beep)).astype(np.float32)

# 삐소리 팝 노이즈 방지용 페이드
beep_fade = int(FS * 0.003)
beep_wave[:beep_fade] *= np.linspace(0, 1, beep_fade, dtype=np.float32)
beep_wave[-beep_fade:] *= np.linspace(1, 0, beep_fade, dtype=np.float32)


# =====================================================================
# 🔄 [스레드 안전 오디오 버퍼 설정]
# =====================================================================
latest_wave = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
wave_lock = threading.Lock()
sample_ptr = 0  # 오디오 재생 포인터 위치


def audio_callback(outdata, frames, time_info, status):
    """
    백그라운드에서 실시간으로 버퍼를 연결해주는 스트리밍 콜백
    """
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

print(f"Vision.py Running (Seamless Audio Callback + Cosine Natural Interpolation + Beep)...")

# 백그라운드 오디오 스트림 시작
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

        # 16개 열 파형 계산 (코사인 보간 적용)
        for c in range(WIDTH):
            a = amps_grid[:, c:c+1]
            b = amps_grid[:, (c + 1) % WIDTH : (c + 1) % WIDTH + 1]
            
            col_envelope[:, :SAMPLES_MAIN] = a
            col_envelope[:, SAMPLES_MAIN:] = a + (b - a) * ramp_up_2d
            
            start_idx = c * SAMPLES_PER_COL
            end_idx = start_idx + SAMPLES_PER_COL
            temp_wave[start_idx:end_idx] = np.sum(base_waves_col * col_envelope, axis=0)

        # 전체 엔벨롭 처리
        temp_wave *= fade_envelope
        temp_wave /= (HEIGHT * 0.4)

        # 🔔 [핵심] 장면 스캔 시작점(맨 첫 부분)에 삐- 소리 더하기
        temp_wave[:BEEP_SAMPLES] += beep_wave

        # 최신 오디오 파형으로 실시간 버퍼 교체
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