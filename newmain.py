import cv2
import numpy as np
import sounddevice as sd
import threading
import time

print("\n--- [현재 연결된 오디오 장치 목록] ---")
print(sd.query_devices())
print("---------------------------------------\n")

# =====================================================================
# [볼륨 및 장치 설정]
AUDIO_DEVICE_ID = 1  # 이어폰 장치 번호
MASTER_VOLUME = 1.0  # ◀ 예전 방식처럼 합성 후 기본 볼륨 크기 (취향껏 0.5 ~ 2.0 사이 조절)
# =====================================================================

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FS = 44100

# 25Hz 배수 주파수 세팅
freqs = 1600 - np.arange(HEIGHT) * 25

# 한 열당 정확히 0.02초 = 882 샘플
COLUMN_SAMPLES = 882 
COLUMN_TIME = COLUMN_SAMPLES / FS

# 위상 반전용 홀수 반주기 필터 계산
half_cycles = (2 * freqs * COLUMN_SAMPLES) // FS
is_odd_half_cycle = (half_cycles % 2 == 1)

# 무음 공백 (5개 열)
MUTE_COLUMNS = 5
TOTAL_COLUMNS = WIDTH + MUTE_COLUMNS

# 글로벌 스레드 동기화 변수
shared_amp_matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix_lock = threading.Lock()

# 오디오 스레드 전용 상태 변수
current_column_idx = 0
samples_played_in_col = 0
per_freq_signs = np.ones(HEIGHT, dtype=np.float32)

last_amps = np.zeros(HEIGHT, dtype=np.float32)
current_amps = np.zeros(HEIGHT, dtype=np.float32)

# 이미지 갱신용 독립 플래그와 이미지 버퍼 (UI 먹통 방지 유지)
update_image_flag = threading.Event()
display_frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
frame_lock = threading.Lock()

# --- 라즈베리파이 카메라 버퍼 지연 방지 및 동기화 스레드 ---
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
        global shared_amp_matrix, display_frame
        while self.running:
            ok, frame = self.cap.read()
            if ok: 
                self.ok = ok
                self.frame = frame
            else:
                time.sleep(0.01)
                continue
                
            if update_image_flag.is_set():
                update_image_flag.clear()
                
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
                with frame_lock:
                    display_frame = small.copy()
                
                raw_amp = small.astype(np.float32) / 255.0
                raw_amp[raw_amp < 0.02] = 0.0  
                
                with matrix_lock:
                    shared_amp_matrix = raw_amp

# --- 샘플 정밀 오디오 콜백 함수 ---
def audio_callback(outdata, frames, time_info, status):
    global current_column_idx, samples_played_in_col, per_freq_signs 
    global shared_amp_matrix, last_amps, current_amps
    
    filled = 0
    audio = np.zeros(frames, dtype=np.float32)
    
    while filled < frames:
        rem_col_samples = COLUMN_SAMPLES - samples_played_in_col
        chunk_size = min(frames - filled, rem_col_samples)
        
        t = np.arange(samples_played_in_col, samples_played_in_col + chunk_size) / FS
        
        if samples_played_in_col == 0:
            if current_column_idx < WIDTH:
                with matrix_lock:
                    current_amps = shared_amp_matrix[:, current_column_idx]
            else:
                current_amps = np.zeros(HEIGHT, dtype=np.float32)

        idx_start = samples_played_in_col
        idx_end = samples_played_in_col + chunk_size
        weight = np.arange(idx_start, idx_end, dtype=np.float32) / COLUMN_SAMPLES
        amps_chunk = last_amps[:, None] * (1.0 - weight) + current_amps[:, None] * weight
        
        waves = np.sin(2 * np.pi * freqs[:, None] * t)
        
        # [복원된 핵심 소리 제어 로직] 
        # 무작정 더한 뒤 뻥튀기하는 게 아니라, 32개 주파수 총합을 HEIGHT(32)로 나누어 평균값을 취합니다.
        # 이 방식으로 합성하면 파형이 찢어지는 현상 없이 이전 코드처럼 깔끔한 믹싱 음을 얻을 수 있습니다.
        audio[filled:filled+chunk_size] = np.sum(
            (per_freq_signs[:, None] * amps_chunk) * waves, axis=0
        ) / HEIGHT
            
        samples_played_in_col += chunk_size
        filled += chunk_size
        
        if samples_played_in_col >= COLUMN_SAMPLES:
            samples_played_in_col = 0
            last_amps = current_amps.copy()
            current_column_idx += 1
            
            if current_column_idx < TOTAL_COLUMNS:
                per_freq_signs[is_odd_half_cycle] *= -1.0
            
            if current_column_idx >= TOTAL_COLUMNS:
                current_column_idx = 0
                per_freq_signs = np.ones(HEIGHT, dtype=np.float32)
                last_amps = np.zeros(HEIGHT, dtype=np.float32)
                update_image_flag.set()

    # 최종 출력단에 마스터 볼륨만 깔끔하게 적용
    outdata[:] = (audio * MASTER_VOLUME).reshape(-1, 1)

# 시스템 가동
update_image_flag.set()
cam = CameraStream()

try:
    stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback, device=AUDIO_DEVICE_ID)
    stream.start()
except Exception as e:
    print(f"\n[❌] 오디오 장치 {AUDIO_DEVICE_ID}번 개방 실패: {e}")
    cam.release()
    exit()

print("\n[✔] 시스템 기동 완료 (예전 소리 제어 + 선형 보간 하이브리드 모드)")
print(f"[*] 현재 마스터 볼륨 배율: {MASTER_VOLUME}x")
print("[*] Press 'q' to quit.\n")

try:
    while True:
        with frame_lock:
            local_frame = display_frame.copy()
            
        cv2.imshow("Selective Phase Inversion (Perfect Sync & Responsive)", 
                   cv2.resize(local_frame, (320, 320), interpolation=cv2.INTER_NEAREST))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()