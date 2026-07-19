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
MASTER_VOLUME = 15.0 # ◀ 소리가 작으면 20.0~30.0까지 올리셔도 됩니다!
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

# [UI 먹통 해결] 이미지 갱신용 독립 플래그와 이미지 버퍼
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
            # 1. 무조건 카메라 버퍼는 최신으로 비워둠
            ok, frame = self.cap.read()
            if ok: 
                self.ok = ok
                self.frame = frame
            else:
                time.sleep(0.01)
                continue
                
            # 2. 오디오가 "새 장면 필요해!"라고 신호(update_image_flag)를 줄 때만 이미지 처리 수행
            if update_image_flag.is_set():
                update_image_flag.clear()
                
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                
                # UI 스레드가 가져가도록 디스플레이 버퍼에 복사 (0.74초마다 끊기며 바뀜)
                with frame_lock:
                    display_frame = small.copy()
                
                # 오디오 스레드로 행렬 주입
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
        audio[filled:filled+chunk_size] = np.sum(
            (per_freq_signs[:, None] * amps_chunk) * waves, axis=0
        )
            
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
                update_image_flag.set() # 카메라 스레드에게 새 컷 갱신 요청

    # [볼륨 강화] MASTER_VOLUME 곱하고, tanh 리미터 임계값을 대폭 올려 소리 밀도를 꽉 채움
    audio = audio * MASTER_VOLUME
    audio = np.clip(audio, -1.5, 1.5) # 소프트 클리핑 전 전압 확보
    audio = np.tanh(audio) * 0.98     # 스피커가 찢어지지 않는 선에서 최대 진폭 출력
    outdata[:] = audio.reshape(-1, 1)

# 시스템 가동
update_image_flag.set() # 최초 프레임 확보용
cam = CameraStream()

try:
    stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback, device=AUDIO_DEVICE_ID)
    stream.start()
except Exception as e:
    print(f"\n[❌] 오디오 장치 {AUDIO_DEVICE_ID}번 개방 실패: {e}")
    cam.release()
    exit()

print("\n[✔] 시스템 기동 완료 (UI 무한 반응 + 볼륨 극대화 모드)")
print(f"[*] 현재 마스터 볼륨 배율: {MASTER_VOLUME}x")
print("[*] Press 'q' to quit.\n")

try:
    # 메인 루프는 오디오를 기다리지 않고 초당 수십 번씩 미친 듯이 돕니다.
    # 덕분에 창을 드래그하거나 터미널을 움직여도 절대 멈추지 않고 매끄럽습니다.
    while True:
        with frame_lock:
            local_frame = display_frame.copy()
            
        cv2.imshow("Selective Phase Inversion (Perfect Sync & Responsive)", 
                   cv2.resize(local_frame, (320, 320), interpolation=cv2.INTER_NEAREST))
        
        # 10ms 단위로 윈도우 OS 이벤트를 상시 처리하여 창이 멈추는 현상 완벽 방지
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()