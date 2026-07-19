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
MASTER_VOLUME = 8.0  # ◀ 소리가 아직도 작다면 이 값을 15.0, 20.0 등으로 더 키우세요!
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

# 위상 반전용 홀수 반주기 필터
half_cycles = (2 * freqs * COLUMN_SAMPLES) // FS
is_odd_half_cycle = (half_cycles % 2 == 1)

# 무음 공백 (5개 열)
MUTE_COLUMNS = 5
TOTAL_COLUMNS = WIDTH + MUTE_COLUMNS

# 글로벌 스레드 동기화 변수
shared_amp_matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix_lock = threading.Lock()

# 오디오-비디오 동기화 플래그
next_frame_trigger = threading.Event()
next_frame_trigger.set()

current_column_idx = 0
samples_played_in_col = 0
per_freq_signs = np.ones(HEIGHT, dtype=np.float32)

# --- 라즈베리파이 카메라 버퍼 지연 방지 스레드 ---
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

# --- 샘플 정밀 오디오 콜백 함수 ---
def audio_callback(outdata, frames, time_info, status):
    global current_column_idx, samples_played_in_col, per_freq_signs, shared_amp_matrix
    
    filled = 0
    audio = np.zeros(frames, dtype=np.float32)
    
    while filled < frames:
        rem_col_samples = COLUMN_SAMPLES - samples_played_in_col
        chunk_size = min(frames - filled, rem_col_samples)
        
        t = np.arange(samples_played_in_col, samples_played_in_col + chunk_size) / FS
        
        if current_column_idx < WIDTH:
            with matrix_lock:
                amps = shared_amp_matrix[:, current_column_idx]
            
            waves = np.sin(2 * np.pi * freqs[:, None] * t)
            audio[filled:filled+chunk_size] = np.sum(
                (per_freq_signs * amps)[:, None] * waves, axis=0
            )
        else:
            audio[filled:filled+chunk_size] = 0.0
            
        samples_played_in_col += chunk_size
        filled += chunk_size
        
        if samples_played_in_col >= COLUMN_SAMPLES:
            samples_played_in_col = 0
            current_column_idx += 1
            
            if current_column_idx < TOTAL_COLUMNS:
                per_freq_signs[is_odd_half_cycle] *= -1.0
            
            if current_column_idx >= TOTAL_COLUMNS:
                current_column_idx = 0
                per_freq_signs = np.ones(HEIGHT, dtype=np.float32)
                next_frame_trigger.set()

    # [볼륨 핵심] 유저가 설정한 MASTER_VOLUME 배수만큼 통째로 부스팅
    audio = audio * MASTER_VOLUME
    
    # 신호가 깨지지 않도록 부드러운 리미터(tanh)를 적용하되 전압 한계치까지 꽉 채우기
    audio = np.tanh(audio) * 0.98
    outdata[:] = audio.reshape(-1, 1)

# 시스템 가동
cam = CameraStream()

try:
    stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback, device=AUDIO_DEVICE_ID)
    stream.start()
except Exception as e:
    print(f"\n[❌] 오디오 장치 {AUDIO_DEVICE_ID}번 개방 실패: {e}")
    cam.release()
    exit()

print("\n[✔] 시스템 기동 완료 (볼륨 부스팅 모드)")
print(f"[*] 현재 마스터 볼륨 배율: {MASTER_VOLUME}x")
print("[*] Press 'q' to quit.\n")

try:
    display_frame = np.zeros((32, 32), dtype=np.uint8)
    
    while True:
        if next_frame_trigger.wait(timeout=1.0):
            next_frame_trigger.clear()
            
            ok, frame = cam.read()
            if not ok: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            display_frame = small.copy()
            
            # 볼륨 확보를 위해 제곱(**2) 계산을 빼고 선형(Linear) 밝기로 매핑
            # 어두운 픽셀도 소리가 확실히 들리도록 노이즈 게이트를 0.02로 대폭 낮췄습니다.
            raw_amp = small.astype(np.float32) / 255.0
            raw_amp[raw_amp < 0.02] = 0.0  
            amp_matrix = raw_amp  # 제곱 제거하여 중간 톤 볼륨 대폭 상승
            
            with matrix_lock:
                shared_amp_matrix = amp_matrix.copy()

        cv2.imshow("Selective Phase Inversion (Synced)", cv2.resize(display_frame, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()