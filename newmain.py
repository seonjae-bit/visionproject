import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FS = 44100

# 25Hz의 배수로 주파수 정렬 (제로 크로싱 최소공배수 조건 충족)
freqs = 1600 - np.arange(HEIGHT) * 25

# 한 열당 정확히 0.02초 = 882 샘플
COLUMN_SAMPLES = 882 
COLUMN_TIME = COLUMN_SAMPLES / FS

# 각 주파수 성분이 한 열 재생 시간 동안 도는 반주기 개수 계산
half_cycles = (2 * freqs * COLUMN_SAMPLES) // FS
# 홀수 개의 반주기를 도는 행 필터 (다음 열 재생 시 위상 반전 대상)
is_odd_half_cycle = (half_cycles % 2 == 1)

# 장면 끝에 배치할 무음 공백 (5개 열 분량 = 약 0.1초 무음)
MUTE_COLUMNS = 5
TOTAL_COLUMNS = WIDTH + MUTE_COLUMNS

# 글로벌 공유 변수 및 스레드 락
shared_amp_matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix_lock = threading.Lock()

# 오디오 스레드 전용 상태 변수
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
        
        # 시간 t는 각 열마다 항상 0부터 리셋 출발
        t = np.arange(samples_played_in_col, samples_played_in_col + chunk_size) / FS
        
        if current_column_idx < WIDTH:
            with matrix_lock:
                amps = shared_amp_matrix[:, current_column_idx]
            
            # 이전 열 꼬리 기울기에 맞춰 개별 부호(per_freq_signs) 반영 후 합성
            waves = np.sin(2 * np.pi * freqs[:, None] * t)
            audio[filled:filled+chunk_size] = np.sum(
                (per_freq_signs * amps)[:, None] * waves, axis=0
            )
        else:
            audio[filled:filled+chunk_size] = 0.0
            
        samples_played_in_col += chunk_size
        filled += chunk_size
        
        # 정확히 한 열의 샘플을 다 채우는 경계면 순간
        if samples_played_in_col >= COLUMN_SAMPLES:
            samples_played_in_col = 0
            current_column_idx = (current_column_idx + 1) % TOTAL_COLUMNS
            
            if current_column_idx == 0:
                # 새 장면 시작할 때는 부호 초기화
                per_freq_signs = np.ones(HEIGHT, dtype=np.float32)
            else:
                # [유저 알고리즘 적용] 홀수 번 반주기 돈 주파수만 위상 부호 토글
                per_freq_signs[is_odd_half_cycle] *= -1.0

    audio = np.tanh(audio) * 0.95
    outdata[:] = audio.reshape(-1, 1)

# 카메라 기동
cam = CameraStream()

# [수정 핵심] 장치 설정을 무시당하지 않도록 OutputStream 생성 시 device=1을 강제로 주입!
stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback, device=1)
stream.start()

print("\n[✔] 1번 오디오 장치(이어폰) 연결 완료 및 시스템 시작.")
print(f"[*] 스캔 속도: 픽셀당 {COLUMN_TIME:.4f}초 | 전체 {COLUMN_TIME * WIDTH:.4f}초 + 무음 0.1초")
print("[*] Press 'q' to quit.\n")

try:
    while True:
        ok, frame = cam.read()
        if not ok: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        raw_amp = small.astype(np.float32) / 255.0
        raw_amp[raw_amp < 0.12] = 0.0  
        amp_matrix = raw_amp ** 2
        
        with matrix_lock:
            shared_amp_matrix = amp_matrix.copy()
            
        cv2.imshow("Selective Phase Inversion", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()