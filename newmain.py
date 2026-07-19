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
MASTER_VOLUME = 6.0  # 보간이 적용되어 소리가 정돈되므로, 취향껏 조절하세요.
# =====================================================================

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FS = 44100

# 25Hz 배수 주파수 세팅 (제로 크로싱 최소공배수 만족)
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

# 오디오-비디오 동기화 플래그
next_frame_trigger = threading.Event()
next_frame_trigger.set()

# 오디오 스레드 전용 상태 변수
current_column_idx = 0
samples_played_in_col = 0
per_freq_signs = np.ones(HEIGHT, dtype=np.float32)

# [보간 핵심 변수] 이전 열과 현재 열의 진폭 가로채기용
last_amps = np.zeros(HEIGHT, dtype=np.float32)
current_amps = np.zeros(HEIGHT, dtype=np.float32)

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
    global current_column_idx, samples_played_in_col, per_freq_signs 
    global shared_amp_matrix, last_amps, current_amps
    
    filled = 0
    audio = np.zeros(frames, dtype=np.float32)
    
    while filled < frames:
        rem_col_samples = COLUMN_SAMPLES - samples_played_in_col
        chunk_size = min(frames - filled, rem_col_samples)
        
        # 시간 t는 각 열마다 항상 0부터 출발
        t = np.arange(samples_played_in_col, samples_played_in_col + chunk_size) / FS
        
        # 1. 현재 연주해야 할 열의 목표 진폭 가져오기
        if samples_played_in_col == 0:
            if current_column_idx < WIDTH:
                with matrix_lock:
                    current_amps = shared_amp_matrix[:, current_column_idx]
            else:
                current_amps = np.zeros(HEIGHT, dtype=np.float32) # 무음 구간

        # 2. [진폭 보간 알고리즘] 
        # 한 열(882샘플)이 지나가는 동안 last_amps에서 current_amps로 선형 보간 진행
        # 전체 882개 중 현재 chunk 구간만큼만 가중치를 계산하여 추출합니다.
        idx_start = samples_played_in_col
        idx_end = samples_played_in_col + chunk_size
        
        # 각 샘플 위치별 보간 가중치 (0.0 ~ 1.0)
        weight = np.arange(idx_start, idx_end, dtype=np.float32) / COLUMN_SAMPLES
        
        # 진폭 보간 처리: 형상 불연속(뾰족한 꺾임) 방지
        # 행별 주파수 계산을 위해 브로드캐스팅(None) 적용
        amps_chunk = last_amps[:, None] * (1.0 - weight) + current_amps[:, None] * weight
        
        # 3. 위상 부호(per_freq_signs) 및 보간된 진폭을 적용하여 사인파 합성
        waves = np.sin(2 * np.pi * freqs[:, None] * t)
        audio[filled:filled+chunk_size] = np.sum(
            (per_freq_signs[:, None] * amps_chunk) * waves, axis=0
        )
            
        samples_played_in_col += chunk_size
        filled += chunk_size
        
        # 4. 정확히 한 열의 샘플(882개)을 다 채운 경계면 순간 처리
        if samples_played_in_col >= COLUMN_SAMPLES:
            samples_played_in_col = 0
            
            # 다음 열로 넘어가므로 현재 진폭이 "이전 진폭(last_amps)"이 됨
            last_amps = current_amps.copy()
            
            current_column_idx += 1
            
            # [위상 연속성 알고리즘] 홀수 번 반주기 돈 주파수만 위상 부호 토글
            if current_column_idx < TOTAL_COLUMNS:
                per_freq_signs[is_odd_half_cycle] *= -1.0
            
            # 한 장면 스캔(데이터+무음)이 완전히 끝난 경우
            if current_column_idx >= TOTAL_COLUMNS:
                current_column_idx = 0
                per_freq_signs = np.ones(HEIGHT, dtype=np.float32) # 부호 리셋
                last_amps = np.zeros(HEIGHT, dtype=np.float32)     # 보간 시작점 리셋
                next_frame_trigger.set()                           # 카메라 새 프레임 요청

    # 마스터 볼륨 부스팅 후 클리핑 방지 리미터 작동
    audio = audio * MASTER_VOLUME
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

print("\n[✔] 시스템 기동 완벽 완료 (위상 및 진폭 완전 보간 모드)")
print(f"[*] 현재 마스터 볼륨 배율: {MASTER_VOLUME}x")
print("[*] Press 'q' to quit.\n")

try:
    display_frame = np.zeros((32, 32), dtype=np.uint8)
    
    while True:
        # 오디오 스레드 신호 대기 (장면별 정밀 동기화)
        if next_frame_trigger.wait(timeout=1.0):
            next_frame_trigger.clear()
            
            ok, frame = cam.read()
            if not ok: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            display_frame = small.copy()
            
            # 볼륨 밸런스를 위한 리니어 밝기 매핑 및 노이즈 게이트 최소화
            raw_amp = small.astype(np.float32) / 255.0
            raw_amp[raw_amp < 0.02] = 0.0  
            amp_matrix = raw_amp
            
            with matrix_lock:
                shared_amp_matrix = amp_matrix.copy()

        # 화면 출력 (0.74초 주기로 딱딱 끊기며 싱크됨)
        cv2.imshow("Selective Phase Inversion (Perfect Anti-Click)", cv2.resize(display_frame, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()