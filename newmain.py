import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# =====================================================================
# [중요] 내 PC의 오디오 장치 번호 찾기
# 코드를 실행하면 터미널 콘솔창에 장치 목록이 출력됩니다.
# 이어폰/헤드폰의 명칭 옆에 있는 '번호'를 아래 AUDIO_DEVICE_ID에 적어주세요.
# =====================================================================
print("\n--- [현재 연결된 오디오 장치 목록] ---")
print(sd.query_devices())
print("---------------------------------------\n")

AUDIO_DEVICE_ID = 1  # ◀ 콘솔창을 보고 이어폰 번호가 1번이 아니라면 이 숫자를 바꾸세요!

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FS = 44100

# 25Hz 배수 주파수 세팅 (제로 크로싱 최소공배수 만족)
freqs = 1600 - np.arange(HEIGHT) * 25

# 한 열당 정확히 0.02초 = 882 샘플
COLUMN_SAMPLES = 882 
COLUMN_TIME = COLUMN_SAMPLES / FS

# 홀수 개의 반주기를 도는 행 필터 계산 (위상 반전용)
half_cycles = (2 * freqs * COLUMN_SAMPLES) // FS
is_odd_half_cycle = (half_cycles % 2 == 1)

# 장면 끝에 배치할 무음 공백 (5개 열 = 약 0.1초)
MUTE_COLUMNS = 5
TOTAL_COLUMNS = WIDTH + MUTE_COLUMNS

# 글로벌 스레드 동기화 변수
shared_amp_matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix_lock = threading.Lock()

# [핵심] 오디오와 비디오의 완벽한 1:1 결합을 위한 싱크 플래그
# 오디오가 한 프레임 스캔을 끝내면 이 플래그를 True로 만듭니다.
next_frame_trigger = threading.Event()
next_frame_trigger.set()  # 첫 프레임 시작을 위해 켜둠

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
        
        # 시간 t는 항상 0부터 리셋 출발
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
            
            # [유저 알고리즘 적용] 홀수 번 반주기 돈 주파수만 위상 부호 토글
            if current_column_idx < TOTAL_COLUMNS:
                per_freq_signs[is_odd_half_cycle] *= -1.0
            
            # [수정의 핵심] 한 장면에 해당하는 가로 스캔 + 무음이 통째로 끝난 순간!
            if current_column_idx >= TOTAL_COLUMNS:
                current_column_idx = 0
                per_freq_signs = np.ones(HEIGHT, dtype=np.float32) # 부호 리셋
                next_frame_trigger.set()  # 메인 스레드에게 "다음 카메라 화면 가져와!" 라고 신호 보냄

    # 소리 증폭 및 리미터 (소리가 너무 작게 묻히지 않도록 기본 마스터 볼륨 업)
    audio = np.tanh(audio * 2.0) * 0.95
    outdata[:] = audio.reshape(-1, 1)

# 시스템 가동
cam = CameraStream()

try:
    stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback, device=AUDIO_DEVICE_ID)
    stream.start()
except Exception as e:
    print(f"\n[❌] 오디오 장치 {AUDIO_DEVICE_ID}번을 열 수 없습니다. 에러내용: {e}")
    print("[*] 콘솔창에 나온 장치 목록을 보고 유효한 다른 출력 장치 번호로 변경해 주세요.")
    cam.release()
    exit()

print("\n[✔] 시스템이 정상 기동되었습니다.")
print(f"[*] 스캔 주기: 약 {COLUMN_TIME * TOTAL_COLUMNS:.4f}초 마다 화면이 툭툭 끊기며 갱신됩니다.")
print("[*] Press 'q' to quit.\n")

try:
    display_frame = np.zeros((32, 32), dtype=np.uint8)
    
    while True:
        # [싱크 핵심] 오디오 스레드에서 한 장면 재생이 끝날 때까지 메인 루프를 대기시킵니다.
        # 이 덕분에 카메라 화면이 오디오 재생 시간에 맞춰 툭, 툭 끊기며 작동합니다!
        if next_frame_trigger.wait(timeout=1.0):
            next_frame_trigger.clear()
            
            ok, frame = cam.read()
            if not ok: break
            
            # 1. 이미지 전처리
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            display_frame = small.copy()
            
            # 2. 노이즈 게이트 대폭 완화 (소리가 안 나는 현상 방지하기 위해 기준치를 0.05로 낮춤)
            raw_amp = small.astype(np.float32) / 255.0
            raw_amp[raw_amp < 0.05] = 0.0  
            amp_matrix = raw_amp ** 2
            
            # 3. 오디오 스레드에 이미지 데이터 주입
            with matrix_lock:
                shared_amp_matrix = amp_matrix.copy()

        # 화면 출력 (스캔 주기인 약 0.74초마다 끊기면서 업데이트 됨)
        cv2.imshow("Selective Phase Inversion (Synced)", cv2.resize(display_frame, (320, 320), interpolation=cv2.INTER_NEAREST))
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()