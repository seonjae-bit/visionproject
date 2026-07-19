import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FS = 44100

# [조건 반영] 0.02초(882 샘플)마다 모든 주파수가 반드시 제로 크로싱(값=0)을 지나도록 25Hz의 배수로 설정
# 최소공배수 개념을 충족하여 경계면에서 모든 파형의 값은 0이 됩니다.
freqs = 1600 - np.arange(HEIGHT) * 25

# 한 열당 정확히 0.02초 = 882 샘플
COLUMN_SAMPLES = 882 
COLUMN_TIME = COLUMN_SAMPLES / FS

# [유저 알고리즘 핵심] 각 주파수 성분이 한 열의 재생 시간 동안 몇 개의 '반주기'를 도는지 계산
# 반주기 개수 = 재생 시간 / (주기 / 2) = 2 * 주파수 * 재생시간
half_cycles = (2 * freqs * COLUMN_SAMPLES) // FS

# 반주기 개수가 '홀수'인 행들만 골라냅니다. (이 행들은 다음 열 재생 시 위상을 뒤집어야 함)
is_odd_half_cycle = (half_cycles % 2 == 1)

# 장면 끝에 배치할 무음 공백 (5개 열 분량 = 약 0.1초 무음)
MUTE_COLUMNS = 5
TOTAL_COLUMNS = WIDTH + MUTE_COLUMNS

# 글로벌 공유 변수 및 스레드 락
shared_amp_matrix = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
matrix_lock = threading.Lock()

# 오디오 스레드 전용 상태 변수 (각 행별 위상 부호를 저장하는 배열)
current_column_idx = 0
samples_played_in_col = 0
per_freq_signs = np.ones(HEIGHT, dtype=np.float32)  # 모든 행 초기 부호는 +1.0

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

# --- [유저 맞춤형] 샘플 정밀 오디오 콜백 함수 ---
def audio_callback(outdata, frames, time_info, status):
    global current_column_idx, samples_played_in_col, per_freq_signs, shared_amp_matrix
    
    filled = 0
    audio = np.zeros(frames, dtype=np.float32)
    
    # 사운드 카드 버퍼가 요구하는 샘플 수(frames)를 채울 때까지 루프
    while filled < frames:
        # 현재 열에서 남아있는 샘플 수
        rem_col_samples = COLUMN_SAMPLES - samples_played_in_col
        # 이번 루프에서 채울 샘플 크기 (열 경계면에서 정확히 잘라내기 위함)
        chunk_size = min(frames - filled, rem_col_samples)
        
        # [유저 요구사항] 시간 t는 항상 현재 열의 시작점(0)부터 계산됩니다.
        t = np.arange(samples_played_in_col, samples_played_in_col + chunk_size) / FS
        
        if current_column_idx < WIDTH:
            with matrix_lock:
                amps = shared_amp_matrix[:, current_column_idx]
            
            # 각 행 고유의 위상 부호(per_freq_signs)를 반영하여 사인파 합성
            # 기울기가 -로 끝난 성분은 -1.0이 곱해져서 다음 열 시작 시 - 기울기로 부드럽게 이어집니다.
            waves = np.sin(2 * np.pi * freqs[:, None] * t)
            audio[filled:filled+chunk_size] = np.sum(
                (per_freq_signs * amps)[:, None] * waves, axis=0
            )
        else:
            # 스캔 영역을 벗어난 프레임 끝자락은 완벽한 무음(0.1초 공백)
            audio[filled:filled+chunk_size] = 0.0
            
        samples_played_in_col += chunk_size
        filled += chunk_size
        
        # 정확히 한 열의 샘플(882샘플)을 다 채워서 다음 열로 넘어가는 순간
        if samples_played_in_col >= COLUMN_SAMPLES:
            samples_played_in_col = 0
            current_column_idx = (current_column_idx + 1) % TOTAL_COLUMNS
            
            if current_column_idx == 0:
                # 0.1초 무음이 끝나고 완전히 새로운 장면(Frame)이 시작될 때는 위상 부호 전체 리셋
                per_freq_signs = np.ones(HEIGHT, dtype=np.float32)
            else:
                # [유저 알고리즘 핵심 적용] 
                # 한 열 동안 홀수 개의 반주기를 돌았던 주파수 성분만 골라서 위상 부호를 반전시킵니다.
                per_freq_signs[is_odd_half_cycle] *= -1.0

    # 최종 출력 볼륨 안정화
    audio = np.tanh(audio) * 0.95
    outdata[:] = audio.reshape(-1, 1)

# 스트림 기동
cam = CameraStream()
sd.default.device = (None, 1)
stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback)
stream.start()

print("\n[✔] Selective Phase Inversion Scan System Started.")
print(f"[*] 스캔 속도: 픽셀당 {COLUMN_TIME:.4f}초 | 전체 {COLUMN_TIME * WIDTH:.4f}초 + 무음 0.1초")
print("[*] Press 'q' to quit.\n")

try:
    while True:
        ok, frame = cam.read()
        if not ok: break
        
        # 1. 이미지 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 2. 노이즈 게이트 및 밝기 제곱 처리
        raw_amp = small.astype(np.float32) / 255.0
        raw_amp[raw_amp < 0.12] = 0.0  
        amp_matrix = raw_amp ** 2
        
        # 3. 오디오 콜백 스레드가 실시간으로 참조하도록 전송
        with matrix_lock:
            shared_amp_matrix = amp_matrix.copy()
            
        # 메인 루프는 화면만 갱신
        cv2.imshow("Selective Phase Inversion", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows() #섹스