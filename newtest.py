import cv2
import numpy as np
import sounddevice as sd
import threading
import time
import math

# --- [핵심] 수학적 최소공배수(LCM) 주기 계산 함수 ---
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

def find_matrix_lcm_period(frequencies, fs=44100):
    """
    각 주파수가 샘플링 레이트(FS) 내에서 '정수 개수의 사이클'을 돌 수 있도록
    모든 주파수의 공통 주기를 샘플 수 단위로 계산합니다.
    """
    current_lcm = frequencies[0]
    for f in frequencies[1:]:
        current_lcm = lcm(current_lcm, f)
    
    base_period_samples = fs // math.gcd(fs, current_lcm)
    
    actual_samples = base_period_samples
    # 32x32 해상도에서 답답하지 않은 속도(전체 0.74초 스캔)를 위해 가드를 1024로 최적화
    while actual_samples < 1024:
        actual_samples += base_period_samples
        
    return actual_samples

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FS = 44100
MIN_FREQ = 130  # LCM 계산을 위해 정수형 주파수 유지
MAX_FREQ = 2080 # 130의 배수

# 1. 해상도(HEIGHT)에 따른 정수형 주파수 매핑 (위쪽=고음, 아래쪽=저음)
freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT).astype(int)

# 2. 모든 주파수가 완벽한 위상으로 끝나는 '황금 샘플 수' 계산
COLUMN_SAMPLES = find_matrix_lcm_period(freqs, FS)
COLUMN_TIME = COLUMN_SAMPLES / FS

print(f"[*] 한 열(Column)당 수학적 완벽 재생 시간: {COLUMN_TIME:.4f} 초 ({COLUMN_SAMPLES} 샘플)")
print(f"[*] 한 장면(Frame) 스캔 소요 시간: {COLUMN_TIME * WIDTH:.4f} 초")
print(f"[*] 장면 간 무음 공백 시간: 0.1000 초")

# 3. 위상이 완벽하게 맞아떨어지는 기저 사인파 매트릭스 생성
t = np.arange(COLUMN_SAMPLES) / FS
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 가로축 1열 분량의 실시간 볼륨 매트릭스 변수 (콜백용)
current_col_amps = np.zeros(HEIGHT, dtype=np.float32)
amp_lock = threading.Lock()

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

# --- 오디오 콜백: 현재 열의 주파수를 공통 주기만큼 무한 반복 재생 ---
sample_index = 0

def audio_callback(outdata, frames, time_info, status):
    global sample_index, current_col_amps
    
    t_chunk = (np.arange(sample_index, sample_index + frames) % COLUMN_SAMPLES) / FS
    
    with amp_lock:
        audio_matrix = np.array([current_col_amps[r] * np.sin(2 * np.pi * freqs[r] * t_chunk) for r in range(HEIGHT)])
    
    audio = np.sum(audio_matrix, axis=0)
    audio = np.tanh(audio) * 0.95  # 절대 볼륨을 보존하는 리미터
    
    outdata[:] = audio.reshape(-1, 1)
    sample_index = (sample_index + frames) % COLUMN_SAMPLES

# 카메라 및 오디오 스트림 기동
cam = CameraStream()
sd.default.device = (None, 1)
stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback)
stream.start()

print("\n[✔] Perfect Mathematical Loop with Blank Gap Started.")
print("[*] Press 'q' to quit.\n")

try:
    while True:
        ok, frame = cam.read()
        if not ok: break
        
        # 1. 이미지 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 2. 독립 노이즈 게이트 및 밝기 제곱 처리
        raw_amp = small.astype(np.float32) / 255.0
        raw_amp[raw_amp < 0.12] = 0.0  # 암흑 노이즈 원천 차단
        amp_matrix = raw_amp ** 2
        
        # 3. [스캔 메커니즘] 가로축을 따라 황금 주기 단위로 소리 업데이트
        for c in range(WIDTH):
            col_start_time = time.time()
            
            with amp_lock:
                current_col_amps = amp_matrix[:, c]
            
            elapsed = time.time() - col_start_time
            sleep_time = max(0.0001, COLUMN_TIME - elapsed)
            time.sleep(sleep_time)
            
        # -----------------------------------------------------------------
        # [추가] 한 장면(Frame) 스캔 완료 후 인위적인 0.1초 무음 처리
        # -----------------------------------------------------------------
        with amp_lock:
            current_col_amps = np.zeros(HEIGHT, dtype=np.float32) # 사운드 즉시 뮤트
        
        time.sleep(0.1) # 0.1초 동안 숨을 고르며 완벽한 무음 공백 생성
        # -----------------------------------------------------------------
        
        # 화면 출력
        cv2.imshow("Mathematical Perfect Loop", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()