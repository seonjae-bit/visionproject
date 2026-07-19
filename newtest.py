import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.6  # 한 장면의 주기 (0.6초마다 이미지 갱신)
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0

# 0.6초 분량의 버퍼 크기 계산
buffer_size = int(FS * FRAME_TIME)

# [유지] 일차함수(linspace) 매핑: 가로선의 위치 변화를 칼같이 인지하기 좋습니다.
freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)
t = np.arange(buffer_size) / FS
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 글로벌 오디오 버퍼 변수
current_audio_block = np.zeros(buffer_size, dtype=np.float32)
audio_lock = threading.Lock()

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

# --- 사운드 콜백 함수 (연속 재생용) ---
def audio_callback(outdata, frames, time_info, status):
    global current_audio_block
    with audio_lock:
        if len(current_audio_block) >= frames:
            outdata[:] = current_audio_block[:frames].reshape(-1, 1)
            current_audio_block = np.roll(current_audio_block, -frames)
            current_audio_block[-frames:] = 0.0
        else:
            outdata.fill(0)

# 카메라 스레드 및 오디오 스트림 기동
cam = CameraStream()
sd.default.device = (None, 1)
stream = sd.OutputStream(samplerate=FS, channels=1, callback=audio_callback)
stream.start()

print("Press q to quit.")

try:
    while True:
        start_time = time.time()
        
        ok, frame = cam.read()
        if not ok: break
        
        # 1. 전처리 및 리사이즈
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 2. [유지] 독립 노이즈 게이트 및 제곱 처리
        raw_amp = small.astype(np.float32) / 255.0
        raw_amp[raw_amp < 0.12] = 0.0  
        amp_matrix = raw_amp ** 2
        
        # 3. 가로축(시간축) 방향 부드러운 선형 보간
        x_old = np.linspace(0, 1, WIDTH)
        x_new = np.linspace(0, 1, buffer_size)
        
        smooth_amps = np.zeros((HEIGHT, buffer_size), dtype=np.float32)
        for r in range(HEIGHT):
            smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
        
        # 4. 오디오 신호 합성
        audio_matrix = base_waves * smooth_amps
        audio = np.sum(audio_matrix, axis=0)
        
        # 5. [복구] tanh 볼륨 감쇄 수식 적용 (선명하고 직관적인 반응성)
        audio = np.tanh(audio) * 0.95
        
        # 6. 오디오 스레드 버퍼에 데이터 전달
        with audio_lock:
            current_audio_block = audio.copy()
        
        # 화면 출력 및 루프 대기
        cv2.imshow("Sonification", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        elapsed = time.time() - start_time
        sleep_time = max(0.001, FRAME_TIME - elapsed)
        time.sleep(sleep_time)

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()