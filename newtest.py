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

# 주파수 배열 (지수함수 매핑) 및 기저 사인파 미리 생성
freqs = np.geomspace(MAX_FREQ, MIN_FREQ, HEIGHT)
t = np.arange(buffer_size) / FS
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 글로벌 오디오 버퍼 변수 (스레드 간 안전한 데이터 교환을 위해 사용)
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

# --- 사운드 카드가 오디오를 쉬지 않고 끊임없이 요청하는 콜백 함수 ---
def audio_callback(outdata, frames, time_info, status):
    global current_audio_block
    if status:
        print(status)
    
    # 사운드카드가 요청하는 프레임 수(대개 1024내외)만큼 글로벌 버퍼에서 잘라서 던져줌
    # 이 메커니즘 덕분에 0.6초 경계선에서 소리가 끊기지 않고 아날로그처럼 매끄럽게 이어집니다.
    with audio_lock:
        # 이 콜백은 사운드 디바이스가 자체 스레드로 매우 빠르게 반복 호출함
        # 필요한 만큼 데이터를 outdata에 복사
        if len(current_audio_block) >= frames:
            outdata[:] = current_audio_block[:frames].reshape(-1, 1)
            # 사용한 데이터는 밀어내고 뒤쪽 데이터를 앞으로 땡김 (롤링 버퍼)
            current_audio_block = np.roll(current_audio_block, -frames)
            current_audio_block[-frames:] = 0.0 # 빈자리는 0으로 채움
        else:
            outdata.fill(0)

# 카메라 스레드 기동
cam = CameraStream()

# 오디오 무한 스트림 가동 (사운드 장치를 항상 열어둠)
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
        
        # 2. 독립 노이즈 게이트 및 제곱 처리
        raw_amp = small.astype(np.float32) / 255.0
        raw_amp[raw_amp < 0.12] = 0.0  
        amp_matrix = raw_amp ** 2
        
        # 3. 가로축(시간축) 방향 부드러운 선형 보간
        x_old = np.linspace(0, 1, WIDTH)
        x_new = np.linspace(0, 1, buffer_size)
        
        smooth_amps = np.zeros((HEIGHT, buffer_size), dtype=np.float32)
        for r in range(HEIGHT):
            smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
        
        # 4. 오디오 신호 합성 및 소프트 리미터
        audio_matrix = base_waves * smooth_amps
        audio = np.sum(audio_matrix, axis=0)
        audio = np.tanh(audio) * 0.95
        
        # 5. [핵심] 새로 계산된 0.6초 소리를 오디오 스레드 버퍼에 끊김 없이 교체
        with audio_lock:
            current_audio_block = audio.copy()
        
        # 화면 출력
        cv2.imshow("Sonification", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 0.6초 주기 타이밍 정확히 제어 (정확히 0.6초마다 이미지 갱신 루프 동기화)
        elapsed = time.time() - start_time
        sleep_time = max(0.001, FRAME_TIME - elapsed)
        time.sleep(sleep_time)

finally:
    stream.stop()
    stream.close()
    cam.release()
    cv2.destroyAllWindows()