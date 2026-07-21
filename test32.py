import cv2
import numpy as np
import sounddevice as sd
import threading
import time

WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.5
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0
COL_TIME = FRAME_TIME / WIDTH
FADE_MS = 1

samples = int(FS * COL_TIME)
t = np.arange(samples) / FS

freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)

fade = max(1, int(FS * FADE_MS / 1000))
env = np.ones(samples, np.float32)
fi = np.linspace(0, 1, fade)
fo = np.linspace(1, 0, fade)
env[:fade] = fi
env[-fade:] = fo
waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)
waves *= env

class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
            
        # =========================================================
        # 💡 [카메라 노출값 설정]
        # 0.25 (또는 1) = 수동 노출 모드 전환
        # EXPOSURE 값을 -6 ~ -10 정도로 낮추면 화면이 어두워집니다.
        # (웹캠 종류에 따라 값의 범위가 다를 수 있습니다)
        # =========================================================
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7) # 숫자가 작을수록/음수일수록 어두워짐

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

cam = CameraStream()

print("Scan Sonification Running... Press Ctrl+C to quit.")

# 오디오 장치 설정 (필요시 번호 변경)
sd.default.device = (None, 1)

try:
    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # [노출 보정 2단계] 소프트웨어 임계값(Threshold) 처리
        # 밝기 값이 25 미만인 자잘한 어두운 노이즈는 아예 0(검은색)으로 잘라냄
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        audio = np.empty(samples * WIDTH, dtype=np.float32)
        
        for c in range(WIDTH):
            amp = (small[:, c].astype(np.float32) / 255.0) ** 2
            col = np.sum(waves * amp[:, None], axis=0)
            m = np.max(np.abs(col))
            if m > 1e-6: 
                col *= 0.9 / m
            audio[c * samples:(c + 1) * samples] = col
            
        # 모니터 없이 실행할 때는 imshow를 끄고 sd.play를 수행합니다.
        cv2.imshow("Scan", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        sd.play(audio, FS)
        sd.wait()

except KeyboardInterrupt:
    print("\nStopping...")
    sd.stop()
    cam.release()
```[cite: 2]

---

### 💡 팁
만약 위 코드를 실행했는데도 화면이 생각만큼 어두워지지 않는다면, `self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)`에서 숫자 **`-7`** 부분을 **`-9`**나 **`-10`**처럼 더 작은 숫자로 바꾸거나, 코드 중반의 `gray[gray < 25] = 0`에서 **`25`** 숫자를 **`40`** 정도까지 올려보세요! 배경 노이즈가 칼같이 정리되면서 한층 맑고 또렷한 스캔 소리를 들으실 수 있을 겁니다.