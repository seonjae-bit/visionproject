import cv2
import numpy as np
import sounddevice as sd
import threading
import time

WIDTH=8
HEIGHT=8
FRAME_TIME=0.5
FS=44100
MIN_FREQ=131.0
MAX_FREQ=2093.0
COL_TIME=FRAME_TIME/WIDTH
FADE_MS=2

samples=int(FS*COL_TIME)
t=np.arange(samples)/FS
freqs=np.linspace(MIN_FREQ,MAX_FREQ,HEIGHT)
fade=max(1,int(FS*FADE_MS/1000))
env=np.ones(samples,np.float32)
fi=np.linspace(0,1,fade)
fo=np.linspace(1,0,fade)
env[:fade]=fi
env[-fade:]=fo
waves=np.array([np.sin(2*np.pi*f*t) for f in freqs],dtype=np.float32)
waves*=env

# --- 카메라 지연 방지를 위한 백그라운드 스레드 클래스 ---
class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
        self.ok, self.frame = self.cap.read()
        self.running = True
        # 데몬 스레드로 설정하여 프로그램 종료 시 함께 종료되도록 함
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

# 카메라 스레드 시작
cam = CameraStream()

print("Press q to quit.")
while True:
    # 버퍼에 고인 프레임이 아니라, 스레드가 실시간으로 업데이트한 최신 프레임을 가져옴
    ok, frame = cam.read()
    if not ok: break
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    audio=np.empty(samples*WIDTH,dtype=np.float32)
    for c in range(WIDTH):
        amp=(small[:,c].astype(np.float32)/255.0)**2
        col=np.sum(waves*amp[:,None],axis=0)
        m=np.max(np.abs(col))
        if m>1e-6: col*=0.9/m
        audio[c*samples:(c+1)*samples]=col
    cv2.imshow("8x8",cv2.resize(small,(320,320),interpolation=cv2.INTER_NEAREST))
    sd.default.device = (None, 1)
    sd.play(audio,FS)
    if cv2.waitKey(1)&0xFF==ord('q'):
        sd.stop()
        break
    sd.wait()

cam.release()
cv2.destroyAllWindows()