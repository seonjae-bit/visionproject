
import cv2
import numpy as np
import sounddevice as sd

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

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera open failed")

print("Press q to quit.")
while True:
    ok,frame=cap.read()
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
cap.release()
cv2.destroyAllWindows()
