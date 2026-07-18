
import cv2
import numpy as np
import sounddevice as sd
import queue

WIDTH,HEIGHT=8,8
FRAME_TIME=0.5
FS=44100
MINF,MAXF=131.0,2093.0
COL_TIME=FRAME_TIME/WIDTH
FADE_MS=2

samples=int(FS*COL_TIME)
t=np.arange(samples)/FS
freqs=np.linspace(MINF,MAXF,HEIGHT)
fade=max(1,int(FS*FADE_MS/1000))
env=np.ones(samples,np.float32)
env[:fade]=np.linspace(0,1,fade)
env[-fade:]=np.linspace(1,0,fade)
waves=np.array([np.sin(2*np.pi*f*t) for f in freqs],dtype=np.float32)*env

q=queue.Queue(maxsize=8)
def callback(outdata, frames, time, status):
    try:
        data=q.get_nowait()
    except queue.Empty:
        data=np.zeros(frames,dtype=np.float32)
    if len(data)!=frames:
        if len(data)>frames:
            data=data[:frames]
        else:
            data=np.pad(data,(0,frames-len(data)))
    outdata[:,0]=data

cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise RuntimeError("Camera open failed")

stream=sd.OutputStream(samplerate=FS,channels=1,blocksize=samples,dtype='float32',callback=callback)
stream.start()
print("Press q to quit.")
try:
    while True:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        small=cv2.resize(gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
        for c in range(WIDTH):
            amp=(small[:,c].astype(np.float32)/255.0)**2
            col=np.sum(waves*amp[:,None],axis=0)
            m=np.max(np.abs(col))
            if m>1e-6: col*=0.9/m
            try:q.put_nowait(col.astype(np.float32))
            except queue.Full:
                pass
        cv2.imshow("8x8",cv2.resize(small,(320,320),interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
finally:
    stream.stop();stream.close()
    cap.release()
    cv2.destroyAllWindows()
