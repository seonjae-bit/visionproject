import numpy as np
import sounddevice as sd

# 설정
FS = 44100          # 샘플링 주파수
DURATION = 2.0      # 재생 시간(초)
FREQ = 440          # A4 음

# 시간축 생성
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)

# 사인파 생성
wave = 0.3 * np.sin(2 * np.pi * FREQ * t)

print("440Hz 소리를 재생합니다.")

sd.play(wave, FS)
sd.wait()

print("재생 완료")