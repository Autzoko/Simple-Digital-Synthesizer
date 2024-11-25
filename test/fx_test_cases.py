import numpy as np
import matplotlib.pyplot as plt

from components.oscillator import Oscillator
from components.fxprocessor import FXProcessor

def test_fx():
    sample_rate = 44100
    duration = 2.0
    
    osc = Oscillator(waveform='sine', frequency=440, amplitude=1.0)
    t = np.linspace(0, duration, int(sample_rate * duration))[:-1]
    signal = osc.generate(duration, sample_rate)
    
    fx = FXProcessor(sample_rate=sample_rate)
    
    fx.add_effect('reverb', decay=0.5, room_size=0.4)
    fx.add_effect('delay', delay_time=0.2, feedback=0.5, mix=0.5)
    fx.add_effect('tremolo', depth=0.5, rate=5.0)
    fx.add_effect('chorus', depth=0.005, rate=1.0, mix=0.5)
    
    processed_signal = fx.process(signal)
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t[:1000], signal[:1000], label="Original Signal")
    plt.title("Original Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t[:1000], processed_signal[:1000], label="Processed Signal", color='orange')
    plt.title("Processed Signal with FX")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()