import numpy as np
import matplotlib.pyplot as plt
from components.oscillator import Oscillator
from components.mixer import Mixer

def test_mixer():
    sample_rate = 44100
    duration = 0.01  # 10 ms
    t = np.linspace(0, duration, int(sample_rate * duration))[:-1]

    osc1 = Oscillator(waveform='sine', frequency=440, amplitude=1.0)
    osc2 = Oscillator(waveform='square', frequency=880, amplitude=0.5)
    osc3 = Oscillator(waveform='sawtooth', frequency=1320, amplitude=0.3)

    mixer = Mixer()
    mixer.add_oscillator(osc1, weight=1.0)
    mixer.add_oscillator(osc2, weight=0.8)
    mixer.add_oscillator(osc3, weight=0.5)

    mixed_signal = mixer.generate(duration, sample_rate)

    plt.figure(figsize=(10, 6))

    plt.subplot(4, 1, 1)
    plt.plot(t, osc1.generate(duration, sample_rate))
    plt.title("Oscillator 1 (Sine Wave)")

    plt.subplot(4, 1, 2)
    plt.plot(t, osc2.generate(duration, sample_rate))
    plt.title("Oscillator 2 (Square Wave)")

    plt.subplot(4, 1, 3)
    plt.plot(t, osc3.generate(duration, sample_rate))
    plt.title("Oscillator 3 (Sawtooth Wave)")

    plt.subplot(4, 1, 4)
    plt.plot(t, mixed_signal)
    plt.title("Mixed Signal")

    plt.tight_layout()
    plt.show()
