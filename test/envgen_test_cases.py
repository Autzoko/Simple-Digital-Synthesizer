import numpy as np
import matplotlib.pyplot as plt

from components.oscillator import Oscillator
from components.envelope_generator import EnvelopeGenerator

def test_envgen():
    sample_rate = 44100
    duration = 2.0
    
    osc = Oscillator(waveform='sine', frequency=440, amplitude=1.0)
    envgen = EnvelopeGenerator(
        attack=0.2,
        decay=0.5,
        sustain_level=0.5,
        release=0.3
    )
    
    t = np.linspace(0, duration, int(sample_rate * duration))[:-1]
    
    wave = osc.generate(duration, sample_rate)
    envelope = envgen.generate(duration)
    
    mod_wave = wave * envelope
    
    plt.subplot(3, 1, 1)
    plt.plot(t, wave, label="Original Wave")
    plt.title("Original Wave")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, envelope, label="Envelope (ADSR + Hold)", color='orange')
    plt.title("Envelope")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, mod_wave, label="Modulated Wave", color='green')
    plt.title("Wave with Envelope Applied")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()