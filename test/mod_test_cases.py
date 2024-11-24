# mod_test_cases.py

from components.oscillator import Oscillator
from components.modulator import Modulator

import numpy as np
import matplotlib.pyplot as plt

def test_mod(mod: str, sample_rate=44100, duration=0.01):
    t = np.linspace(0, duration, int(sample_rate * duration))[:-1]
    
    if mod == 'AM':
        carrier_osc = Oscillator(waveform='sine', frequency=1000.0, amplitude=1.0)
        modulator_osc = Oscillator(waveform='sine', frequency=100.0, amplitude=1.0)
    
        carrier_signal = carrier_osc.generate(duration, sample_rate)
        modulator_signal = modulator_osc.generate(duration, sample_rate)
    
        modulator = Modulator(modulation_type='AM', modulation_index=0.8)
    
        modulated_signal = modulator.modulate(carrier_signal, modulator_signal)
    elif mod == 'FM':
        modulator_osc = Oscillator(waveform='sine', frequency=100.0, amplitude=1.0)
        modulator_signal = modulator_osc.generate(duration, sample_rate)
        
        modulator = Modulator(modulation_type='FM', modulation_index=5.0)
        
        carrier_freq = 1000.0
        
        modulated_signal = modulator.modulate(carrier_signal=0, modulator_signal=modulator_signal, carrier_frequency=carrier_freq, t=t)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, modulated_signal)
    plt.title(f'{mod} Modulation Test')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
