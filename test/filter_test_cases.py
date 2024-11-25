import numpy as np
import matplotlib.pyplot as plt
from components.oscillator import Oscillator
from components.filter import Filter

def test_filter():
    sample_rate = 44100
    duration = 1.0  # 1 秒

    t = np.linspace(0, duration, int(sample_rate * duration))[:-1]
    osc = Oscillator(waveform='sawtooth', frequency=500, amplitude=1.0)
    signal = osc.generate(duration, sample_rate)

    filter_instance = Filter(filter_type='lowpass', cutoff=1000.0, order=4, sample_rate=sample_rate)

    chunk_size = 1024
    filtered_signal = np.zeros_like(signal)

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]

        if i >= int(0.5 * sample_rate):
            filter_instance.set_cutoff(2000.0)

        filtered_chunk = filter_instance.apply(chunk)
        filtered_signal[i:i+len(filtered_chunk)] = filtered_chunk

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title("Original Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_signal)
    plt.title("Filtered Signal with Realtime Cutoff Frequency Change")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
