# test_synthesizer.py

import numpy as np
import matplotlib.pyplot as plt
from asset.e_piano import EPiano

def test_epiano():
    sample_rate = 44100
    duration = 2.0  # 2 秒

    synth = EPiano(sample_rate=sample_rate)

    #synth.note_on(60, velocity=1.0)  # C4
    #synth.note_on(64, velocity=1.0)  # E4
    synth.note_on(67, velocity=1.0)  # G4

    audio = synth.generate_audio(duration=duration)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    plt.figure(figsize=(10, 4))
    plt.plot(t[:1000], audio[:1000])
    plt.title("Synthesized Audio")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    synth.note_off(60)
    synth.note_off(64)
    synth.note_off(67)

    from scipy.io.wavfile import write
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write('synth_output.wav', sample_rate, scaled)
