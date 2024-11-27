# e_piano.py

import numpy as np
from components.oscillator import Oscillator
from components.mixer import Mixer, OscillatorSignal
from components.filter import Filter
from components.envelope_generator import EnvelopeGenerator
from components.fxprocessor import FXProcessor

class EPiano:
    """Basic E Piano Synthesizer"""
    def __init__(self, sample_rate=44100):
        """
        Initialize e-piano syntheizer
        
        params:
        - sample_rate (int): sample rate, default=44100
        """
        self.sample_rate = sample_rate
        self.notes = []
        self.oscillators = {}
        self.envelopes = {}
        self.mixer = Mixer()
        self.filter = Filter(filter_type='lowpass', cutoff=5000.0, order=2, sample_rate=sample_rate)
        self.fx_processor = FXProcessor(sample_rate=sample_rate)
        self.fx_processor.add_effect('reverb', decay=0.5, room_size=0.5)
        self.duration = 0.1
        
    def note_on(self, note: int, velocity=1.0):
        """
        Play notes
        
        params:
        - note (int): note code, MIDI Standard Note Code (0-127)
        - velocity (float): Note strength, range [0, 1]
        """
        frequency = self._midi_to_freq(note)
        osc = Oscillator(waveform='sine', frequency=frequency, amplitude=velocity)
        envelope = EnvelopeGenerator(
            attack=0.01,
            decay=0.1,
            sustain_level=0.7,
            release=0.2
        )
        self.oscillators[note] = osc
        self.envelopes[note] = envelope
        self.notes.append(note)
        
    def note_off(self, note: int):
        """
        Stop playing.
        
        params:
        - note (int): note code
        """
        if note in self.notes:
            self.notes.remove(note)
            del self.oscillators[note]
            del self.envelopes[note]
            
    def generate_audio(self, duration=None):
        if duration is None:
            duration = self.duration
            
        t = np.linspace(0, duration, int(self.sample_rate * duration))[:-1]
        self.mixer = Mixer()
        
        for note in self.notes:
            osc = self.oscillators[note]
            envelope = self.envelopes[note]
            wave = osc.generate(duration, self.sample_rate)
            env = envelope.generate(duration, trigger_on=True)
            signal = wave * env
            self.mixer.add_oscillator(OscillatorSignal(signal), weight=1.0)
            
        mixed_signal = self.mixer.generate(duration, self.sample_rate)
        
        filtered_signal = self.filter.apply(mixed_signal)
        
        processed_signal = self.fx_processor.process(filtered_signal)
        
        return processed_signal
    
    def _midi_to_freq(self, note: int):
        return 440.0 * 2 ** ((note - 69) / 12)