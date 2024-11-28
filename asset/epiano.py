# e_piano.py

import numpy as np
from components.oscillator import Oscillator
from components.mixer import Mixer
from components.filter import Filter
from components.envelope_generator import EnvelopeGenerator
from components.fxprocessor import FXProcessor
from components.modulator import Modulator

class EPianoNote:
    """Single note for E-Piano"""
    def __init__(self, duration: float, sample_rate: int):
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_samples = int(duration * sample_rate)
        
        self.osc1 = Oscillator(waveform='sine')
        self.osc2 = Oscillator(waveform='sine')
        self.osc3 = Oscillator(waveform='sine')
        self.osc4 = Oscillator(waveform='sawtooth')
        
        self.mixer = Mixer()
        self.mixer.add_oscillator(self.osc1)
        self.mixer.add_oscillator(self.osc2)
        self.mixer.add_oscillator(self.osc3)
        self.mixer.add_oscillator(self.osc4)
        
        self.filter = Filter(filter_type='lowpass', cutoff=1000.0, order=2, sample_rate=sample_rate)
        
        self.envelope = EnvelopeGenerator(attack=0.01, decay=0.15, sustain_level=0.8, release=0.2, sample_rate=sample_rate, curve='exp')
        
        self.fx = FXProcessor(self.sample_rate)
        self.fx.add_effect('tremolo', depth=0.5, rate=5.0, sample_rate=self.sample_rate)
        self.fx.add_effect('reverb', decay=0.3, room_size=0.5, sample_rate=self.sample_rate)
        
        
        