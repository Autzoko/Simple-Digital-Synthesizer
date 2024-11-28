# envelope_generator.py

import numpy as np
from numba import jit

class EnvelopeGenerator:
    """Generate ADSR envelope"""
    def __init__(self, attack=0.01, decay=0.1, sustain_level=0.7, release=0.2, sample_rate=44100, curve='lin'):
        """
        Initialize EnvelopeGenerator
        
        params:
        - attack (float): attack time
        - decay (float): decay time
        - sustain_level: sustain level, range[0, 1]
        - release (float): release time
        - sample_rate (int): sample rate
        - curve (str): curve type, ['lin', 'exp', 'log']
        """
        
        self.sample_rate = sample_rate
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release
        self.curve = curve.lower()
        
        self.attack_samples = int(self.attack * sample_rate)
        self.decay_samples = int(self.decay * sample_rate)
        self.release_samples = int(self.release * sample_rate)
        
    def set_paramters(self, attack=None, decay=None, sustain_level=None, release=None, curve='None'):
        if attack is not None:
            self.attack = attack
            self.attack_samples = int(self.attack * self.sample_rate)
        if decay is not None:
            self.decay = decay
            self.decay_samples = int(self.decay * self.sample_rate)
        if sustain_level is not None:
            self.sustain_level = sustain_level
        if release is not None:
            self.release = release
            self.release_samples = int(self.release * self.sample_rate)
        if curve is not None:
            self.curve = curve.lower()
            
    @staticmethod
    @jit(nopython=True)
    def _generate_curve(start: float, end: float, num_samples: int, curve: str) -> np.ndarray:
        if curve == 'lin':
            return np.linspace(start, end, num_samples)
        elif curve == 'exp':
            return np.geomspace(max(start, 1e-6), max(end, 1e-6), num_samples) - 1e-6
        elif curve == 'log':
            return np.logspace(np.log10(max(start, 1e-6)), np.log10(max(end, 1e-6)), num_samples) - 1e-6
        else:
            raise ValueError(f'Unsupported curve type: {curve}')
        
    def generate(self, duration: float, trigger_on=True) -> np.ndarray:
        num_samples = int(duration * self.sample_rate)
        envelope = np.zeros(num_samples)
        
        if trigger_on:
            if self.attack_samples > 0:
                attack_signal = self._generate_curve(0, 1, self.attack_samples, self.curve)
                envelope[:self.attack_samples] = attack_signal
                
            if self.decay_samples > 0:
                start = self.attack_samples
                end = start + self.decay_samples
                decay_signal = self._generate_curve(1, self.sustain_level, self.decay_samples, self.curve)
                envelope[start : end] = decay_signal
                
            sustain_start = self.attack_samples + self.decay_samples
            sustain_end = num_samples
            envelope[sustain_start : sustain_end] = self.sustain_level
        
        else:
            if self.release_samples > 0:
                release_signal = self._generate_curve(self.sustain_level, 0, self.release_samples, self.curve)
                end = min(num_samples, self.release_samples)
                envelope[: end] = release_signal
        
        return envelope[:-1]