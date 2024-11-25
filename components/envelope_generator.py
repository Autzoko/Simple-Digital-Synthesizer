# envelope_generator.py

import numpy as np
from numba import jit

class EnvelopeGenerator:
    """Envelope generator class, implement ADSR envelope generator"""
    
    def __init__(self, attack=0.01, decay=0.1, sustain_level=0.7, release=0.2, sample_rate=44100):
        """
        Initialize envelope generator
        
        params:
        - attack (float): attack time duration (s)
        - decay (float): decay time duration (s)
        - sustain_level (float): sustain level, range[0, 1].
        - release (float): release time duration (s)
        - sample_rate (int): sample rate
        """
        self.sample_rate = sample_rate
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release
        
        self.attack_samples = int(self.attack * sample_rate)
        self.decay_samples = int(self.decay * sample_rate)
        self.release_samples = int(self.decay * sample_rate)
        
        self.current_stage = 'idle'
        
    def set_parameters(self, attack=None, decay=None, sustain_level=None, release=None):
        """Real time update ADSR parameters"""
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
    
    @staticmethod
    @jit(nopython=True)
    def _generate_attack(num_samples: int) -> np.ndarray:
        return np.linspace(0, 1, num_samples)
    
    @staticmethod
    @jit(nopython=True)
    def _generate_decay(num_samples: int, sustain_level: float) -> np.ndarray:
        return np.linspace(1, sustain_level, num_samples)
    
    @staticmethod
    @jit(nopython=True)
    def _generate_release(num_samples: int, sustain_level: float) -> np.ndarray:
        return np.linspace(sustain_level, 0, num_samples)
    
    def generate(self, duration: float, trigger_on=True) -> np.ndarray:
        """
        Generate ADSR envelope
        
        params:
        - duration (float): signal time duration
        - trigger_on (bool): is triggering on attack stage
        
        return:
        - np.ndarray: envelope signal
        """
        num_samples = int(duration * self.sample_rate)
        envelope = np.zeros(num_samples)
        
        if trigger_on:
            if self.attack_samples > 0:
                attack_signal = self._generate_attack(self.attack_samples)
                envelope[:self.attack_samples] = attack_signal
            
            if self.decay_samples > 0:
                start = self.attack_samples
                end = start + self.decay_samples
                decay_signal = self._generate_decay(self.decay_samples, self.sustain_level)
                envelope[start: end] = decay_signal
            
            sustain_start = self.attack_samples + self.decay_samples
            sustain_end = num_samples
            envelope[sustain_start: num_samples] = self.sustain_level
        else:
            if self.release_samples > 0:
                release_signal = self._generate_release(self.release_samples, self.sustain_level)
                release_start = min(num_samples, self.release_samples)
                envelope[:release_start] = release_signal
                
        return envelope[:-1]