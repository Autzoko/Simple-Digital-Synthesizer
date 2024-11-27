# envelope_generator.py

import numpy as np
from numba import jit

class EnvelopeGenerator:
    """Envelope generator class, implement ADSR envelope generator"""
    
    def __init__(self, attack=0.01, decay=0.1, sustain_level=0.7, release=0.2, sample_rate=44100):
        """
        Initialize envelope generator
        
        params:
        - attack (float): attack time (s)
        - decay (float): decay time (s)
        - sustain_level (float): sustain level, range [0, 1]
        - release (float): release time (s)
        - sample_rate (int): sample rate
        """
        
        self.attack_time = attack
        self.decay_time = decay
        self.sustain_level = sustain_level
        self.release_time = release
        self.sample_rate = sample_rate
        
        self.attack_samples = max(1, int(self.attack_time * self.sample_rate))
        self.decay_samples = max(1, int(self.decay_time * self.sample_rate))
        self.release_samples = max(1, int(self.release_time * self.sample_rate))
        
        self.state = 'idle'
        self.position = 0
        self.triggered = False
        self.current_level = 0.0
        
    def trigger_on(self):
        """Start generating envelope"""
        self.state = 'attack'
        self.position = 0
        self.triggered = True
        
    def trigger_off(self):
        """Start release"""
        self.state = 'release'
        self.position = 0
        self.triggered = False
        self.release_samples = max(1, int(self.release_time * self.sample_rate))
        self.release_start_level = self.current_level
        
    @staticmethod
    @jit(nopython=True)
    def _generate_envelope(num_samples: int, envelope: np.ndarray, state: str, position: int, current_level: float,
                           triggered: bool, attack_samples: int, decay_samples: int, sustain_level: float, release_samples: int,
                           release_start_level: float) -> np.ndarray:
        new_state = state
        new_position = position
        new_current_level = current_level
        
        for i in range(num_samples):
            if new_state == 'idle':
                envelope[i] = 0.0
                new_current_level = 0.0
                if triggered:
                    new_state = 'attack'
                    new_position = 0
            elif new_state == 'attack':
                if new_position < attack_samples:
                    new_current_level = new_position / attack_samples
                    envelope[i] = new_current_level
                    new_position += 1
                else:
                    new_state = 'decay'
                    new_position = 0
                    new_current_level = 1.0
                    envelope[i] = new_current_level
            elif new_state == 'decay':
                if new_position < decay_samples:
                    decay_factor = (1.0 - sustain_level)
                    new_current_level = 1.0 - decay_factor * (new_position / decay_samples)
                    envelope[i] = new_current_level
                    new_position += 1
                else:
                    new_state = 'sustain'
                    new_current_level = sustain_level
                    envelope[i] = new_current_level
            elif new_state == 'sustain':
                new_current_level = sustain_level
                envelope[i] = new_current_level
                if not triggered:
                    new_state = 'release'
                    new_position = 0
                    release_start_level = new_current_level
            elif new_state == 'release':
                if new_position < release_samples:
                    new_current_level = release_start_level * (1.0 - (new_position / release_samples))
                    envelope[i] = new_current_level
                    new_position += 1
                else:
                    new_state = 'idle'
                    new_current_level = 0.0
                    envelope[i] = new_current_level
            else:
                envelope[i] = 0.0
                new_current_level = 0.0
                new_state = 'idle'
                
        EnvelopeGenerator._new_state = new_state
        EnvelopeGenerator._new_position = new_position
        EnvelopeGenerator._new_current_level = new_current_level
        
        return envelope
    
    def generate(self, num_samples: int) -> np.ndarray:
        envelope = np.zeros(num_samples, dtype=np.float32)
        
        envelope = self._generate_envelope(
            num_samples,
            envelope,
            self.state,
            self.position,
            self.current_level,
            self.triggered,
            self.attack_samples,
            self.decay_samples,
            self.sustain_level,
            self.release_samples,
            self.release_start_level
        )
        
        self.state = self._new_state
        self.position = self._new_position
        self.current_level = self._new_current_level
        
        return envelope
            
        
        