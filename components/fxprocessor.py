# fxprocessor.py

import numpy as np
from numba import jit

class FXProcessor:
    """Multi-purpose effctor"""
    def __init__(self, sample_rate=44100):
        """
        Initialize FXProcessor
        
        params:
        - sample_rate (int): sample rate
        """
        
        self.sample_rate = sample_rate
        self.effects = []
        
    def add_effect(self, effect, **kwargs):
        """
        Add effect into effect chain
        
        params:
        - effect (str): effect type, selectables ['reverb', 'delay', 'chorus', 'tremolo']
        - kwargs: effect args
        """
        
        if effect == 'reverb':
            self.effects.append(('reverb', kwargs))
        elif effect == 'delay':
            self.effects.append(('delay', kwargs))
        elif effect == 'chorus':
            self.effects.append(('chorus', kwargs))
        elif effect == 'tremolo':
            self.effects.append(('tremolo', kwargs))
        else:
            raise ValueError(f'Unsupported effect: {effect}')
        
    def clear_effects(self):
        """Reset effect chain"""
        self.effects = []
    
    @staticmethod
    @jit(nopython=True)
    def _apply_reverb(signal: np.ndarray, decay=0.5, room_size=0.5, sample_rate=44100) -> np.ndarray:
        delay_samples = int(room_size * sample_rate)
        output = np.zeros(len(signal))
        for i in range(len(signal)):
            output[i] = signal[i]
            if i >= delay_samples:
                output[i] += decay * output[i - delay_samples]
        return output
    
    @staticmethod
    @jit(nopython=True)
    def _apply_delay(signal: np.ndarray, delay_time=0.3, feed_back=0.5, mix=0.5, sample_rate=44100):
        delay_samples = int(delay_time * sample_rate)
        output = np.zeros(len(signal))
        for i in range((len(signal))):
            dry_signal = signal[i]
            wet_signal = feed_back * output[i - delay_samples] if i >= delay_samples else 0
        
        