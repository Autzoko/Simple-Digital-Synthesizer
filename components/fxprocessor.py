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
    def _apply_delay(signal: np.ndarray, delay_time=0.3, feedback=0.5, mix=0.5, sample_rate=44100) -> np.ndarray:
        delay_samples = int(delay_time * sample_rate)
        output = np.zeros(len(signal))
        for i in range((len(signal))):
            dry_signal = signal[i]
            wet_signal = feedback * output[i - delay_samples] if i >= delay_samples else 0
            output[i] = (1.0 - mix) * dry_signal + mix * (dry_signal + wet_signal)
        return output
    
    @staticmethod
    @jit(nopython=True)
    def _apply_chorus(signal: np.ndarray, depth=0.01, rate=0.1, mix=0.5, sample_rate=44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        mod_signal = np.sin(2 * np.pi * rate * t) * depth * sample_rate
        output = np.zeros(num_samples)
        
        for i in range(num_samples):
            delay = int(mod_signal[i])
            wet_signal = signal[i - delay] if i >= delay else 0
            output[i] = (1.0 - mix) * signal[i] + mix * (signal[i] + wet_signal)
        return output
    
    @staticmethod
    @jit(nopython=True)
    def _apply_tremolo(signal: np.ndarray, depth=0.5, rate=5.0, sample_rate=44100) -> np.ndarray:
        num_samples = len(signal)
        t = np.arange(num_samples) / sample_rate
        mod_signal = 1 - depth * (0.5 * (1 + np.sin(2 * np.pi * rate * t)))
        return signal * mod_signal
    
    def process(self, signal: np.ndarray) -> np.ndarray:
        for effect, params in self.effects:
            if effect == 'reverb':
                signal = self._apply_reverb(signal, **params)
            elif effect == 'delay':
                signal = self._apply_delay(signal, **params)
            elif effect == 'chorus':
                signal = self._apply_chorus(signal, **params)
            elif effect == 'tremolo':
                signal = self._apply_tremolo(signal, **params)
        return signal
        
        