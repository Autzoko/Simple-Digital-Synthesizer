# mixer.py

import numpy as np
from numba import jit

@jit(nopython=True)
def _mix_signals(signals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    num_samples = signals.shape[1]
    mixed_signal = np.zeros(num_samples, dtype=np.float64)
    
    for i in range(signals.shape[0]):
        for j in range(num_samples):
            mixed_signal[j] += weights[i] * signals[i, j]
    
    return mixed_signal

class Mixer:
    """
    Mixer class, for mixing outputs from various oscillators
    """
    def __init__(self):
        self.oscillators = []
        self.weights = []
        
    def add_oscillator(self, oscillator, weight=1.0):
        """
        Add new oscillator into the mixer
        
        params:
        - oscillator (Oscillator): a oscillator instance
        - weight: the weight of this oscillator instance, default=1.0
        """
        self.oscillators.append(oscillator)
        self.weights.append(weight)
        
    def set_weight(self, index: int, weight: float):
        """
        Set the weight of an added oscillator
        
        params:
        - index: index of the oscillator
        - weight: new weight of the oscillator
        """
        if index < 0 or index >= len(self.weights):
            raise IndexError("Invalid oscillator index")
        self.weights[index] = weight
        
    def generate(self, duration: float, sample_rate: int) -> np.ndarray:
        """
        Generate mixed signal
        
        params:
        - duration (float): time lapse of the signal (s)
        - sample_rate (int): sample rate
        
        return:
        - np.ndarray: mixed signal
        """
        
        if len(self.oscillators) == 0:
            raise ValueError("No oscillator added")
        
        signals = np.array([osc.generate(duration, sample_rate) for osc in self.oscillators])
        weights = np.array(self.weights, dtype=np.float64)
        
        return _mix_signals(signals, weights)
    
    
class OscillatorSignal:
    def __init__(self, signal):
        self.signal = signal
        
    def generate(self, duration, sample_rate):
        return self.signal
        