import numpy as np
from numba import jit

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
        return self._mix_signals(duration, sample_rate, self.oscillators, self.weights)
        
    @staticmethod
    @jit(nopython=True)
    def _mix_signals(duration: float, sample_rate: int, oscillators, weights: float) -> np.ndarray:
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)[:-1]
        
        mixed_signal = np.zeros_like(t, dtype=np.float64)
        
        for i, osc in enumerate(oscillators):
            signal = osc.generate(duration, sample_rate)
            weight = weight[i]
            
            for j in range(num_samples):
                mixed_signal[j] += weight * signal[j]
                
        return mixed_signal
        