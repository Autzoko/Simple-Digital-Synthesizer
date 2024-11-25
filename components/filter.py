# filter.py

import numpy as np

from scipy.signal import butter
from numba import jit

class Filter:
    """Filter class, for design and apply filter to signal"""
    def __init__(self, filter_type='lowpass', cutoff=1000.0, order=4, sample_rate=44100, bandwidth=None):
        """
        Initialize filter
        
        params:
        - filter_type (str): type of filter, selectables = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        - cutoff (float | tuple): cutoff frequency (Hz)
            - float: only for lowpass filter and highpass filter
            - tuple(float, float): only for bandpass filter and bandstop filter
        - order (int): order of filter
        - sample_rate (int): sample rate
        - bandwidth (float): only used in some of the filters
        """
        self.filter_type = filter_type.lower()
        self.cutoff = cutoff
        self.order = order
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        
        self.b = None
        self.a = None
        self.zi = None
        
        self._design_filter()
        
    def _design_filter(self):
        nyquist = 0.5 * self.sample_rate
        
        if self.filter_type in ['lowpass', 'highpass']:
            normal_cutoff = self.cutoff / nyquist
            self.b, self.a = butter(self.order, normal_cutoff, btype=self.filter_type, analog=False)
        elif self.filter_type in ['bandpass', 'bandstop']:
            if isinstance(self.cutoff, (list, tuple)) and len(self.cutoff) == 2:
                low = self.cutoff[0] / nyquist
                high = self.cutoff[1] / nyquist
                self.b, self.a = butter(self.order, [low, high], btype=self.filter_type, analog=False)
            else:
                raise ValueError("Cutoff must be a tuple for bandpass filter and bandstop filter")
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")
        
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
        
    def reset(self):
        """Clear filter states, reset"""
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
        
    def set_cutoff(self, cutoff: int | tuple):
        """Set cutoff frequency"""
        self.cutoff = cutoff
        self._design_filter()
        self.reset()
        
    def set_order(self, order: int):
        """Set filter order"""
        self.order = order
        self._design_filter()
        self.reset()
        
    def set_filter_type(self, filter_type: str):
        """Set filter type"""
        self.filter_type = filter_type
        self._design_filter()
        self.reset()
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply filter, filter the signal
        
        params:
        - signal (np.ndarray): input signal
        
        return:
        - np.ndarray: filtered signal
        """
        
    @staticmethod
    @jit(nopython=True)
    def _apply_filter(signal: np.ndarray, b, a, zi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filtered_signal = np.zeros_like(signal)
        zi = zi.copy()
        
        for i in range(len(signal)):
            filtered_signal[i], zi = Filter._filter_step(signal[i], b, a, zi)
            
        return filtered_signal, zi
        
    @staticmethod
    @jit(nopython=True)
    def _filter_step(x, b, a, zi):
        y = b[0] * x + zi[0]
        for i in range(1, len(b)):
            zi[i - 1] = b[i] * x + zi[i] - a[i] * y
        return y, zi
        
        
        