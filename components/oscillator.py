# oscillator.py

import numpy as np
from numba import jit

class Oscillator:
    """Oscillator class, for generating various types of wave"""
    
    def __init__(self, waveform='sine', frequency=440.0, amplitude=1.0, phase=0.0, duty_cycle=0.5):
        """
        Initialize the oscillator
        
        params:
        - waveform (str): waveform type, selectable types: sine, sawtooth, square, triangle, noise, pulse.
        - frequency (float): wave frequency (Hz), default=440.0
        - amplitude (float): wave amplitude, default=1.0
        - phase (float): Radian, default=0.0
        - duty_cycle (float): only for pulse wave, range from 0.0 to 1.0
        """
        
        self.waveform = waveform.lower()
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.duty_cycle = duty_cycle
        
    def set_waveform(self, waveform: str):
        """Set type of waveform."""
        self.waveform = waveform.lower()
        
    def set_frequency(self, frequency: float):
        """Set wave frequency."""
        self.frequency = frequency
    
    def set_amplitude(self, amplitude: float):
        """Set wave amplitude."""
        self.amplitude = amplitude
        
    def set_phase(self, phase: float):
        """Set phase of wave."""
        self.phase = phase
        
    def set_duty_cycle(self, duty_cycle: float):
        """Set duty cycle of the pulse wave, range from 0.0 to 1.0."""
        if 0.0 <= duty_cycle <= 1.0:
            self.duty_cycle = duty_cycle
        else:
            raise ValueError("Duty cycle must be between 0.0 and 1.0")
        
    def generate(self, duration: float, sample_rate: int) -> np.ndarray:
        """
        Generate wave signal
        
        params:
        - duration (float): duration of the signal (s)
        - sample_rate (int): sample rate (Hz)
        
        return:
        - np.ndarray: generated wave signal
        """
        return Oscillator._generate(
            duration=duration,
            sample_rate=sample_rate,
            waveform=self.waveform,
            frequency=self.frequency,
            amplitude=self.amplitude,
            phase=self.phase,
            duty_cycle=self.duty_cycle
        )
    
    @staticmethod
    @jit(nopython=True)
    def _generate(duration: float, sample_rate: int, waveform: str, frequency: float, amplitude: float, phase: float, duty_cycle: float) -> np.ndarray:
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)[:-1]
        omega = 2 * np.pi * frequency
        phi = phase
        
        if waveform == 'sine':
            signal = amplitude * np.sin(omega * t + phi)
        elif waveform == 'square':
            signal = amplitude * np.sign(np.sin(omega * t + phi))
        elif waveform == 'sawtooth':
            signal = amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
        elif waveform == 'triangle':
            signal = amplitude * (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1)
        elif waveform == 'noise':
            np.random.seed(2024)
            signal = amplitude * np.random.uniform(-1, 1, size=num_samples)
        elif waveform == 'pulse':
            signal = amplitude * (np.mod(omega * t + phi, 2 * np.pi) < (2 * np.pi * duty_cycle)).astype(np.float64) * 2 - 1
        else:
            raise ValueError(f"Unsupported waveform type: {waveform}")
        
        return signal
    