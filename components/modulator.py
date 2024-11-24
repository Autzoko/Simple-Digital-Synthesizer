# modulator.py

import numpy as np
from numba import jit

class Modulator:
    """
    Modulator class, modulates the carrier signal.
    Supports amplitude modulation (AM) and frequency modulation (FM)
    """
    
    def __init__(self, modulation_type='AM', modulation_index=1.0):
        """
        Initialize modulator
        
        params:
        - modulation_type (str): type of modulation: AM, FM
        - modulation_index (float): depth of modulation
        """
        
        self.modulation_type = modulation_type.upper()
        self.modulation_index = modulation_index
        
    def set_modulation_type(self, modulation_type: str):
        """Set modulation type: AM, FM"""
        if modulation_type == 'AM' or modulation_type == 'FM':
            self.modulation_type = modulation_type.upper()
        raise ValueError(f'Unsupported modulation type: {modulation_type}')
    
    def set_modulation_index(self, modulation_index: float):
        self.modulation_index = modulation_index
        
    def modulate(self, carrier_signal: np.ndarray, modulator_signal: np.ndarray, carrier_frequency=None, t=None) -> np.ndarray:
        """
        Modulate the carrier signal
        
        params:
        - carrier_signal: array of carray signal
        - modulator_signal: array of modulator signal
        - carrier_frequency: frequency of carrier signal, only used in FM
        - t: time array, only used in FM
        
        return:
        - np.ndarray: modulated signal
        """
        
        return Modulator._modulate(
            modulation_type=self.modulation_type,
            modulation_index=self.modulation_index,
            carrier_signal=carrier_signal,
            modulator_signal=modulator_signal,
            carrier_frequency=carrier_frequency,
            t=t
        )
    
    @staticmethod
    @jit(nopython=True)
    def _modulate(modulation_type: str, modulation_index: float, carrier_signal: np.ndarray, modulator_signal: np.ndarray, carrier_frequency=None, t=None) -> np.ndarray:
        if modulation_type == 'AM':
            norm = modulator_signal / np.max(np.abs(modulator_signal))
            signal = (1 + modulation_index * norm) * carrier_signal
        elif modulation_type == 'FM':
            if carrier_frequency is None or t is None:
                raise ValueError("`carrier_frequency` and `t` parameters are required for FM modulation.")
            phase = 2 * np.pi * carrier_frequency * t + modulation_index * modulator_signal
            signal = np.cos(phase)
        else:
            raise ValueError(f'Unsupported modulation type: {modulation_type}')
        
        return signal
        
        
        