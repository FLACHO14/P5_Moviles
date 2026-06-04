# ofdm_rx.py
import numpy as np
from scipy.interpolate import interp1d
from ofdm_utils import symbols_to_bits

class OFDMReceiver:
    def __init__(self, tx, params):
        self.tx = tx
        self.params = params
        self.fft_size = tx.fft_size
        self.cp_len = tx.cp_len
        self.data_indices = tx.data_indices
        self.pilot_indices = tx.pilot_indices
        self.pilot_values = tx.generate_pilots()
        # Convertir modulación string a número M (4, 16, 64)
        mod_str = params.modulation
        if mod_str == 'QPSK':
            self.M = 4
        elif mod_str == '16QAM':
            self.M = 16
        elif mod_str == '64QAM':
            self.M = 64
        else:
            self.M = 16

    def process(self, rx_signal):
        sym_len = self.fft_size + self.cp_len
        num_symbols = len(rx_signal) // sym_len
        if num_symbols > self.tx.num_ofdm_symbols:
            rx_signal = rx_signal[:self.tx.num_ofdm_symbols * sym_len]
            num_symbols = self.tx.num_ofdm_symbols
        rx_symbols = []
        for i in range(num_symbols):
            start = i * sym_len
            rx_symbols.append(rx_signal[start + self.cp_len : start + sym_len])
        fft_symbols = [np.fft.fft(sym) / np.sqrt(self.fft_size) for sym in rx_symbols]
        all_useful = sorted(self.data_indices + self.pilot_indices)
        all_positions = np.arange(len(all_useful))
        pilot_positions_in_useful = [list(all_useful).index(p) for p in self.pilot_indices]
        rx_pilots_all = []
        for sym in fft_symbols:
            rx_pilots = np.array([sym[idx] for idx in self.pilot_indices])
            h_pilot = rx_pilots / self.pilot_values
            f_interp = interp1d(pilot_positions_in_useful, h_pilot, kind='linear',
                                fill_value='extrapolate')
            h_all = f_interp(all_positions)
            h_full = np.zeros(self.fft_size, dtype=complex)
            for pos, idx in enumerate(all_useful):
                h_full[idx] = h_all[pos]
            rx_pilots_all.append(h_full)
        rx_data_symbols = []
        rx_symbols_raw = []
        for sym_idx, sym_fft in enumerate(fft_symbols):
            h = rx_pilots_all[sym_idx]
            raw = sym_fft[self.data_indices]
            eq = raw / (h[self.data_indices] + 1e-12)
            rx_symbols_raw.extend(raw)
            rx_data_symbols.extend(eq)
        # Ahora pasamos el número M (self.M) en lugar del string
        bits_rx = symbols_to_bits(np.array(rx_data_symbols), self.M)
        return bits_rx, np.array(rx_symbols_raw), np.array(rx_data_symbols), rx_pilots_all