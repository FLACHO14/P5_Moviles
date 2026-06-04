# ofdm_tx.py
import numpy as np
from ofdm_params import OFDMParams
from ofdm_utils import bits_to_symbols

class OFDMTransmitter:
    def __init__(self, params: OFDMParams):
        self.params = params
        self.fft_size = params.fft_size
        self.cp_len = params.cp_length()
        self.data_indices = params.data_indices
        self.pilot_indices = params.pilot_indices
        self.num_data = len(self.data_indices)
        self.num_pilots = len(self.pilot_indices)

    def generate_pilots(self):
        """Pilotos con valor conocido (BPSK: +1)"""
        return np.ones(self.num_pilots, dtype=complex)

    def create_ofdm_symbols(self, data_symbols):
        total_data = len(data_symbols)
        num_symbols = int(np.ceil(total_data / self.num_data))
        # Rellenar con ceros si es necesario
        if total_data < num_symbols * self.num_data:
            data_symbols = np.pad(data_symbols, (0, num_symbols * self.num_data - total_data),
                                  constant_values=0)
        data_2d = data_symbols.reshape(num_symbols, self.num_data)
        pilot_vals = self.generate_pilots()
        tx_signal = []
        self.subcarrier_grid = []  # guardar para visualización
        self.papr_values = []
        for sym_idx in range(num_symbols):
            # Inicializar vector de subportadoras
            subc = np.zeros(self.fft_size, dtype=complex)
            # Datos
            for i, idx in enumerate(self.data_indices):
                subc[idx] = data_2d[sym_idx, i]
            # Pilotos
            for i, idx in enumerate(self.pilot_indices):
                subc[idx] = pilot_vals[i]
            self.subcarrier_grid.append(subc.copy())
            # IFFT
            time_domain = np.fft.ifft(subc) * np.sqrt(self.fft_size)
            # Calcular PAPR
            power = np.abs(time_domain)**2
            avg_power = np.mean(power)
            papr = 10 * np.log10(np.max(power) / (avg_power + 1e-12))
            self.papr_values.append(papr)
            # Prefijo cíclico
            cp = time_domain[-self.cp_len:]
            tx_symbol = np.concatenate([cp, time_domain])
            tx_signal.append(tx_symbol)
        self.tx_signal = np.concatenate(tx_signal)
        self.num_ofdm_symbols = num_symbols
        return self.tx_signal

    def get_info(self):
        return {
            'fft_size': self.fft_size,
            'cp_len': self.cp_len,
            'num_data_subc': self.num_data,
            'num_pilot_subc': self.num_pilots,
            'num_total_subc': self.params.N_total_subcarriers,
            'num_useful_subc': self.params.N_useful_subcarriers,
            'num_ofdm_symbols': self.num_ofdm_symbols,
            'total_data_symbols': self.num_data * self.num_ofdm_symbols
        }