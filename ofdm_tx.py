import numpy as np

class OFDMTransmitter:
    def __init__(self, params):
        self.params = params
        self.fft_size = params.fft_size
        self.cp_len = params.cp_length()
        self.data_indices = params.data_indices
        self.pilot_indices = params.pilot_indices
        self.num_data = len(self.data_indices)
        self.num_pilots = len(self.pilot_indices)

    def generate_pilots(self):
        return np.ones(self.num_pilots, dtype=complex)

    def create_ofdm_symbols(self, data_symbols):
        total_data = len(data_symbols)
        num_symbols = int(np.ceil(total_data / self.num_data))
        if total_data < num_symbols * self.num_data:
            data_symbols = np.pad(data_symbols, (0, num_symbols * self.num_data - total_data), constant_values=0)
        data_2d = data_symbols.reshape(num_symbols, self.num_data)
        pilot_vals = self.generate_pilots()
        tx_signal = []
        self.subcarrier_grid = []
        self.papr_values = []
        for sym_idx in range(num_symbols):
            subc = np.zeros(self.fft_size, dtype=complex)
            for i, idx in enumerate(self.data_indices):
                subc[idx] = data_2d[sym_idx, i]
            for i, idx in enumerate(self.pilot_indices):
                subc[idx] = pilot_vals[i]
            self.subcarrier_grid.append(subc.copy())
            time_domain = np.fft.ifft(subc) * np.sqrt(self.fft_size)
            power = np.abs(time_domain)**2
            avg_power = np.mean(power)
            papr = 10 * np.log10(np.max(power) / (avg_power + 1e-12))
            self.papr_values.append(papr)
            cp = time_domain[-self.cp_len:]
            tx_symbol = np.concatenate([cp, time_domain])
            tx_signal.append(tx_symbol)
        self.tx_signal = np.concatenate(tx_signal)
        self.num_ofdm_symbols = num_symbols
        return self.tx_signal

    def get_first_subcarriers(self):
        """Devuelve el vector de subportadoras del primer símbolo"""
        if hasattr(self, 'subcarrier_grid') and len(self.subcarrier_grid) > 0:
            return self.subcarrier_grid[0]
        else:
            return None

    def get_spectrum(self, fs):
        N = len(self.tx_signal)
        f = np.fft.fftfreq(N, 1/fs)
        X = np.fft.fft(self.tx_signal)
        psd = 20 * np.log10(np.abs(X) + 1e-12)
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
        return f, psd

    def get_info(self):
        return {
            'fft_size': self.fft_size,
            'cp_len': self.cp_len,
            'num_data_subc': self.num_data,
            'num_pilot_subc': self.num_pilots,
            'num_total_subc': self.params.N_total_subcarriers,
            'num_useful_subc': self.params.N_useful_subcarriers,
            'num_ofdm_symbols': self.num_ofdm_symbols,
        }