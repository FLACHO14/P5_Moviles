# ofdm_params.py
import numpy as np

DELTA_F = 15000.0          # 15 kHz
FC = 2.1e9                 # 2.1 GHz

class OFDMParams:
    def __init__(self):
        self.bandwidth_mhz = 5.0
        self.subcarrier_spacing = 15e3
        self.cp_type = 'normal'
        self.fft_size = 512
        self.modulation = '16QAM'
        self.channel_los = True
        self.num_multipath = 3
        self.doppler_speed_kmh = 30.0
        self.snr_db = 20.0
        self.pilot_spacing = 8
        self.guard_band_ratio = 0.10

    def compute_subcarriers(self):
        # Cálculo robusto sin np.log2 para evitar errores de tipo
        N_total = int(round(self.bandwidth_mhz * 1e6 / self.subcarrier_spacing))
        if N_total <= 0:
            N_total = 64
        N_useful = int(N_total * (1 - self.guard_band_ratio))
        # Potencia de 2 más cercana (mayor o igual)
        self.fft_size = 1
        while self.fft_size < N_total:
            self.fft_size <<= 1
        self.N_total_subcarriers = N_total
        self.N_useful_subcarriers = N_useful
        self.N_null = self.fft_size - N_useful
        pilot_indices_in_useful = list(range(0, N_useful, self.pilot_spacing))
        self.num_pilots = len(pilot_indices_in_useful)
        self.num_data = N_useful - self.num_pilots
        offset = (self.fft_size - N_useful) // 2
        self.data_indices = []
        self.pilot_indices = []
        for i in range(N_useful):
            idx = offset + i
            if i in pilot_indices_in_useful:
                self.pilot_indices.append(idx)
            else:
                self.data_indices.append(idx)
        return self.fft_size, self.N_total_subcarriers, self.N_useful_subcarriers

    def cp_length(self):
        if self.cp_type == 'normal':
            return self.fft_size // 8
        else:
            return self.fft_size // 4