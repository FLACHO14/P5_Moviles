import numpy as np
from scipy.signal import lfilter, freqz

class OFDMChannel:
    def __init__(self, params):
        self.params = params
        self.los = params.channel_los
        self.num_taps = params.num_multipath
        self.velocity = params.doppler_speed_kmh
        self.fs = params.bandwidth_mhz * 1e6

    def get_impulse_response(self):
        delays = np.arange(1, self.num_taps + 1) * 2
        powers = np.exp(-0.5 * np.arange(self.num_taps))
        if self.los:
            powers[0] += 3
        powers = powers / np.sum(powers)
        taps = np.sqrt(powers) * (np.random.randn(self.num_taps) + 1j * np.random.randn(self.num_taps)) / np.sqrt(2)
        return taps, delays

    def apply_doppler(self, signal):
        if self.velocity <= 0:
            return signal
        fc = 2.1e9
        v_ms = self.velocity / 3.6
        fd = (v_ms * fc) / 3e8
        t = np.arange(len(signal)) / self.fs
        return signal * np.exp(1j * 2 * np.pi * fd * t)

    def apply_channel(self, tx_signal, snr_db):
        taps, delays = self.get_impulse_response()
        max_delay = int(np.max(delays))
        filtered = lfilter(taps, 1.0, tx_signal)
        rx_signal = filtered[max_delay:]
        rx_signal = self.apply_doppler(rx_signal)
        signal_power = np.mean(np.abs(rx_signal)**2)
        snr_lin = 10**(snr_db/10)
        noise_power = signal_power / snr_lin
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_signal)) + 1j * np.random.randn(len(rx_signal)))
        rx_signal += noise
        w, H_f = freqz(taps, worN=512, fs=self.fs)
        # Clasificación
        delay_spread = np.max(delays) / self.fs if np.max(delays) > 0 else 0
        coherence_bw = 1 / (5 * delay_spread) if delay_spread > 0 else np.inf
        if coherence_bw > self.params.bandwidth_mhz * 1e6:
            freq_sel = "Plano en frecuencia"
        else:
            freq_sel = "Selectivo en frecuencia"
        if self.velocity > 0:
            fd = (self.velocity/3.6 * 2.1e9) / 3e8
            coherence_time = 0.423 / fd
            symbol_duration = (self.params.fft_size + self.params.cp_length()) / self.fs
            if coherence_time > symbol_duration:
                time_sel = "Lento"
            else:
                time_sel = "Rápido"
        else:
            time_sel = "Estático"
        classification = f"{freq_sel} | {time_sel}"
        return rx_signal, taps, delays, (w, H_f), classification