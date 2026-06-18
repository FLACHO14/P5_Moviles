import numpy as np
from ofdm_params import FC, DELTA_F

def multipath_rayleigh_channel(L, seed=None):
    rng = np.random.default_rng(seed)
    h = (rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)) / np.sqrt(2)
    return h / np.sqrt(np.sum(np.abs(h)**2))

def multipath_rician_channel(L, K_factor=3, seed=None):
    rng = np.random.default_rng(seed)
    nlos = (rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)) / np.sqrt(2)
    h = np.sqrt(1/(K_factor+1)) * nlos
    h[0] += np.sqrt(K_factor/(K_factor+1))
    return h / np.sqrt(np.sum(np.abs(h)**2))

def get_channel_profile(profile_name, taps_L=None, seed=None):
    rng = np.random.default_rng(seed)
    fs = 1.92e6
    name = profile_name.lower()
    if 'ideal' in name:
        return np.array([1.0+0j])
    elif 'rayleigh' in name:
        return multipath_rayleigh_channel(taps_L, seed)
    elif 'rician' in name:
        return multipath_rician_channel(taps_L, seed=seed)
    elif 'epa' in name or 'suburbano' in name:
        delays_us = np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24])
        powers_dB = np.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    elif 'eva' in name or 'urbano' in name:
        delays_us = np.array([0.0, 0.03, 0.15, 0.31, 0.37, 0.71, 1.09, 1.73, 2.51])
        powers_dB = np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
    elif 'etu' in name or 'rural' in name:
        delays_us = np.array([0.0, 0.05, 0.12, 0.20, 0.28, 0.36, 0.45, 0.55, 0.70])
        powers_dB = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0])
    else:
        raise ValueError(f"Perfil desconocido: {profile_name}")
    delays_samples = np.round(delays_us * 1e-6 * fs).astype(int)
    max_delay = np.max(delays_samples)
    h = np.zeros(max_delay + 1, dtype=complex)
    powers_lin = 10 ** (powers_dB / 10)
    for d, p in zip(delays_samples, powers_lin):
        tap = (rng.normal(0,1) + 1j*rng.normal(0,1)) * np.sqrt(p/2)
        h[d] += tap
    return h / np.sqrt(np.sum(np.abs(h)**2))

def apply_doppler(signal, velocity_kmh, fs):
    if velocity_kmh <= 0:
        return signal
    v_ms = velocity_kmh / 3.6
    fd = (v_ms * FC) / 3e8
    t = np.arange(len(signal)) / fs
    return signal * np.exp(1j * 2 * np.pi * fd * t)

def apply_channel(tx_signal, channel_impulse_response, snr_db, velocity_kmh=0, fs=1.92e6):
    y = np.convolve(tx_signal, channel_impulse_response, mode='full')[:len(tx_signal)]
    y = apply_doppler(y, velocity_kmh, fs)
    sig_pow = np.mean(np.abs(y)**2)
    if sig_pow == 0:
        sig_pow = 1.0
    snr_lin = 10**(snr_db/10)
    noise_pow = sig_pow / snr_lin
    noise = (np.random.randn(len(y)) + 1j*np.random.randn(len(y))) * np.sqrt(noise_pow/2)
    return y + noise, channel_impulse_response


# -------------------------------------------------------------------
# SIMO: NR antenas de recepción con desvanecimiento independiente
# -------------------------------------------------------------------

def generate_mimo_channels(NR, profile_name, taps_L=None):
    """Genera NR realizaciones independientes del canal (baja correlación espacial).

    Cada antena receptora experimenta un desvanecimiento completamente
    independiente, modelando separación espacial suficiente (>= lambda/2).
    """
    return [get_channel_profile(profile_name, taps_L) for _ in range(NR)]


def apply_channel_mimo(tx_signal, channels, snr_db, velocity_kmh=0, fs=1.92e6):
    """Aplica NR canales independientes a la misma señal TX.

    Cada antena recibe: y_r = conv(tx, h_r) + doppler + n_r
    El ruido es independiente por antena (ruido térmico no correlado).
    """
    rx_signals = []
    for h in channels:
        y = np.convolve(tx_signal, h, mode='full')[:len(tx_signal)]
        y = apply_doppler(y, velocity_kmh, fs)
        sig_pow = np.mean(np.abs(y) ** 2)
        if sig_pow == 0:
            sig_pow = 1.0
        snr_lin = 10 ** (snr_db / 10)
        noise_pow = sig_pow / snr_lin
        noise = (np.random.randn(len(y)) + 1j * np.random.randn(len(y))) * np.sqrt(noise_pow / 2)
        rx_signals.append(y + noise)
    return rx_signals