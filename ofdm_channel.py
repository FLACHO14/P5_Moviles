# ofdm_channel.py
# Modelos de canal para la simulación OFDM: Rayleigh multitrayecto, Rician y canal ideal.
# Incluye efecto Doppler y adición de ruido AWGN.
#
# Cambio respecto a la versión anterior: se separaron los modelos Rayleigh y Rician
# en funciones distintas con generación reproducible (seed). La función apply_channel
# calcula la potencia de ruido a partir de la señal recibida (post-canal), no de la
# señal transmitida, lo que da una estimación correcta de SNR en presencia de fading.

import numpy as np


def multipath_rayleigh_channel(L, seed=None):
    """Canal Rayleigh con L taps independientes (desvanecimiento plano por trama).

    Cada tap sigue una distribución CN(0,1). La normalización a norma 1
    preserva la potencia de señal independientemente de L, permitiendo que
    la curva BER vs SNR sea comparable entre distintos valores de L.
    """
    rng = np.random.default_rng(seed)
    h = (rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)) / np.sqrt(2)
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def multipath_rician_channel(L, K_factor=3, seed=None):
    """Canal Rician: componente LOS (determinista) + NLOS (aleatoria Rayleigh).

    K_factor es la relación potencia LOS / potencia NLOS. K=3 dB es típico
    de ambientes semi-abiertos con línea de visión directa moderada.
    La componente LOS solo se agrega al primer tap (tap de llegada directa).
    """
    rng = np.random.default_rng(seed)
    nlos = (rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)) / np.sqrt(2)
    h = np.sqrt(1 / (K_factor + 1)) * nlos
    h[0] += np.sqrt(K_factor / (K_factor + 1))  # componente LOS en tap 0
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def apply_channel(tx_signal, channel_type, snr_db, h=None, velocity_kmh=0, fs=1e6):
    """Aplica el canal al dominio del tiempo: convolución multitrayecto + Doppler + AWGN.

    Para canal Ideal (h=None), la señal pasa sin distorsión (solo AWGN).
    Para Rayleigh/Rician, se convoluciona con h y se trunca a la longitud
    original; el ISI de la última trama se desprecia (la CP lo maneja).

    El desplazamiento Doppler modela la variación de fase por movimiento del
    receptor. Se aplica como rotación compleja: e^{j2π·fd·t}, donde
    fd = v·fc/c (frecuencia Doppler máxima con fc=2.1 GHz para 4G).

    El ruido AWGN se dimensiona sobre la potencia de la señal recibida (y),
    no de la transmitida, para respetar la definición práctica de SNR en RX.
    """
    fc = 2.1e9  # frecuencia portadora 4G (Hz)

    if channel_type == "Ideal" or h is None:
        # Canal ideal: impulso unitario → sin distorsión de multitrayecto
        h_used = np.array([1.0 + 0j])
        y = tx_signal.astype(np.complex128).copy()
    else:
        h_used = h
        # mode='full' devuelve len(tx)+len(h)-1; se trunca para mantener
        # la longitud original. El CP absorbe la cola de ISI siempre que
        # cp_len >= len(h) - 1.
        y = np.convolve(tx_signal, h_used, mode="full")[: len(tx_signal)]

    # Efecto Doppler: rotación de fase proporcional a la velocidad y el tiempo
    if velocity_kmh > 0:
        v_ms = velocity_kmh / 3.6
        fd = (v_ms * fc) / 3e8  # frecuencia Doppler máxima [Hz]
        t = np.arange(len(y)) / fs
        y = y * np.exp(1j * 2 * np.pi * fd * t)

    # AWGN: ruido complejo con varianza noise_pow/2 por componente I y Q
    sig_pow = np.mean(np.abs(y) ** 2)
    if sig_pow == 0:
        sig_pow = 1.0  # evita división por cero en señal nula
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = sig_pow / snr_lin
    noise = (np.random.randn(len(y)) + 1j * np.random.randn(len(y))) * np.sqrt(
        noise_pow / 2
    )
    return y + noise, h_used
