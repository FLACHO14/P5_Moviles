# ofdm_tx.py
# Transmisor OFDM: modulación QAM con codificación Gray y generación de
# símbolos OFDM por bloques (IFFT + prefijo cíclico).
#
# Cambio respecto a la versión anterior: se eliminó el esquema de pilotos
# (inserción en subportadoras específicas) y se adoptó un enfoque de bloques
# puros donde cada trama de Nfft símbolos se transforma directamente con IFFT.
# Esto permite que la ecualización en RX use la respuesta H conocida del canal
# (bloque a bloque), evitando la interpolación de pilotos que era fuente de error.

import numpy as np


def next_pow2(n):
    """Devuelve la potencia de 2 más pequeña >= n (necesaria para elegir Nfft)."""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def qam_constellation(M):
    """Genera los niveles reales normalizados para un mapeo M-QAM cuadrado.
    La normalización a energía unitaria (Es=1) garantiza comparación justa
    entre modulaciones al calcular BER vs SNR.
    """
    m = int(np.sqrt(M))
    levels = np.arange(-(m - 1), m, 2)
    Es = (2 * (m**2 - 1)) / 3
    return levels / np.sqrt(Es)


def qam_mod(bits, M):
    """Modula bits en símbolos M-QAM con codificación Gray.
    La codificación Gray garantiza que símbolos vecinos en la constelación
    difieran en un solo bit, minimizando la BER en canales ruidosos.
    """
    k = int(np.log2(M))
    levels = qam_constellation(M)
    kb = k // 2  # bits por componente I o Q
    b = bits.reshape(-1, k)
    # Convertir grupos de bits a índices enteros
    bi = b[:, :kb].dot(1 << np.arange(kb - 1, -1, -1))
    bq = b[:, kb:].dot(1 << np.arange(kb - 1, -1, -1))
    # Aplicar codificación Gray: n -> n XOR (n >> 1)
    gi = np.array([x ^ (x >> 1) for x in bi], dtype=int)
    gq = np.array([x ^ (x >> 1) for x in bq], dtype=int)
    return levels[gi] + 1j * levels[gq]


def calculate_papr(x):
    """PAPR (Peak-to-Average Power Ratio) en dB para un símbolo OFDM en tiempo."""
    power = np.abs(x) ** 2
    mean_pow = np.mean(power)
    if mean_pow == 0:
        return 0.0
    return 10 * np.log10(np.max(power) / mean_pow)


def ofdm_tx_block(symbols, Nfft, cp_len):
    """Genera la señal OFDM completa a partir de símbolos QAM.

    Proceso: zero-pad → reshape en tramas de Nfft → IFFT por trama →
    añadir prefijo cíclico (últimas cp_len muestras) → serializar.

    El prefijo cíclico convierte la convolución lineal del canal en circular,
    lo que permite ecualizar en frecuencia con una simple división por H[k].
    Se requiere cp_len >= longitud del canal - 1 para eliminar ISI.

    Retorna: (señal_tx serializada, lista de PAPR por símbolo OFDM)
    """
    # Zero-padding para completar el último bloque de Nfft símbolos
    pad = (-len(symbols)) % Nfft
    if pad:
        symbols = np.concatenate([symbols, np.zeros(pad, dtype=np.complex128)])
    frames = symbols.reshape(-1, Nfft)

    # IFFT convierte el dominio frecuencia → tiempo
    x = np.fft.ifft(frames, axis=1)
    papr_list = [calculate_papr(xi) for xi in x]

    # Prefijo cíclico: copia de las últimas cp_len muestras al inicio de cada símbolo
    cp = x[:, -cp_len:] if cp_len > 0 else np.empty((x.shape[0], 0), dtype=complex)
    tx = np.concatenate([cp, x], axis=1)
    return tx.reshape(-1), papr_list
