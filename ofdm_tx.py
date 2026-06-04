# ofdm_tx.py
# Transmisor OFDM 4G LTE: modulación QAM con Gray, inserción de pilotos
# en el mapa de subportadoras, IFFT y adición de prefijo cíclico.
#
# Cambios respecto a la versión anterior:
# - ofdm_tx_block ahora recibe un mapa de subportadoras (sc_map) y un
#   valor de piloto. Cada símbolo OFDM contiene datos, pilotos y ceros
#   (banda de guarda + DC), igual que en LTE real.
# - La cantidad de datos por símbolo OFDM es sc_map['n_data'], menor
#   que Nfft, lo que incrementa el número de ejecuciones de la IFFT.

import numpy as np


def next_pow2(n):
    """Siguiente potencia de 2 mayor o igual a n."""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def qam_constellation(M):
    """Niveles de la constelación QAM normalizados a energía unitaria promedio."""
    m = int(np.sqrt(M))
    levels = np.arange(-(m - 1), m, 2)
    Es = (2 * (m**2 - 1)) / 3
    return levels / np.sqrt(Es)


def qam_mod(bits, M):
    """Modulación QAM con codificación Gray.

    Mapea bloques de log2(M) bits a símbolos complejos I+jQ.
    Los bits de entrada deben ser múltiplo de log2(M).
    """
    k = int(np.log2(M))
    levels = qam_constellation(M)
    kb = k // 2
    b = bits.reshape(-1, k)
    bi = b[:, :kb].dot(1 << np.arange(kb - 1, -1, -1))
    bq = b[:, kb:].dot(1 << np.arange(kb - 1, -1, -1))
    gi = np.array([x ^ (x >> 1) for x in bi], dtype=int)
    gq = np.array([x ^ (x >> 1) for x in bq], dtype=int)
    return levels[gi] + 1j * levels[gq]


def calculate_papr(x):
    """PAPR (dB) de una señal temporal: 10·log10(P_pico / P_media)."""
    power = np.abs(x) ** 2
    mean_pow = np.mean(power)
    if mean_pow == 0:
        return 0.0
    return 10 * np.log10(np.max(power) / mean_pow)


def ofdm_tx_block(data_symbols, Nfft, cp_len, sc_map, pilot_value):
    """Genera la señal OFDM transmitida con pilotos insertados.

    Estructura de cada símbolo OFDM en frecuencia:
      - sc_map['data_indices']  -> símbolos QAM de datos
      - sc_map['pilot_indices'] -> valor piloto conocido (para estimación de canal)
      - Banda de guarda + DC    -> ceros

    Flujo por símbolo: asignar subportadoras -> IFFT -> añadir CP

    Returns
    -------
    tx_signal : ndarray complex
        Señal OFDM concatenada en el dominio del tiempo.
    papr_list : list[float]
        PAPR (dB) de cada símbolo OFDM (antes del CP).
    n_ofdm : int
        Número de símbolos OFDM generados (= ejecuciones de IFFT).
    """
    n_data = sc_map["n_data"]
    n_ofdm = int(np.ceil(len(data_symbols) / n_data))

    # Padding para completar el último símbolo
    total_data = n_ofdm * n_data
    pad = total_data - len(data_symbols)
    if pad > 0:
        data_symbols = np.concatenate([data_symbols, np.zeros(pad, dtype=complex)])

    data_frames = data_symbols.reshape(n_ofdm, n_data)

    sym_len = Nfft + cp_len
    tx_signal = np.zeros(n_ofdm * sym_len, dtype=complex)
    papr_list = []

    data_idx = sc_map["data_indices"]
    pilot_idx = sc_map["pilot_indices"]

    for i in range(n_ofdm):
        # Mapeo de subportadoras
        X = np.zeros(Nfft, dtype=complex)
        X[data_idx] = data_frames[i]
        X[pilot_idx] = pilot_value
        # Guarda + DC permanecen en 0

        # IFFT -> dominio del tiempo
        x = np.fft.ifft(X)
        papr_list.append(calculate_papr(x))

        # Prefijo cíclico: últimas cp_len muestras copiadas al inicio
        offset = i * sym_len
        if cp_len > 0:
            tx_signal[offset : offset + cp_len] = x[-cp_len:]
            tx_signal[offset + cp_len : offset + sym_len] = x
        else:
            tx_signal[offset : offset + sym_len] = x

    return tx_signal, papr_list, n_ofdm
