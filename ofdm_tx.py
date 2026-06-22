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


def get_best_dft_size(n_data, Nfft):
    """Calcula el mejor tamaño de DFT para SC-FDMA (potencia de 2, < Nfft, <= n_data)."""
    max_m = min(n_data, Nfft - 1)
    if max_m < 2:
        return 2
    return 2 ** int(np.floor(np.log2(max_m)))


def scfdma_tx_block(data_symbols, Nfft, cp_len, sc_map, pilot_value, M_dft):
    """SC-FDMA TX: QAM → DFT(M_dft) → mapeo subportadoras → IFFT(Nfft) → CP.

    La FFT previa (DFT precoding) reduce el PAPR al hacer que la señal
    temporal se asemeje a portadora única.
    """
    n_data = M_dft
    n_ofdm = int(np.ceil(len(data_symbols) / n_data))

    total_data = n_ofdm * n_data
    pad = total_data - len(data_symbols)
    if pad > 0:
        data_symbols = np.concatenate([data_symbols, np.zeros(pad, dtype=complex)])

    data_frames = data_symbols.reshape(n_ofdm, n_data)

    sym_len = Nfft + cp_len
    tx_signal = np.zeros(n_ofdm * sym_len, dtype=complex)
    papr_list = []

    data_idx = sc_map["data_indices"][:M_dft]
    pilot_idx = sc_map["pilot_indices"]

    for i in range(n_ofdm):
        X_dft = np.fft.fft(data_frames[i], n=M_dft)

        X = np.zeros(Nfft, dtype=complex)
        X[data_idx] = X_dft
        X[pilot_idx] = pilot_value

        x = np.fft.ifft(X)
        papr_list.append(calculate_papr(x))

        offset = i * sym_len
        if cp_len > 0:
            tx_signal[offset : offset + cp_len] = x[-cp_len:]
            tx_signal[offset + cp_len : offset + sym_len] = x
        else:
            tx_signal[offset : offset + sym_len] = x

    return tx_signal, papr_list, n_ofdm


# -------------------------------------------------------------------
# SFBC (Space-Frequency Block Code) para MISO
# -------------------------------------------------------------------

def sfbc_tx_2ant(data_symbols, Nfft, cp_len, sc_map, pilot_value):
    """SFBC para 2 antenas TX. Mapea símbolos en pares adyacentes.

    Estructura Alamouti en frecuencia:
    Antena 1, subportadora k:     s0
    Antena 2, subportadora k:     s1
    Antena 1, subportadora k+1:   -s1*
    Antena 2, subportadora k+1:   s0*

    La tasa es 1 (igual que SISO): cada par de subportadoras adyacentes
    transporta dos símbolos de datos (s0, s1). Los pilotos son ortogonales:
    la antena 1 transmite en pilot_indices_ant1 y cero en los de la antena 2,
    y viceversa, lo que permite estimar H1 y H2 por separado.

    Returns
    -------
    tx1, tx2 : ndarray complex
        Señales TX de ambas antenas.
    papr_list : list[float]
        PAPR promedio de ambas antenas.
    n_ofdm : int
        Número de símbolos OFDM.
    pilot_indices_ant1, pilot_indices_ant2 : ndarray
        Índices ortogonales para pilotos por antena.
    """
    n_data = sc_map["n_data"]
    n_ofdm = int(np.ceil(len(data_symbols) / n_data))

    total_data = n_ofdm * n_data
    pad = total_data - len(data_symbols)
    if pad > 0:
        data_symbols = np.concatenate([data_symbols, np.zeros(pad, dtype=complex)])

    data_frames = data_symbols.reshape(n_ofdm, n_data)

    sym_len = Nfft + cp_len
    tx1 = np.zeros(n_ofdm * sym_len, dtype=complex)
    tx2 = np.zeros(n_ofdm * sym_len, dtype=complex)
    papr_list = []

    data_idx = sc_map["data_indices"]
    pilot_idx = sc_map["pilot_indices"]

    pilot_indices_ant1 = pilot_idx[::2]
    pilot_indices_ant2 = pilot_idx[1::2]

    n_pairs = (n_data // 2) * 2

    for i in range(n_ofdm):
        frame = data_frames[i]

        X1 = np.zeros(Nfft, dtype=complex)
        X2 = np.zeros(Nfft, dtype=complex)

        for p in range(0, n_pairs, 2):
            k1 = data_idx[p]
            k2 = data_idx[p + 1]
            s0 = frame[p]
            s1 = frame[p + 1]

            # Antena 1: [s0, -s1*]   Antena 2: [s1, s0*]
            X1[k1] = s0
            X1[k2] = -np.conj(s1)
            X2[k1] = s1
            X2[k2] = np.conj(s0)

        X1[pilot_indices_ant1] = pilot_value
        X2[pilot_indices_ant2] = pilot_value

        x1 = np.fft.ifft(X1)
        x2 = np.fft.ifft(X2)
        papr_list.append((calculate_papr(x1) + calculate_papr(x2)) / 2)

        offset = i * sym_len
        if cp_len > 0:
            tx1[offset : offset + cp_len] = x1[-cp_len:]
            tx1[offset + cp_len : offset + sym_len] = x1
            tx2[offset : offset + cp_len] = x2[-cp_len:]
            tx2[offset + cp_len : offset + sym_len] = x2
        else:
            tx1[offset : offset + sym_len] = x1
            tx2[offset : offset + sym_len] = x2

    return tx1, tx2, papr_list, n_ofdm, pilot_indices_ant1, pilot_indices_ant2


def sfbc_tx_3ant(data_symbols, Nfft, cp_len, sc_map, pilot_value):
    """SFBC para 3 antenas TX usando esquema extendido.

    Estructura por bloques de 4 subportadoras:
    Ant1: [s0, s1, -s2*, -s3*]
    Ant2: [s1, -s0*, s3*, -s2]
    Ant3: [s2, s3, s0*, s1*]
    """
    n_data = sc_map["n_data"]
    n_ofdm = int(np.ceil(len(data_symbols) / (4 * n_data // 3)))

    block_size = 4
    symbols_per_ofdm = (len(sc_map["data_indices"]) // block_size) * block_size
    n_ofdm = int(np.ceil(len(data_symbols) / symbols_per_ofdm))

    total_data = n_ofdm * symbols_per_ofdm
    pad = total_data - len(data_symbols)
    if pad > 0:
        data_symbols = np.concatenate([data_symbols, np.zeros(pad, dtype=complex)])

    data_blocks = data_symbols.reshape(n_ofdm, symbols_per_ofdm)

    sym_len = Nfft + cp_len
    tx1 = np.zeros(n_ofdm * sym_len, dtype=complex)
    tx2 = np.zeros(n_ofdm * sym_len, dtype=complex)
    tx3 = np.zeros(n_ofdm * sym_len, dtype=complex)
    papr_list = []

    data_idx = sc_map["data_indices"][:symbols_per_ofdm]
    pilot_idx = sc_map["pilot_indices"]

    pilot_indices_ant1 = pilot_idx[::3]
    pilot_indices_ant2 = pilot_idx[1::3]
    pilot_indices_ant3 = pilot_idx[2::3]

    for i in range(n_ofdm):
        X1 = np.zeros(Nfft, dtype=complex)
        X2 = np.zeros(Nfft, dtype=complex)
        X3 = np.zeros(Nfft, dtype=complex)

        data_block = data_blocks[i]

        for j in range(0, len(data_idx) - 3, 4):
            if j + 3 < len(data_block):
                s0, s1, s2, s3 = data_block[j:j+4]
                k0, k1, k2, k3 = data_idx[j:j+4]

                X1[k0], X1[k1], X1[k2], X1[k3] = s0, s1, -np.conj(s2), -np.conj(s3)
                X2[k0], X2[k1], X2[k2], X2[k3] = s1, -np.conj(s0), np.conj(s3), -s2
                X3[k0], X3[k1], X3[k2], X3[k3] = s2, s3, np.conj(s0), np.conj(s1)

        X1[pilot_indices_ant1] = pilot_value
        X2[pilot_indices_ant2] = pilot_value
        X3[pilot_indices_ant3] = pilot_value

        x1 = np.fft.ifft(X1)
        x2 = np.fft.ifft(X2)
        x3 = np.fft.ifft(X3)
        papr_list.append((calculate_papr(x1) + calculate_papr(x2) + calculate_papr(x3)) / 3)

        offset = i * sym_len
        if cp_len > 0:
            tx1[offset : offset + cp_len] = x1[-cp_len:]
            tx1[offset + cp_len : offset + sym_len] = x1
            tx2[offset : offset + cp_len] = x2[-cp_len:]
            tx2[offset + cp_len : offset + sym_len] = x2
            tx3[offset : offset + cp_len] = x3[-cp_len:]
            tx3[offset + cp_len : offset + sym_len] = x3
        else:
            tx1[offset : offset + sym_len] = x1
            tx2[offset : offset + sym_len] = x2
            tx3[offset : offset + sym_len] = x3

    return tx1, tx2, tx3, papr_list, n_ofdm, pilot_indices_ant1, pilot_indices_ant2, pilot_indices_ant3


def sfbc_tx_4ant(data_symbols, Nfft, cp_len, sc_map, pilot_value):
    """SFBC para 4 antenas TX usando esquema ortogonal.

    Estructura por bloques de 4 subportadoras (2x2 Alamouti real):
    Ant1: [s0, -s1*]
    Ant2: [s1, s0*]
    Ant3: [s2, -s3*]
    Ant4: [s3, s2*]
    """
    n_data = sc_map["n_data"]
    block_size = 4
    symbols_per_ofdm = (len(sc_map["data_indices"]) // block_size) * block_size
    n_ofdm = int(np.ceil(len(data_symbols) / symbols_per_ofdm))

    total_data = n_ofdm * symbols_per_ofdm
    pad = total_data - len(data_symbols)
    if pad > 0:
        data_symbols = np.concatenate([data_symbols, np.zeros(pad, dtype=complex)])

    data_blocks = data_symbols.reshape(n_ofdm, symbols_per_ofdm)

    sym_len = Nfft + cp_len
    tx_sigs = [np.zeros(n_ofdm * sym_len, dtype=complex) for _ in range(4)]
    papr_list = []

    data_idx = sc_map["data_indices"][:symbols_per_ofdm]
    pilot_idx = sc_map["pilot_indices"]

    pilot_indices = [pilot_idx[i::4] for i in range(4)]

    for i in range(n_ofdm):
        X_sigs = [np.zeros(Nfft, dtype=complex) for _ in range(4)]

        data_block = data_blocks[i]

        for j in range(0, len(data_idx) - 3, 4):
            if j + 3 < len(data_block):
                s0, s1, s2, s3 = data_block[j:j+4]
                k0, k1, k2, k3 = data_idx[j:j+4]

                X_sigs[0][k0], X_sigs[0][k1] = s0, -np.conj(s1)
                X_sigs[1][k0], X_sigs[1][k1] = s1, np.conj(s0)
                X_sigs[2][k2], X_sigs[2][k3] = s2, -np.conj(s3)
                X_sigs[3][k2], X_sigs[3][k3] = s3, np.conj(s2)

        for ant in range(4):
            X_sigs[ant][pilot_indices[ant]] = pilot_value

        x_sigs = [np.fft.ifft(X) for X in X_sigs]
        papr_list.append(np.mean([calculate_papr(x) for x in x_sigs]))

        offset = i * sym_len
        for ant in range(4):
            if cp_len > 0:
                tx_sigs[ant][offset : offset + cp_len] = x_sigs[ant][-cp_len:]
                tx_sigs[ant][offset + cp_len : offset + sym_len] = x_sigs[ant]
            else:
                tx_sigs[ant][offset : offset + sym_len] = x_sigs[ant]

    return tx_sigs[0], tx_sigs[1], tx_sigs[2], tx_sigs[3], papr_list, n_ofdm, pilot_indices
