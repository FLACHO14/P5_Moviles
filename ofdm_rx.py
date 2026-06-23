# ofdm_rx.py
# Receptor OFDM 4G LTE: eliminación de CP, FFT, ecualización por pilotos
# (interpolación lineal) o canal conocido, y demodulación QAM con Gray.
#
# Cambios respecto a la versión anterior:
# - ofdm_rx_block retorna array 2-D (n_ofdm, Nfft) para procesar cada
#   símbolo de forma independiente.
# - equalize_with_pilots: estima H interpolando desde las subportadoras
#   piloto y ecualiza (ZF) por símbolo. Esto es la cadena realista.
# - equalize_with_known_h: ecualización con H ideal (para referencia).
# - demod_without_equalization: extrae datos crudos sin corregir el canal,
#   para demostrar la importancia de la ecualización por pilotos.

import numpy as np
from scipy.interpolate import interp1d


def gray_decode(g):
    """Decodifica un entero en Gray a binario natural."""
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


# -------------------------------------------------------------------
# Bloque receptor OFDM
# -------------------------------------------------------------------

def ofdm_rx_block(rx_signal, Nfft, cp_len):
    """Elimina el prefijo cíclico y aplica FFT a cada símbolo OFDM.

    Returns
    -------
    Y_frames : ndarray complex, shape (n_ofdm, Nfft)
        Cada fila es un símbolo OFDM en el dominio de la frecuencia.
    """
    sym_len = Nfft + cp_len
    n_ofdm = len(rx_signal) // sym_len
    if n_ofdm == 0:
        return np.zeros((0, Nfft), dtype=complex)
    rx = rx_signal[: n_ofdm * sym_len].reshape(n_ofdm, sym_len)
    r_nocp = rx[:, cp_len:]  # eliminar CP
    Y = np.fft.fft(r_nocp, axis=1)
    return Y


# -------------------------------------------------------------------
# Ecualización
# -------------------------------------------------------------------

def equalize_with_pilots(Y_frames, sc_map, pilot_value, Nfft):
    """Ecualización Zero-Forcing basada en subportadoras piloto.

    Para cada símbolo OFDM recibido:
    1. H_est[k_pilot] = Y[k_pilot] / pilot_value  (estimación puntual)
    2. H_est[k] = interpolación lineal sobre todos los índices
    3. X_hat[k] = Y[k] / H_est[k]   (ecualización ZF)
    4. Se extraen solo las subportadoras de datos

    Returns
    -------
    data_symbols : ndarray complex
        Símbolos de datos ecualizados, concatenados.
    H_estimates : list[ndarray]
        Canal estimado por símbolo OFDM (útil para visualización).
    """
    n_frames = Y_frames.shape[0]
    pilot_idx = sc_map["pilot_indices"]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)

    # Pre-ordenar pilotos por índice para interpolación
    sort_order = np.argsort(pilot_idx)
    p_sorted = pilot_idx[sort_order]

    all_data = np.zeros(n_frames * n_data, dtype=complex)
    H_estimates = []

    all_idx = np.arange(Nfft)

    for i in range(n_frames):
        Y = Y_frames[i]

        # Estimación de canal en posiciones piloto
        H_pilots = Y[pilot_idx] / pilot_value
        H_sorted = H_pilots[sort_order]

        # Interpolación real e imaginaria por separado
        f_re = interp1d(
            p_sorted, np.real(H_sorted),
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        f_im = interp1d(
            p_sorted, np.imag(H_sorted),
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        H_est = f_re(all_idx) + 1j * f_im(all_idx)
        H_estimates.append(H_est)

        # Ecualización ZF + extracción de datos
        Y_eq = Y / (H_est + 1e-12)
        all_data[i * n_data : (i + 1) * n_data] = Y_eq[data_idx]

    return all_data, H_estimates


def equalize_with_known_h(Y_frames, H_freq, sc_map):
    """Ecualización ZF con canal perfectamente conocido (referencia ideal).

    Sirve para comparar el rendimiento máximo teórico contra la
    estimación por pilotos.
    """
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)
    n_frames = Y_frames.shape[0]

    all_data = np.zeros(n_frames * n_data, dtype=complex)
    for i in range(n_frames):
        Y_eq = Y_frames[i] / (H_freq + 1e-12)
        all_data[i * n_data : (i + 1) * n_data] = Y_eq[data_idx]
    return all_data


def demod_without_equalization(Y_frames, sc_map):
    """Extrae datos SIN ecualizar: demuestra el efecto destructivo del canal.

    Sin corrección del canal la constelación recibida está distorsionada
    y el BER es muy alto, especialmente en canales selectivos en frecuencia.
    """
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)
    n_frames = Y_frames.shape[0]

    all_data = np.zeros(n_frames * n_data, dtype=complex)
    for i in range(n_frames):
        all_data[i * n_data : (i + 1) * n_data] = Y_frames[i][data_idx]
    return all_data


# -------------------------------------------------------------------
# Demodulación QAM
# -------------------------------------------------------------------

def qam_demod(symbols, M):
    """Demodulación QAM con decodificación Gray.

    Para cada símbolo recibido, encuentra el punto de constelación más
    cercano y lo decodifica a bits.
    """
    from ofdm_tx import qam_constellation

    k = int(np.log2(M))
    levels = qam_constellation(M)
    kb = k // 2

    I = np.real(symbols)
    Q = np.imag(symbols)

    # Decisión de mínima distancia
    idxI = np.argmin(np.abs(I[:, None] - levels[None, :]), axis=1)
    idxQ = np.argmin(np.abs(Q[:, None] - levels[None, :]), axis=1)

    # Gray -> natural
    ii = np.array([gray_decode(int(g)) for g in idxI], dtype=int)
    qq = np.array([gray_decode(int(g)) for g in idxQ], dtype=int)

    bi = ((ii[:, None] >> np.arange(kb - 1, -1, -1)) & 1).astype(np.uint8)
    bq = ((qq[:, None] >> np.arange(kb - 1, -1, -1)) & 1).astype(np.uint8)
    return np.concatenate([bi, bq], axis=1).reshape(-1).astype(np.uint8)


# -------------------------------------------------------------------
# Diversidad RX: estimación de canal y algoritmos de combinación
# -------------------------------------------------------------------

def estimate_channel_from_pilots(Y_frames, sc_map, pilot_value, Nfft):
    """Estima el canal H por símbolo OFDM usando subportadoras piloto.

    Misma lógica que equalize_with_pilots pero retorna solo las
    estimaciones de canal, sin ecualizar. Se usa cuando la ecualización
    la realiza un combinador de diversidad externo.
    """
    n_frames = Y_frames.shape[0]
    pilot_idx = sc_map["pilot_indices"]
    sort_order = np.argsort(pilot_idx)
    p_sorted = pilot_idx[sort_order]
    all_idx = np.arange(Nfft)
    H_estimates = []

    for i in range(n_frames):
        H_pilots = Y_frames[i][pilot_idx] / pilot_value
        H_sorted = H_pilots[sort_order]
        f_re = interp1d(
            p_sorted, np.real(H_sorted),
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        f_im = interp1d(
            p_sorted, np.imag(H_sorted),
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        H_estimates.append(f_re(all_idx) + 1j * f_im(all_idx))

    return H_estimates


def combine_mrc(Y_frames_list, H_est_all, sc_map):
    """Maximum-Ratio Combining en el dominio de la frecuencia.

    Para cada subportadora k, símbolo OFDM i:
      X_hat[k] = sum_r( H_r*[k] · Y_r[k] ) / sum_r( |H_r[k]|² )

    Alinea fases y pondera por ganancia de canal → maximiza SNR combinada.
    Los pesos se aplican ANTES de la de-precodificación (IDFT de tamaño M).
    """
    NR = len(Y_frames_list)
    n_frames = Y_frames_list[0].shape[0]
    Nfft = Y_frames_list[0].shape[1]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)

    all_data = np.zeros(n_frames * n_data, dtype=complex)
    H_combined_power = []

    for i in range(n_frames):
        num = np.zeros(Nfft, dtype=complex)
        den = np.zeros(Nfft, dtype=float)
        for r in range(NR):
            H = H_est_all[r][i]
            Y = Y_frames_list[r][i]
            num += np.conj(H) * Y
            den += np.abs(H) ** 2
        den = np.maximum(den, 1e-12)
        X_hat = num / den
        H_combined_power.append(den)
        all_data[i * n_data : (i + 1) * n_data] = X_hat[data_idx]

    return all_data, H_combined_power


def combine_sc(Y_frames_list, H_est_all, sc_map):
    """Selection Combining: selecciona la antena con mayor |H|² por subportadora.

    Para cada subportadora k:
      r* = argmax_r |H_r[k]|²
      X_hat[k] = Y_{r*}[k] / H_{r*}[k]
    """
    NR = len(Y_frames_list)
    n_frames = Y_frames_list[0].shape[0]
    Nfft = Y_frames_list[0].shape[1]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)

    all_data = np.zeros(n_frames * n_data, dtype=complex)

    for i in range(n_frames):
        H_stack = np.array([H_est_all[r][i] for r in range(NR)])
        Y_stack = np.array([Y_frames_list[r][i] for r in range(NR)])
        best = np.argmax(np.abs(H_stack) ** 2, axis=0)
        k_idx = np.arange(Nfft)
        X_hat = Y_stack[best, k_idx] / (H_stack[best, k_idx] + 1e-12)
        all_data[i * n_data : (i + 1) * n_data] = X_hat[data_idx]

    return all_data


def combine_mmse(Y_frames_list, H_est_all, sc_map, snr_db):
    """MMSE Combining: minimiza el error cuadrático medio.

    Para SIMO (1 TX, NR RX):
      X_hat[k] = sum_r(H_r*[k]·Y_r[k]) / (sum_r(|H_r[k]|²) + σ²_n)

    Similar a MRC pero con regularización por ruido, lo que evita
    amplificar el ruido en subportadoras con canal débil.
    """
    NR = len(Y_frames_list)
    n_frames = Y_frames_list[0].shape[0]
    Nfft = Y_frames_list[0].shape[1]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)

    noise_var = 1.0 / (10 ** (snr_db / 10)) if snr_db > -30 else 1e3

    all_data = np.zeros(n_frames * n_data, dtype=complex)

    for i in range(n_frames):
        num = np.zeros(Nfft, dtype=complex)
        den = np.zeros(Nfft, dtype=float)
        for r in range(NR):
            H = H_est_all[r][i]
            Y = Y_frames_list[r][i]
            num += np.conj(H) * Y
            den += np.abs(H) ** 2
        X_hat = num / np.maximum(den + noise_var, 1e-12)
        all_data[i * n_data : (i + 1) * n_data] = X_hat[data_idx]

    return all_data


def scfdma_despread(eq_symbols, M_dft):
    """IDFT de-spreading para SC-FDMA: deshace el DFT precoding del TX."""
    n_total = len(eq_symbols)
    n_ofdm = n_total // M_dft
    if n_ofdm == 0:
        return eq_symbols
    used = eq_symbols[: n_ofdm * M_dft].reshape(n_ofdm, M_dft)
    despread = np.fft.ifft(used, axis=1)
    result = despread.reshape(-1)
    remaining = eq_symbols[n_ofdm * M_dft :]
    if len(remaining) > 0:
        result = np.concatenate([result, remaining])
    return result


# -------------------------------------------------------------------
# SFBC Decoding (Space-Frequency Block Code)
# -------------------------------------------------------------------

def sfbc_decode_2ant(Y_frames_rx, H_est_tx1, H_est_tx2, sc_map, data_idx):
    """Decodifica SFBC de 2 antenas TX en dominio de frecuencia (1 RX).

    Estructura Alamouti:
    Subportadora k: X1[k]=s0, X2[k]=s1
    Subportadora k+1: X1[k+1]=-s1*, X2[k+1]=s0*

    Decodificación:
    ŝ0 = (H1*·Y[k] + H2·Y*[k+1]) / (|H1|² + |H2|²)
    ŝ1 = (H2*·Y[k] - H1·Y*[k+1]) / (|H1|² + |H2|²)

    El escalamiento interno de potencia del TX se absorbe en la estimación
    de canal por pilotos, por lo que se cancela en el cociente y no afecta
    la decodificación. La subportadora dummy de paridad (si el número de
    subportadoras es impar) se recupera con ecualización ZF de una antena.
    """
    n_frames = Y_frames_rx.shape[0]
    n_used = len(data_idx)
    dummy_added = (n_used % 2 == 1)
    n_pairs = n_used - 1 if dummy_added else n_used
    all_data = []

    for i in range(n_frames):
        Y = Y_frames_rx[i]
        H1 = H_est_tx1[i]
        H2 = H_est_tx2[i]

        decoded = np.zeros(n_used, dtype=complex)

        for j in range(0, n_pairs, 2):
            k1 = data_idx[j]
            k2 = data_idx[j + 1]

            h1, h2 = H1[k1], H2[k1]
            y1, y2_conj = Y[k1], np.conj(Y[k2])

            denom = np.abs(h1) ** 2 + np.abs(h2) ** 2 + 1e-12
            s0 = (np.conj(h1) * y1 + h2 * y2_conj) / denom
            s1 = (np.conj(h2) * y1 - h1 * y2_conj) / denom

            decoded[j] = s0
            decoded[j + 1] = s1

        # Subportadora dummy sobrante: ZF con la antena 1
        if dummy_added:
            k_last = data_idx[n_used - 1]
            h1 = H1[k_last]
            decoded[n_used - 1] = Y[k_last] * np.conj(h1) / (np.abs(h1) ** 2 + 1e-12)

        all_data.extend(decoded)

    return np.array(all_data)


def sfbc_decode_2ant_2rx(Y_rx_list, H_tx1_list, H_tx2_list, sc_map, data_idx):
    """Decodifica SFBC de 2 antenas TX combinando 2 antenas RX (MIMO 2x2).

    Cada antena receptora aporta una observación independiente del mismo
    bloque Alamouti. Las observaciones se combinan sumando numeradores y
    denominadores (MRC entre receptores), lo que maximiza la probabilidad
    de recuperar los símbolos frente a desvanecimientos profundos:

    ŝ0 = sum_r(H1_r*·Y_r[k] + H2_r·Y_r*[k+1]) / sum_r(|H1_r|² + |H2_r|²)
    ŝ1 = sum_r(H2_r*·Y_r[k] - H1_r·Y_r*[k+1]) / sum_r(|H1_r|² + |H2_r|²)

    El orden de diversidad resultante es 4 (2 TX x 2 RX).
    """
    NR = len(Y_rx_list)
    n_frames = Y_rx_list[0].shape[0]
    n_used = len(data_idx)
    dummy_added = (n_used % 2 == 1)
    n_pairs = n_used - 1 if dummy_added else n_used
    all_data = []

    for i in range(n_frames):
        decoded = np.zeros(n_used, dtype=complex)

        for j in range(0, n_pairs, 2):
            k1 = data_idx[j]
            k2 = data_idx[j + 1]

            num0 = 0j
            num1 = 0j
            den = 0.0
            for r in range(NR):
                Y = Y_rx_list[r][i]
                h1 = H_tx1_list[r][i][k1]
                h2 = H_tx2_list[r][i][k1]
                y1 = Y[k1]
                y2_conj = np.conj(Y[k2])
                num0 += np.conj(h1) * y1 + h2 * y2_conj
                num1 += np.conj(h2) * y1 - h1 * y2_conj
                den += np.abs(h1) ** 2 + np.abs(h2) ** 2

            den += 1e-12
            decoded[j] = num0 / den
            decoded[j + 1] = num1 / den

        if dummy_added:
            k_last = data_idx[n_used - 1]
            num = 0j
            den = 0.0
            for r in range(NR):
                h1 = H_tx1_list[r][i][k_last]
                num += np.conj(h1) * Y_rx_list[r][i][k_last]
                den += np.abs(h1) ** 2
            decoded[n_used - 1] = num / (den + 1e-12)

        all_data.extend(decoded)

    return np.array(all_data)


def sfbc_decode_3ant(Y_frames_rx, H_est_list, sc_map, data_idx):
    """Decodifica SFBC de 3 antenas TX (esquema extendido)."""
    n_frames = Y_frames_rx.shape[0]
    all_data = []

    for i in range(n_frames):
        Y = Y_frames_rx[i]
        H_list = [H[i] for H in H_est_list]

        decoded = np.zeros(len(data_idx), dtype=complex)

        for j in range(0, len(data_idx) - 3, 4):
            if j + 3 < len(data_idx):
                k0, k1, k2, k3 = data_idx[j:j+4]

                h1_0, h1_1, h1_2, h1_3 = H_list[0][k0], H_list[0][k1], H_list[0][k2], H_list[0][k3]
                h2_0, h2_1, h2_2, h2_3 = H_list[1][k0], H_list[1][k1], H_list[1][k2], H_list[1][k3]
                h3_0, h3_1, h3_2, h3_3 = H_list[2][k0], H_list[2][k1], H_list[2][k2], H_list[2][k3]

                y0, y1, y2_conj, y3_conj = Y[k0], Y[k1], np.conj(Y[k2]), np.conj(Y[k3])

                G = (np.abs(h1_0) ** 2 + np.abs(h2_0) ** 2 + np.abs(h3_0) ** 2 +
                     np.abs(h1_1) ** 2 + np.abs(h2_1) ** 2 + np.abs(h3_1) ** 2 + 1e-12)

                s0 = (np.conj(h1_0) * y0 + np.conj(h2_1) * y1 + h3_2 * y2_conj + h3_3 * y3_conj) / G
                s1 = (np.conj(h1_1) * y0 - np.conj(h2_0) * y1 + h3_3 * y2_conj - h3_2 * y3_conj) / G
                s2 = (np.conj(h1_2) * y0 + np.conj(h2_3) * y1 + h3_0 * y2_conj + h3_1 * y3_conj) / G
                s3 = (np.conj(h1_3) * y0 - np.conj(h2_2) * y1 + h3_1 * y2_conj - h3_0 * y3_conj) / G

                decoded[j:j+4] = [s0, s1, s2, s3]

        all_data.extend(decoded)

    return np.array(all_data)


def sfbc_decode_4ant(Y_frames_rx, H_est_list, sc_map, data_idx):
    """Decodifica SFBC de 4 antenas TX (2x2 Alamouti pares)."""
    n_frames = Y_frames_rx.shape[0]
    all_data = []

    for i in range(n_frames):
        Y = Y_frames_rx[i]
        H_list = [H[i] for H in H_est_list]

        decoded = np.zeros(len(data_idx), dtype=complex)

        for j in range(0, len(data_idx) - 3, 4):
            if j + 3 < len(data_idx):
                k0, k1, k2, k3 = data_idx[j:j+4]

                h1_0, h1_1 = H_list[0][k0], H_list[0][k1]
                h2_0, h2_1 = H_list[1][k0], H_list[1][k1]
                h3_2, h3_3 = H_list[2][k2], H_list[2][k3]
                h4_2, h4_3 = H_list[3][k2], H_list[3][k3]

                y0, y1, y2_conj, y3_conj = Y[k0], Y[k1], np.conj(Y[k2]), np.conj(Y[k3])

                G = (np.abs(h1_0) ** 2 + np.abs(h2_0) ** 2 +
                     np.abs(h1_1) ** 2 + np.abs(h2_1) ** 2 +
                     np.abs(h3_2) ** 2 + np.abs(h4_2) ** 2 +
                     np.abs(h3_3) ** 2 + np.abs(h4_3) ** 2 + 1e-12)

                s0 = (np.conj(h1_0) * y0 + h2_1 * y1_conj) / G
                s1 = (np.conj(h2_0) * y0 - h1_1 * y1_conj) / G
                s2 = (np.conj(h3_2) * y2_conj + h4_3 * y3_conj) / G
                s3 = (np.conj(h4_2) * y2_conj - h3_3 * y3_conj) / G

                decoded[j:j+4] = [s0, s1, s2, s3]

        all_data.extend(decoded)

    return np.array(all_data)


# -------------------------------------------------------------------
# MRC en dominio de frecuencia (antes de IDFT para SC-FDMA)
# -------------------------------------------------------------------

def combine_mrc_frequency(Y_frames_list, H_est_all, sc_map):
    """MRC en dominio de frecuencia sin hacer IDFT internamente.

    Retorna símbolos combinados en frecuencia listos para ecualizador ZF
    o de-spreading SC-FDMA. Util para SC-FDMA donde IDFT se hace después.

    X_combined[k] = sum_r(H_r*[k]·Y_r[k]) / sum_r(|H_r[k]|²)
    """
    NR = len(Y_frames_list)
    n_frames = Y_frames_list[0].shape[0]
    Nfft = Y_frames_list[0].shape[1]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)

    all_data = np.zeros(n_frames * n_data, dtype=complex)

    for i in range(n_frames):
        num = np.zeros(Nfft, dtype=complex)
        den = np.zeros(Nfft, dtype=float)

        for r in range(NR):
            H = H_est_all[r][i]
            Y = Y_frames_list[r][i]
            num += np.conj(H) * Y
            den += np.abs(H) ** 2

        X_combined = num / np.maximum(den, 1e-12)
        all_data[i * n_data : (i + 1) * n_data] = X_combined[data_idx]

    return all_data
