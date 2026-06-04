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
