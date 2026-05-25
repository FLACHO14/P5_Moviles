# ofdm_rx.py
# Receptor OFDM: eliminación de prefijo cíclico, FFT, ecualización en frecuencia
# y demodulación QAM con decodificación Gray.
#
# Cambio respecto a la versión anterior: se reemplazó la demodulación basada en
# estimación por pilotos por una ecualización directa usando la respuesta H del
# canal (conocida en simulación). Esto elimina el error de interpolación de pilotos
# que degradaba la imagen recibida.

import numpy as np


def gray_decode(g):
    """Decodificación Gray inversa: convierte índice Gray a índice binario natural.
    Se usa xor iterativo sobre la representación en bits hasta que la máscara sea 0.
    """
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def ofdm_rx_block(rx_signal, Nfft, cp_len):
    """Demodulación OFDM: elimina CP y aplica FFT a cada símbolo recibido.

    Se descartan muestras sobrantes (señal no divisible en símbolos completos)
    para evitar corrupción de datos en el reshape.
    Retorna los símbolos en frecuencia aplanados: shape (n_ofdm * Nfft,).
    """
    sym_len = Nfft + cp_len
    n_ofdm = len(rx_signal) // sym_len
    # Truncar a múltiplo exacto de sym_len antes del reshape
    rx = rx_signal[: n_ofdm * sym_len].reshape(n_ofdm, sym_len)
    r_nocp = rx[:, cp_len:]  # eliminar el prefijo cíclico
    Y = np.fft.fft(r_nocp, axis=1)
    return Y.reshape(-1)


def equalize(Y, H, eps=1e-12):
    """Ecualización por forzado de cero (Zero-Forcing): divide por H[k] en frecuencia.

    El término eps evita división por cero en subportadoras con canal nulo.
    Zero-forcing es óptimo para AWGN; en Rayleigh con desvanecimiento profundo
    puede amplificar el ruido (limitación conocida aceptable en este simulador).
    """
    return Y / (H + eps)


def qam_demod(symbols, M):
    """Demodulación M-QAM: mínima distancia + decodificación Gray → bits.

    La función importa qam_constellation desde ofdm_tx (importación local
    para evitar dependencia circular a nivel de módulo).
    """
    from ofdm_tx import qam_constellation

    k = int(np.log2(M))
    levels = qam_constellation(M)
    kb = k // 2
    I, Q = np.real(symbols), np.imag(symbols)

    # Decisión por mínima distancia (hard decision) en I y Q por separado
    idxI = np.argmin(np.abs(I[:, None] - levels[None, :]), axis=1)
    idxQ = np.argmin(np.abs(Q[:, None] - levels[None, :]), axis=1)

    # Decodificar índices Gray a enteros naturales para obtener los bits
    ii = np.array([gray_decode(int(g)) for g in idxI], dtype=int)
    qq = np.array([gray_decode(int(g)) for g in idxQ], dtype=int)

    # Convertir cada entero a sus kb bits individuales
    bi = ((ii[:, None] >> np.arange(kb - 1, -1, -1)) & 1).astype(np.uint8)
    bq = ((qq[:, None] >> np.arange(kb - 1, -1, -1)) & 1).astype(np.uint8)
    return np.concatenate([bi, bq], axis=1).reshape(-1).astype(np.uint8)
