import numpy as np

def estimate_channel(Y, pilot_indices, pilot_value, Nfft):
    """Estimación de canal por pilotos y expansión por interpolación."""
    # Extraer los valores recibidos en las posiciones de pilotos
    H_pilots = Y[pilot_indices] / pilot_value
    
    # Interpolación lineal para obtener todo el canal H
    all_indices = np.arange(Nfft)
    H_est = np.interp(all_indices, pilot_indices, H_pilots)
    return H_est

def equalize(Y, H_est):
    return Y / (H_est + 1e-12)

def gray_decode(g):
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def qam_demod(symbols, M):
    k = int(np.log2(M))
    m = int(np.sqrt(M))
    levels = np.arange(-(m - 1), m, 2)
    Es = (2 * (m**2 - 1)) / 3
    norm = np.sqrt(Es)
    scaled_levels = levels / norm
    
    I, Q = np.real(symbols), np.imag(symbols)
    idxI = np.argmin(np.abs(I[:, None] - scaled_levels[None, :]), axis=1)
    idxQ = np.argmin(np.abs(Q[:, None] - scaled_levels[None, :]), axis=1)
    
    ii = np.array([gray_decode(x) for x in idxI])
    qq = np.array([gray_decode(x) for x in idxQ])
    
    kb = k // 2
    bi = ((ii[:, None] >> np.arange(kb - 1, -1, -1)) & 1)
    bq = ((qq[:, None] >> np.arange(kb - 1, -1, -1)) & 1)
    return np.concatenate([bi, bq], axis=1).flatten().astype(np.uint8)