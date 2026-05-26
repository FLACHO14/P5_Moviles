import numpy as np
from scipy import interpolate

def gray_decode(g):
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def ofdm_rx_block(rx_signal, Nfft, cp_len):
    sym_len = Nfft + cp_len
    n_ofdm = len(rx_signal) // sym_len
    rx = rx_signal[:n_ofdm * sym_len].reshape(n_ofdm, sym_len)
    r_nocp = rx[:, cp_len:]
    Y = np.fft.fft(r_nocp, axis=1)
    return Y.reshape(-1)

def extract_pilots(Y, Nfft, pilot_spacing, pilot_value):
    pilot_mask = np.zeros(Nfft, dtype=bool)
    pilot_mask[::pilot_spacing] = True
    data_mask = ~pilot_mask
    n_frames = len(Y) // Nfft
    Y_frames = Y[:n_frames * Nfft].reshape(n_frames, Nfft)
    H_pilot = np.mean(Y_frames[:, pilot_mask] / pilot_value, axis=0)
    pilot_positions = np.where(pilot_mask)[0]
    return H_pilot, pilot_positions, data_mask

def estimate_channel_from_pilots(H_pilot, pilot_positions, Nfft, method='linear'):
    f = interpolate.interp1d(pilot_positions, H_pilot, kind=method, fill_value='extrapolate')
    all_pos = np.arange(Nfft)
    H_est = f(all_pos)
    return H_est

def equalize(Y, H_est, eps=1e-12):
    return Y / (H_est + eps)

def qam_demod(symbols, M):
    from ofdm_tx import qam_constellation
    k = int(np.log2(M))
    levels = qam_constellation(M)
    kb = k // 2
    I, Q = np.real(symbols), np.imag(symbols)
    idxI = np.argmin(np.abs(I[:, None] - levels[None, :]), axis=1)
    idxQ = np.argmin(np.abs(Q[:, None] - levels[None, :]), axis=1)
    ii = np.array([gray_decode(int(g)) for g in idxI], dtype=int)
    qq = np.array([gray_decode(int(g)) for g in idxQ], dtype=int)
    bi = ((ii[:, None] >> np.arange(kb-1, -1, -1)) & 1).astype(np.uint8)
    bq = ((qq[:, None] >> np.arange(kb-1, -1, -1)) & 1).astype(np.uint8)
    return np.concatenate([bi, bq], axis=1).reshape(-1).astype(np.uint8)