import numpy as np

def next_pow2(n):
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

def qam_constellation(M):
    m = int(np.sqrt(M))
    levels = np.arange(-(m-1), m, 2)
    Es = (2 * (m**2 - 1)) / 3
    return levels / np.sqrt(Es)

def qam_mod(bits, M):
    k = int(np.log2(M))
    levels = qam_constellation(M)
    kb = k // 2
    b = bits.reshape(-1, k)
    bi = b[:, :kb].dot(1 << np.arange(kb-1, -1, -1))
    bq = b[:, kb:].dot(1 << np.arange(kb-1, -1, -1))
    gi = np.array([x ^ (x >> 1) for x in bi], dtype=int)
    gq = np.array([x ^ (x >> 1) for x in bq], dtype=int)
    return levels[gi] + 1j * levels[gq]

def insert_pilots(symbols, Nfft, pilot_spacing, pilot_value):
    pad = (-len(symbols)) % Nfft
    if pad:
        symbols = np.pad(symbols, (0, pad), constant_values=0)
    frames = symbols.reshape(-1, Nfft)
    data_mask = np.ones(Nfft, dtype=bool)
    data_mask[::pilot_spacing] = False
    for frame in frames:
        frame[~data_mask] = pilot_value
    return frames.reshape(-1), data_mask

def calculate_papr(x):
    power = np.abs(x)**2
    mean_pow = np.mean(power)
    if mean_pow == 0:
        return 0.0
    return 10 * np.log10(np.max(power) / mean_pow)

def ofdm_tx_block(symbols, Nfft, cp_len):
    pad = (-len(symbols)) % Nfft
    if pad:
        symbols = np.pad(symbols, (0, pad), constant_values=0)
    frames = symbols.reshape(-1, Nfft)
    x = np.fft.ifft(frames, axis=1)
    papr_list = [calculate_papr(xi) for xi in x]
    if cp_len > 0:
        cp = x[:, -cp_len:]
        tx = np.concatenate([cp, x], axis=1)
    else:
        tx = x
    return tx.reshape(-1), papr_list