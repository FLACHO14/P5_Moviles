import numpy as np

def get_qam_constellation(M):
    m = int(np.sqrt(M))
    levels = np.arange(-(m - 1), m, 2)
    Es = (2 * (m**2 - 1)) / 3
    norm = np.sqrt(Es)
    return levels / norm

def qam_mod(bits, M):
    k = int(np.log2(M))
    levels = get_qam_constellation(M)
    m = int(np.sqrt(M))
    b = bits.reshape(-1, k)
    kb = k // 2
    bi = b[:, :kb].dot(1 << np.arange(kb - 1, -1, -1))
    bq = b[:, kb:].dot(1 << np.arange(kb - 1, -1, -1))
    
    # Gray encoding simplificado
    gi = np.array([x ^ (x >> 1) for x in bi])
    gq = np.array([x ^ (x >> 1) for x in bq])
    return levels[gi] + 1j * levels[gq]

def generate_ofdm_symbol(data_symbols, Nfft, pilot_indices, pilot_value):
    """Crea un símbolo OFDM insertando datos y pilotos."""
    symbol = np.zeros(Nfft, dtype=complex)
    
    # Índices de datos (los que no son pilotos y no son el DC en 0)
    all_indices = np.arange(Nfft)
    data_indices = np.delete(all_indices, pilot_indices)
    
    # Solo llenamos hasta donde alcancen los datos
    n_data = min(len(data_symbols), len(data_indices))
    symbol[pilot_indices] = pilot_value
    symbol[data_indices[:n_data]] = data_symbols[:n_data]
    
    x = np.fft.ifft(symbol)
    return x, n_data

def calculate_papr(x):
    power = np.abs(x)**2
    return 10 * np.log10(np.max(power) / np.mean(power))