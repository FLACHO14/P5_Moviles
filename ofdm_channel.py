import numpy as np

def apply_complex_channel(signal, channel_type="Rayleigh", K_factor=3, velocity_kmh=0, fs=1e6):
    L = 8 # Taps del canal
    c = 3e8 # Velocidad de la luz
    fc = 2.1e9 # Frecuencia portadora 4G (2.1 GHz)
    
    # Calcular Doppler
    v_ms = velocity_kmh / 3.6
    fd = (v_ms * fc) / c
    
    # Generar respuesta al impulso
    if channel_type == "Ideal":
        h = np.array([1.0 + 0j])
    elif channel_type == "Rician":
        # Componente LOS + NLOS
        los = np.sqrt(K_factor / (K_factor + 1))
        nlos = np.sqrt(1 / (K_factor + 1)) * (np.random.randn(L) + 1j*np.random.randn(L))/np.sqrt(2)
        h = nlos
        h[0] += los
    else: # Rayleigh
        h = (np.random.randn(L) + 1j*np.random.randn(L))/np.sqrt(2)
    
    h = h / np.linalg.norm(h)
    
    # Aplicar convolución
    y = np.convolve(signal, h, mode='same')
    
    # Simular variación temporal (Doppler) simplificada
    if fd > 0:
        t = np.arange(len(y)) / fs
        doppler_shift = np.exp(1j * 2 * np.pi * fd * t)
        y = y * doppler_shift
        
    return y, h

def add_awgn(signal, snr_db):
    sig_pow = np.mean(np.abs(signal)**2)
    snr_lin = 10**(snr_db/10)
    noise_pow = sig_pow / snr_lin
    noise = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * np.sqrt(noise_pow/2)
    return signal + noise