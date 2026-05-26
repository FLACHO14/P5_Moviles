import numpy as np
import matplotlib.pyplot as plt
from ofdm_params import DELTA_F, FC

def next_pow2(n):
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

def get_nfft_cp(bw_mhz, cp_mode):
    bw_hz = bw_mhz * 1e6
    N_used = int(round(bw_hz / DELTA_F))
    if N_used < 8:
        N_used = 8
    Nfft = next_pow2(N_used)
    cp_len = Nfft // 8 if cp_mode == "Normal" else Nfft // 4
    return Nfft, cp_len, N_used

def calculate_resource_stats(img_bits, Nfft, M):
    k = int(np.log2(M))
    bits_per_ofdm = Nfft * k
    total_ofdm = int(np.ceil(len(img_bits) / bits_per_ofdm))
    padding = total_ofdm * bits_per_ofdm - len(img_bits)
    return {
        "bits_per_ofdm": bits_per_ofdm,
        "total_ofdm_symbols": total_ofdm,
        "padding_zeros": padding,
    }

def run_analysis(bits_tx, Nfft, cp_len, chan_profile, taps_L, snr_list, n_mc):
    import ofdm_tx, ofdm_channel, ofdm_rx
    mods = {"QPSK":4, "16QAM":16, "64QAM":64}
    ber_data = {}
    ccdf_data = {}
    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = Nfft * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = np.pad(bits_tx, (0, pad), constant_values=0) if pad else bits_tx.copy()
        # CCDF PAPR
        n_syms_papr = 1000
        bits_rand = np.random.randint(0,2, n_syms_papr * Nfft * k).astype(np.uint8)
        s_rand = ofdm_tx.qam_mod(bits_rand, M)
        _, papr_vals = ofdm_tx.ofdm_tx_block(s_rand, Nfft, cp_len)
        papr_sorted = np.sort(papr_vals)
        ccdf = 1.0 - np.arange(1, len(papr_sorted)+1)/len(papr_sorted)
        ccdf_data[mod_name] = (papr_sorted, ccdf)
        # BER
        ber_curve = []
        fs = Nfft * DELTA_F
        for snr_db in snr_list:
            acc = 0.0
            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)
                tx, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len)
                h_impulse = ofdm_channel.get_channel_profile(chan_profile, taps_L)
                rx, _ = ofdm_channel.apply_channel(tx, h_impulse, snr_db, velocity_kmh=0, fs=fs)
                Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
                H_freq = np.fft.fft(h_impulse, n=Nfft)
                n_frames = len(Y)//Nfft
                Xhat = ofdm_rx.equalize(Y, np.tile(H_freq, n_frames))
                bits_hat = ofdm_rx.qam_demod(Xhat, M)[:len(bits_in)]
                acc += np.mean(bits_hat != bits_in)
            ber_curve.append(acc/n_mc)
        ber_data[mod_name] = ber_curve
    return ber_data, ccdf_data

def compute_papr_vs_snr(Nfft, cp_len, M, snr_list, n_mc=10, n_symbols=100):
    import ofdm_tx
    papr_means = []
    papr_stds = []
    for _ in snr_list:
        papr_vals = []
        for _ in range(n_mc):
            bits = np.random.randint(0,2, n_symbols * Nfft * int(np.log2(M))).astype(np.uint8)
            sym = ofdm_tx.qam_mod(bits, M)
            _, papr_list = ofdm_tx.ofdm_tx_block(sym, Nfft, cp_len)
            papr_vals.append(np.max(papr_list))
        papr_means.append(np.mean(papr_vals))
        papr_stds.append(np.std(papr_vals))
    conf_intervals = [1.96 * s / np.sqrt(n_mc) for s in papr_stds]
    return papr_means, conf_intervals

def plot_subcarrier_analysis(ax, Nfft, fs):
    import ofdm_tx
    from scipy import signal
    symbols = np.ones(Nfft)
    tx, _ = ofdm_tx.ofdm_tx_block(symbols, Nfft, 0)
    f_psd, Pxx = signal.welch(tx, fs, nperseg=256)
    ax[0].semilogy(f_psd/1e3, Pxx)
    ax[0].set_title("Densidad espectral de potencia (OFDM)")
    ax[0].set_xlabel("Frecuencia (kHz)")
    ax[0].set_ylabel("PSD (dB)")
    ax[0].grid(True)
    f_sinc = np.linspace(-2*DELTA_F, 2*DELTA_F, 500)
    H_sinc = np.abs(np.sinc(f_sinc/DELTA_F))
    ax[1].plot(f_sinc/1e3, H_sinc)
    ax[1].set_title("Respuesta espectral de una subportadora")
    ax[1].set_xlabel("Frecuencia (kHz)")
    ax[1].set_ylabel("|H(f)|")
    ax[1].grid(True)
    ax[1].axvline(0, color='r', linestyle='--')
    ax[1].axvline(DELTA_F/1e3, color='r', linestyle='--')
    t = np.linspace(0, 2/fs, 200)
    eye = np.real(tx[:len(t)])
    ax[2].plot(t*1e6, eye, 'b', alpha=0.5)
    ax[2].set_title("Diagrama de ojos (señal OFDM)")
    ax[2].set_xlabel("Tiempo (μs)")
    ax[2].set_ylabel("Amplitud")
    ax[2].grid(True)

def plot_channel_response(ax_time, ax_freq, h, fs, Nfft, channel_name):
    t_axis = np.arange(len(h)) / fs * 1e6
    ax_time.stem(t_axis, np.abs(h), basefmt=" ")
    ax_time.set_title(f"Respuesta al impulso del canal ({channel_name})")
    ax_time.set_xlabel("Tiempo (μs)")
    ax_time.set_ylabel("|h(t)|")
    ax_time.grid(True)
    H_f = np.fft.fft(h, n=Nfft)
    f_axis = np.fft.fftfreq(Nfft, 1/fs) / 1e3
    ax_freq.plot(f_axis, 20*np.log10(np.abs(H_f)+1e-12))
    ax_freq.set_title("Respuesta en frecuencia del canal")
    ax_freq.set_xlabel("Frecuencia (kHz)")
    ax_freq.set_ylabel("|H(f)| (dB)")
    ax_freq.grid(True)

def plot_cp_analysis(ax, cp_normal, cp_extended, Nfft, fs):
    labels = ['CP Normal', 'CP Extendido']
    values = [cp_normal, cp_extended]
    colors = ['#3498db', '#9b59b6']
    ax.bar(labels, values, color=colors)
    ax.set_title("Longitud del Prefijo Cíclico (muestras)")
    ax.set_ylabel("Muestras")
    for i, v in enumerate(values):
        ax.text(i, v+0.5, str(v), ha='center', fontweight='bold')
    ax2 = ax.twinx()
    duration_us = np.array(values) / fs * 1e6
    ax2.bar(labels, duration_us, alpha=0.5, color='orange', label="Duración (μs)")
    ax2.set_ylabel("Duración (μs)")
    ax2.legend(loc='upper right')