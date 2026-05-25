# ofdm_utils.py
# Utilidades de análisis OFDM: cálculo de parámetros del sistema, simulación
# Monte Carlo de BER, CCDF del PAPR y funciones de graficación.
#
# Cambio respecto a la versión anterior:
# - get_nfft_cp(): Nfft ahora se deriva del ancho de banda y Delta_f (15 kHz)
#   en lugar de usar NFFT_DEFAULT fijo. Esto permite cambiar el BW dinámicamente.
# - run_analysis(): nueva función que ejecuta la simulación Monte Carlo para las
#   tres modulaciones (QPSK/16QAM/64QAM) y devuelve curvas BER vs SNR y CCDF PAPR.
# - plot_cp_comparison(): ahora recibe los valores de CP como argumentos en lugar
#   de usar constantes fijas, para reflejar el Nfft actual de la simulación.

import numpy as np
import matplotlib.pyplot as plt
from ofdm_params import DELTA_F


def next_pow2(n):
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def get_nfft_cp(bw_mhz, cp_mode):
    """Calcula Nfft y cp_len a partir del ancho de banda y el modo de CP.

    Se calcula cuántas subportadoras caben en el BW con espaciado Delta_f=15 kHz
    (estándar 4G LTE), luego se redondea a la siguiente potencia de 2 para usar FFT.
    CP Normal = Nfft/8; CP Extendido = Nfft/4 (mayor protección contra delay spread).
    """
    bw_hz = bw_mhz * 1e6
    N_used = int(round(bw_hz / DELTA_F))
    if N_used < 8:
        N_used = 8
    Nfft = next_pow2(N_used)
    cp_len = Nfft // 8 if cp_mode == "Normal" else Nfft // 4
    return Nfft, cp_len


def calculate_resource_stats(img_bits, Nfft, M):
    """Calcula cuántos símbolos OFDM son necesarios para transmitir img_bits."""
    k = int(np.log2(M))
    bits_per_ofdm = Nfft * k
    total_ofdm = int(np.ceil(len(img_bits) / bits_per_ofdm))
    padding = total_ofdm * bits_per_ofdm - len(img_bits)
    return {
        "bits_per_ofdm": bits_per_ofdm,
        "total_ofdm_symbols": total_ofdm,
        "padding_zeros": padding,
    }


def run_analysis(bits_tx, Nfft, cp_len, chan_type, taps_L, snr_list, n_mc):
    """Simulación Monte Carlo de BER vs SNR y cálculo de CCDF del PAPR.

    Para cada modulación (QPSK, 16QAM, 64QAM):
    - CCDF PAPR: se transmiten 1000 símbolos aleatorios y se ordenan sus PAPR
      para construir la función complementaria de distribución acumulada.
    - BER: para cada valor de SNR se promedian n_mc realizaciones independientes
      del canal, usando los bits de la imagen como datos de prueba.

    Los módulos TX/RX/Canal se importan localmente para evitar dependencias
    circulares a nivel de módulo.
    """
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    ber_data = {}
    ccdf_data = {}

    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = Nfft * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = (
            np.concatenate([bits_tx, np.zeros(pad, np.uint8)]) if pad else bits_tx.copy()
        )

        # --- CCDF del PAPR ---
        # Se usan bits aleatorios (no la imagen) para una estadística imparcial del PAPR
        n_syms_papr = 1000
        bits_rand = np.random.randint(0, 2, n_syms_papr * Nfft * k).astype(np.uint8)
        s_rand = ofdm_tx.qam_mod(bits_rand, M)
        _, papr_vals = ofdm_tx.ofdm_tx_block(s_rand, Nfft, cp_len)
        papr_sorted = np.sort(papr_vals)
        # CCDF: probabilidad de que el PAPR exceda el valor x
        ccdf = 1.0 - np.arange(1, len(papr_sorted) + 1) / len(papr_sorted)
        ccdf_data[mod_name] = (papr_sorted, ccdf)

        # --- BER Monte Carlo ---
        ber_curve = []
        for snr_db in snr_list:
            acc = 0.0
            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)
                tx, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len)

                # Nueva realización del canal para cada iteración Monte Carlo
                if chan_type == "Rayleigh":
                    h = ofdm_channel.multipath_rayleigh_channel(taps_L)
                elif chan_type == "Rician":
                    h = ofdm_channel.multipath_rician_channel(taps_L)
                else:
                    h = None  # Ideal: solo AWGN

                rx, h_used = ofdm_channel.apply_channel(tx, chan_type, snr_db, h)
                Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)

                # Ecualización con la respuesta H conocida (bloque fading constante)
                H = np.fft.fft(h_used, n=Nfft)
                n_ofdm = len(Y) // Nfft
                Xhat = ofdm_rx.equalize(Y, np.tile(H, n_ofdm))

                bits_hat = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
                acc += np.mean(bits_hat != bits_in)
            ber_curve.append(acc / n_mc)
        ber_data[mod_name] = ber_curve

    return ber_data, ccdf_data


def plot_subcarrier_spacing(ax):
    """Visualiza la ortogonalidad entre subportadoras contiguas con Delta_f=15 kHz.
    Cada subportadora tiene un cero exactamente en el pico de sus vecinas.
    """
    f = np.linspace(-30000, 45000, 1000)
    sinc0 = np.abs(np.sinc(f / DELTA_F))
    sinc1 = np.abs(np.sinc((f - DELTA_F) / DELTA_F))
    ax.plot(f / 1000, sinc0, label="Subportadora k", color="blue")
    ax.plot(f / 1000, sinc1, label="Subportadora k+1", color="red")
    ax.axvline(0, color="gray", linestyle="--")
    ax.axvline(DELTA_F / 1000, color="gray", linestyle="--")
    ax.set_title(f"Ortogonalidad con $\\Delta f$ = {DELTA_F/1000} kHz")
    ax.set_xlabel("Frecuencia (kHz)")
    ax.set_ylabel("Amplitud")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_cp_comparison(ax, cp_normal, cp_extended):
    """Compara visualmente el CP Normal (Nfft/8) vs Extendido (Nfft/4).
    Los valores son dinámicos según el Nfft actual de la simulación.
    """
    labels = ["CP Normal", "CP Extendido"]
    values = [cp_normal, cp_extended]
    colors = ["#3498db", "#9b59b6"]
    ax.barh(labels, values, color=colors)
    ax.set_title("Comparativa de Prefijo Cíclico (Muestras)")
    ax.set_xlabel("Número de muestras de guarda")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.02, i, str(v), va="center", fontweight="bold")
