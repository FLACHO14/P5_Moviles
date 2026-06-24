# ofdm_beamforming.py
# Módulo de Beamforming (precodificación con CSI ideal en el transmisor).
#
# Implementa transmisión con varias antenas que aplican pesos complejos por
# subportadora para compensar la fase del canal y combinar de forma coherente
# en el receptor. Con NT antenas se obtiene una ganancia de potencia recibida
# de hasta un factor NT respecto a un sistema de una sola antena.
#
# El módulo es independiente y se activa o desactiva mediante un parámetro,
# conservando la estructura de los demás scripts del simulador.

import numpy as np

import ofdm_tx
import ofdm_rx
import ofdm_channel
import ofdm_utils
from ofdm_params import DELTA_F


# -------------------------------------------------------------------
# Modelado de canales con correlación de antenas
# -------------------------------------------------------------------

def generate_beamforming_channels(NT, profile_name, taps_L=None, correlation="low"):
    """Genera NT canales TX para beamforming con la correlación deseada.

    correlation = "low":  canales independientes. El beamforming aporta
                          ganancia de potencia y ganancia de diversidad.
    correlation = "high": canales casi idénticos. El beamforming aporta
                          ganancia de potencia pero no protege frente al
                          desvanecimiento, pues todas las antenas se desvanecen
                          a la vez.

    El grado de correlación se controla con un coeficiente rho que mezcla una
    componente común a todas las antenas con una componente independiente.
    """
    rho = 0.98 if correlation == "high" else 0.0

    h_common = ofdm_channel.get_channel_profile(profile_name, taps_L)
    channels = []
    for _ in range(NT):
        h_ind = ofdm_channel.get_channel_profile(profile_name, taps_L)
        L = max(len(h_common), len(h_ind))
        hc = np.pad(h_common, (0, L - len(h_common)))
        hi = np.pad(h_ind, (0, L - len(h_ind)))
        h = np.sqrt(rho) * hc + np.sqrt(1.0 - rho) * hi
        norm = np.sqrt(np.sum(np.abs(h) ** 2))
        if norm > 0:
            h = h / norm
        channels.append(h)
    return channels


# -------------------------------------------------------------------
# Precodificación (cálculo de pesos por subportadora)
# -------------------------------------------------------------------

def beamforming_weights(H_freq_list):
    """Pesos de precodificación por subportadora con CSI ideal.

    Para cada subportadora k y antena i, el peso compensa la fase del canal
    y pondera por su magnitud, de modo que las contribuciones se sumen de
    forma coherente en el receptor:

        v_i[k] = H_i*[k] / sqrt( sum_j |H_j[k]|^2 )

    La normalización mantiene la potencia transmitida total por subportadora
    constante (sum_i |v_i[k]|^2 = 1), por lo que la mejora proviene de la
    combinación coherente y no de transmitir más potencia.

    Retorna una lista de NT vectores de pesos, uno por antena.
    """
    NT = len(H_freq_list)
    Nfft = len(H_freq_list[0])
    denom = np.zeros(Nfft, dtype=float)
    for H in H_freq_list:
        denom += np.abs(H) ** 2
    denom = np.sqrt(np.maximum(denom, 1e-12))

    weights = [np.conj(H) / denom for H in H_freq_list]
    return weights


def beamforming_tx_block(data_symbols, Nfft, cp_len, sc_map, pilot_value, channels):
    """Transmisión OFDM con precodificación de beamforming por subportadora.

    El transmisor conoce el canal (CSI ideal). Calcula la respuesta en
    frecuencia de cada antena, obtiene los pesos por subportadora y aplica
    la precodificación tanto a los datos como a los pilotos. Cada antena
    genera su propia señal temporal.

    Returns
    -------
    tx_signals : list[ndarray]
        Señal temporal de cada una de las NT antenas.
    papr_list : list[float]
        PAPR promedio de las antenas por símbolo OFDM.
    n_ofdm : int
        Número de símbolos OFDM.
    H_eff : ndarray
        Canal efectivo por subportadora tras la precodificación, util para
        analizar el relleno de los huecos de desvanecimiento.
    """
    NT = len(channels)
    H_freq_list = [np.fft.fft(h, Nfft) for h in channels]
    weights = beamforming_weights(H_freq_list)

    # Canal efectivo: sum_i H_i[k] v_i[k] = sqrt(sum_i |H_i[k]|^2)
    H_eff = np.zeros(Nfft, dtype=complex)
    for H, v in zip(H_freq_list, weights):
        H_eff += H * v

    n_data = sc_map["n_data"]
    n_ofdm = int(np.ceil(len(data_symbols) / n_data))
    total = n_ofdm * n_data
    pad = total - len(data_symbols)
    if pad > 0:
        data_symbols = np.concatenate([data_symbols, np.zeros(pad, dtype=complex)])
    data_frames = data_symbols.reshape(n_ofdm, n_data)

    data_idx = sc_map["data_indices"]
    pilot_idx = sc_map["pilot_indices"]
    sym_len = Nfft + cp_len

    tx_signals = [np.zeros(n_ofdm * sym_len, dtype=complex) for _ in range(NT)]
    papr_list = []

    for i in range(n_ofdm):
        # Símbolo OFDM en frecuencia sin precodificar (datos + pilotos)
        S = np.zeros(Nfft, dtype=complex)
        S[data_idx] = data_frames[i]
        S[pilot_idx] = pilot_value

        paprs = []
        for a in range(NT):
            X = weights[a] * S  # precodificación por subportadora
            x = np.fft.ifft(X)
            paprs.append(ofdm_tx.calculate_papr(x))

            offset = i * sym_len
            if cp_len > 0:
                tx_signals[a][offset : offset + cp_len] = x[-cp_len:]
                tx_signals[a][offset + cp_len : offset + sym_len] = x
            else:
                tx_signals[a][offset : offset + sym_len] = x
        papr_list.append(float(np.mean(paprs)))

    return tx_signals, papr_list, n_ofdm, H_eff


def apply_channel_beamforming(tx_signals, channels, snr_db, velocity_kmh=0, fs=1.92e6):
    """Canal MISO para beamforming con ruido referido a la potencia transmitida.

    A diferencia de la cadena SFBC, aquí el ruido se calibra respecto a la
    potencia TRANSMITIDA (no a la recibida), de modo que la ganancia de
    potencia que aporta el beamforming en el receptor sea visible como mejora
    de relación señal a ruido. El receptor suma las contribuciones de las NT
    antenas a través de sus canales.
    """
    len_sig = len(tx_signals[0])
    rx = np.zeros(len_sig, dtype=complex)
    for tx, h in zip(tx_signals, channels):
        y = np.convolve(tx, h, mode="full")[:len_sig]
        y = ofdm_channel.apply_doppler(y, velocity_kmh, fs)
        rx += y

    # Potencia transmitida total (suma de antenas). Sirve de referencia fija
    # para el ruido, independiente de la ganancia de array del beamforming.
    tx_pow = sum(np.mean(np.abs(tx) ** 2) for tx in tx_signals)
    if tx_pow == 0:
        tx_pow = 1.0
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = tx_pow / snr_lin
    noise = (np.random.randn(len_sig) + 1j * np.random.randn(len_sig)) * np.sqrt(noise_pow / 2)

    return rx + noise


# -------------------------------------------------------------------
# Análisis Monte Carlo: SISO vs SFBC vs Beamforming
# -------------------------------------------------------------------

BF_COLORS = {"SISO": "#7f8c8d", "MISO-SFBC": "#e67e22", "Beamforming": "#8e44ad"}
BF_MARKERS = {"SISO": "o", "MISO-SFBC": "D", "Beamforming": "s"}


def run_analysis_beamforming(
    bits_tx, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_list, n_mc, velocity_kmh, NT, correlation="low",
):
    """Monte Carlo BER con IC 95%: SISO vs MISO-SFBC vs Beamforming.

    SISO usa una antena; SFBC usa dos antenas con código Alamouti; Beamforming
    usa NT antenas con precodificación por CSI ideal. La curva de beamforming
    evidencia la ganancia de potencia de hasta un factor NT respecto a SISO.
    """
    n_data = sc_map["n_data"]
    fs = Nfft * DELTA_F

    pilot_idx = sc_map["pilot_indices"]
    sc_a1 = dict(sc_map); sc_a1["pilot_indices"] = pilot_idx[::2]
    sc_a2 = dict(sc_map); sc_a2["pilot_indices"] = pilot_idx[1::2]
    data_idx = sc_map["data_indices"]

    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    techniques = ["SISO", "MISO-SFBC", "Beamforming"]
    results = {t: {} for t in techniques}

    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = n_data * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

        means = {t: [] for t in techniques}
        cis = {t: [] for t in techniques}

        for snr_db in snr_list:
            iters = {t: [] for t in techniques}
            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)

                # SISO
                tx_siso, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
                h = ofdm_channel.get_channel_profile(chan_profile, taps_L)
                rx, _ = ofdm_channel.apply_channel(tx_siso, h, snr_db, velocity_kmh, fs)
                Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
                Xs, _ = ofdm_rx.equalize_with_pilots(Y, sc_map, pilot_value, Nfft)
                iters["SISO"].append(_ber(Xs, M, bits_in))

                # MISO-SFBC (2 TX)
                tx1, tx2, _, _, _, _ = ofdm_tx.sfbc_tx_2ant(s, Nfft, cp_len, sc_map, pilot_value)
                h_miso = ofdm_channel.generate_miso_channels(2, chan_profile, taps_L)
                rxm, _ = ofdm_channel.apply_channel_miso([tx1, tx2], h_miso, snr_db, velocity_kmh, fs)
                Ym = ofdm_rx.ofdm_rx_block(rxm, Nfft, cp_len)
                H1 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a1, pilot_value, Nfft)
                H2 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a2, pilot_value, Nfft)
                Xsf = ofdm_rx.sfbc_decode_2ant(Ym, H1, H2, sc_map, data_idx)
                iters["MISO-SFBC"].append(_ber(Xsf, M, bits_in))

                # Beamforming (NT TX, CSI ideal)
                ch_bf = generate_beamforming_channels(NT, chan_profile, taps_L, correlation)
                tx_bf, _, _, _ = beamforming_tx_block(s, Nfft, cp_len, sc_map, pilot_value, ch_bf)
                rxb = apply_channel_beamforming(tx_bf, ch_bf, snr_db, velocity_kmh, fs)
                Yb = ofdm_rx.ofdm_rx_block(rxb, Nfft, cp_len)
                Xb, _ = ofdm_rx.equalize_with_pilots(Yb, sc_map, pilot_value, Nfft)
                iters["Beamforming"].append(_ber(Xb, M, bits_in))

            for t in techniques:
                m = np.mean(iters[t])
                sd = np.std(iters[t], ddof=1) if n_mc > 1 else 0
                means[t].append(m)
                cis[t].append(1.96 * sd / np.sqrt(n_mc))

        for t in techniques:
            results[t][mod_name] = {"mean": means[t], "ci": cis[t]}

    return results


def _ber(Xhat, M, bits_in):
    bh = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
    if len(bh) < len(bits_in):
        bh = np.pad(bh, (0, len(bits_in) - len(bh)))
    return float(np.mean(bh != bits_in))


# -------------------------------------------------------------------
# Datos de potencia instantánea recibida (relleno de huecos)
# -------------------------------------------------------------------

def generate_beamforming_power_data(
    Nfft, cp_len, sc_map, pilot_value, chan_profile, taps_L,
    NT, correlation="low",
):
    """Potencia del canal por subportadora: una antena vs beamforming.

    Retorna |H_1[k]|^2 de una sola antena y |H_eff[k]|^2 del canal efectivo
    tras la precodificación, para mostrar cómo el beamforming rellena los
    huecos de desvanecimiento y eleva la potencia recibida.
    """
    channels = generate_beamforming_channels(NT, chan_profile, taps_L, correlation)
    H_list = [np.fft.fft(h, Nfft) for h in channels]
    weights = beamforming_weights(H_list)

    H_eff = np.zeros(Nfft, dtype=complex)
    for H, v in zip(H_list, weights):
        H_eff += H * v

    H_single = np.abs(H_list[0]) ** 2
    H_beam = np.abs(H_eff) ** 2
    return H_single, H_beam


# -------------------------------------------------------------------
# Transmisión de imagen: SISO vs SFBC vs Beamforming a SNR bajo
# -------------------------------------------------------------------

def transmit_image_beamforming(
    bits_tx, img_arr, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_db, velocity_kmh, M, NT, correlation="low",
):
    """Transmite la misma imagen por SISO, MISO-SFBC y Beamforming.

    Pensada para SNR bajo, donde el beamforming entrega una imagen más limpia
    gracias a la ganancia de antena en la dirección del receptor. La
    reconstrucción de bits a píxeles es robusta ante errores.
    """
    fs = Nfft * DELTA_F
    n_bits = len(bits_tx)
    k = int(np.log2(M))
    n_data = sc_map["n_data"]
    bits_per_ofdm = n_data * k
    n_ofdm_needed = int(np.ceil(n_bits / bits_per_ofdm))
    total_cap = n_ofdm_needed * bits_per_ofdm
    pad = total_cap - n_bits
    bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

    pilot_idx = sc_map["pilot_indices"]
    sc_a1 = dict(sc_map); sc_a1["pilot_indices"] = pilot_idx[::2]
    sc_a2 = dict(sc_map); sc_a2["pilot_indices"] = pilot_idx[1::2]
    data_idx = sc_map["data_indices"]
    img_shape = img_arr.shape

    s = ofdm_tx.qam_mod(bits_in, M)

    # SISO
    tx_siso, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
    h = ofdm_channel.get_channel_profile(chan_profile, taps_L)
    rx, _ = ofdm_channel.apply_channel(tx_siso, h, snr_db, velocity_kmh, fs)
    Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
    Xs, _ = ofdm_rx.equalize_with_pilots(Y, sc_map, pilot_value, Nfft)
    bits_siso = ofdm_rx.qam_demod(Xs, M)

    # MISO-SFBC
    tx1, tx2, _, _, _, _ = ofdm_tx.sfbc_tx_2ant(s, Nfft, cp_len, sc_map, pilot_value)
    h_miso = ofdm_channel.generate_miso_channels(2, chan_profile, taps_L)
    rxm, _ = ofdm_channel.apply_channel_miso([tx1, tx2], h_miso, snr_db, velocity_kmh, fs)
    Ym = ofdm_rx.ofdm_rx_block(rxm, Nfft, cp_len)
    H1 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a1, pilot_value, Nfft)
    H2 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a2, pilot_value, Nfft)
    Xsf = ofdm_rx.sfbc_decode_2ant(Ym, H1, H2, sc_map, data_idx)
    bits_sfbc = ofdm_rx.qam_demod(Xsf, M)

    # Beamforming
    ch_bf = generate_beamforming_channels(NT, chan_profile, taps_L, correlation)
    tx_bf, _, _, H_eff = beamforming_tx_block(s, Nfft, cp_len, sc_map, pilot_value, ch_bf)
    rxb = apply_channel_beamforming(tx_bf, ch_bf, snr_db, velocity_kmh, fs)
    Yb = ofdm_rx.ofdm_rx_block(rxb, Nfft, cp_len)
    Xb, _ = ofdm_rx.equalize_with_pilots(Yb, sc_map, pilot_value, Nfft)
    bits_bf = ofdm_rx.qam_demod(Xb, M)

    img_siso = ofdm_utils._bits_to_image(bits_siso, n_bits, total_cap, img_shape)
    img_sfbc = ofdm_utils._bits_to_image(bits_sfbc, n_bits, total_cap, img_shape)
    img_bf = ofdm_utils._bits_to_image(bits_bf, n_bits, total_cap, img_shape)

    mse_siso, psnr_siso = ofdm_utils._img_metrics(img_arr, img_siso)
    mse_sfbc, psnr_sfbc = ofdm_utils._img_metrics(img_arr, img_sfbc)
    mse_bf, psnr_bf = ofdm_utils._img_metrics(img_arr, img_bf)

    def ber(bits_rx):
        b = np.asarray(bits_rx).astype(np.uint8).ravel()
        if len(b) < total_cap:
            b = np.pad(b, (0, total_cap - len(b)))
        return float(np.mean(b[:n_bits] != bits_tx))

    return {
        "img_orig": img_arr,
        "img_siso": img_siso, "psnr_siso": psnr_siso, "mse_siso": mse_siso, "ber_siso": ber(bits_siso),
        "img_sfbc": img_sfbc, "psnr_sfbc": psnr_sfbc, "mse_sfbc": mse_sfbc, "ber_sfbc": ber(bits_sfbc),
        "img_bf": img_bf, "psnr_bf": psnr_bf, "mse_bf": mse_bf, "ber_bf": ber(bits_bf),
        "snr_db": snr_db, "NT": NT, "correlation": correlation,
    }


# -------------------------------------------------------------------
# Visualizaciones
# -------------------------------------------------------------------

def plot_beamforming_ber(ax, snr_list, bf_results, mod_name, NT):
    """BER vs SNR para SISO vs MISO-SFBC vs Beamforming con IC 95%."""
    snr_arr = np.array(snr_list)
    for tech in ["SISO", "MISO-SFBC", "Beamforming"]:
        if mod_name not in bf_results[tech]:
            continue
        means = np.array(bf_results[tech][mod_name]["mean"])
        cis = np.array(bf_results[tech][mod_name]["ci"])
        color = BF_COLORS[tech]
        marker = BF_MARKERS[tech]
        mask = means > 0
        label = tech if tech != "Beamforming" else f"Beamforming (NT={NT})"
        if np.any(mask):
            ax.semilogy(snr_arr[mask], means[mask], marker=marker, color=color,
                        label=label, markersize=5)
            upper = means[mask] + cis[mask]
            lower = np.maximum(means[mask] - cis[mask], 1e-10)
            ax.fill_between(snr_arr[mask], lower, upper, alpha=0.15, color=color)
        else:
            ax.semilogy([], [], marker=marker, color=color, label=f"{label} (BER=0)")

    ax.set_title(f"{mod_name} — BER vs SNR + IC 95%")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


def plot_beamforming_power(ax, H_single, H_beam, Nfft, NT, correlation):
    """Potencia del canal por subportadora: 1 antena vs beamforming.

    Muestra cómo el beamforming eleva la potencia recibida y rellena los
    huecos de desvanecimiento que sufre un sistema de una sola antena.
    """
    active = np.where((H_single > 1e-10) & (H_beam > 1e-10))[0]
    if len(active) == 0:
        ax.text(0.5, 0.5, "Sin datos activos", ha="center", va="center",
                transform=ax.transAxes)
        return

    sc_centered = active - Nfft // 2
    H1_dB = 10 * np.log10(H_single[active] + 1e-20)
    Hb_dB = 10 * np.log10(H_beam[active] + 1e-20)

    from scipy.ndimage import uniform_filter1d
    win = max(3, len(active) // 80)
    H1_s = uniform_filter1d(H1_dB, size=win)
    Hb_s = uniform_filter1d(Hb_dB, size=win)

    ax.fill_between(sc_centered, H1_s, alpha=0.15, color="#e74c3c")
    ax.plot(sc_centered, H1_s, color="#e74c3c", alpha=0.85, linewidth=0.9,
            label="1 antena (SISO)")
    ax.fill_between(sc_centered, Hb_s, alpha=0.15, color="#8e44ad")
    ax.plot(sc_centered, Hb_s, color="#8e44ad", alpha=0.9, linewidth=1.3,
            label=f"Beamforming (NT={NT})")

    gain = np.mean(Hb_dB) - np.mean(H1_dB)
    corr_txt = "alta" if correlation == "high" else "baja"
    ax.annotate(
        f"Ganancia de potencia: {gain:.1f} dB\nCorrelación {corr_txt}  |  factor NT={NT}",
        xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_title(f"Potencia Recibida: relleno de huecos (NT={NT})")
    ax.set_xlabel("Subportadora (centrada en DC)")
    ax.set_ylabel("|H|² (dB)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
