# ofdm_spatial_mux.py
# Multiplexación Espacial MIMO 2x2 con detector MMSE y CSI perfecto.
#
# A diferencia de SFBC (que busca redundancia), aquí el objetivo es DUPLICAR
# la tasa de datos: el flujo se divide en dos capas independientes que se
# transmiten a la vez por dos antenas sobre las mismas subportadoras. El
# receptor separa las capas con un detector lineal MMSE por subportadora.

import numpy as np

import ofdm_tx
import ofdm_rx
import ofdm_channel
import ofdm_utils
from ofdm_params import DELTA_F


# -------------------------------------------------------------------
# 1. Transmisor: división de flujos por intercalado + OFDM independiente
# -------------------------------------------------------------------

def split_streams(symbols):
    """Divide el flujo de símbolos en dos capas por intercalado.

    Los símbolos de índice par forman la capa 1 (antena 1) y los de índice
    impar la capa 2 (antena 2). De este modo ambas antenas transportan
    información distinta y la tasa efectiva se duplica.
    """
    return symbols[0::2], symbols[1::2]


def mimo2x2_tx(bits_in, Nfft, cp_len, sc_map, M):
    """Transmisor de multiplexación espacial 2x2.

    Modula los bits, divide el flujo en dos capas intercaladas y modula cada
    capa con su propia IFFT y prefijo cíclico. Cada antena emite a la mitad de
    la potencia para que la comparación con un sistema de una sola antena sea
    justa.

    Returns
    -------
    x1, x2 : ndarray  -> señales temporales de las antenas 1 y 2.
    n_ofdm : int      -> símbolos OFDM por capa.
    s1, s2 : ndarray  -> símbolos transmitidos por capa (referencia).
    """
    n_data = sc_map["n_data"]
    k = int(np.log2(M))

    # Asegurar un número de símbolos múltiplo de 2*n_data para un troceado limpio
    bits_per_block = 2 * n_data * k
    pad = (-len(bits_in)) % bits_per_block
    if pad:
        bits_in = np.pad(bits_in, (0, pad))

    symbols = ofdm_tx.qam_mod(bits_in, M)
    s1, s2 = split_streams(symbols)  # capas intercaladas

    n_ofdm = len(s1) // n_data
    data_idx = sc_map["data_indices"]
    scale = 1.0 / np.sqrt(2.0)
    sym_len = Nfft + cp_len

    x1 = np.zeros(n_ofdm * sym_len, dtype=complex)
    x2 = np.zeros(n_ofdm * sym_len, dtype=complex)

    f1 = s1[: n_ofdm * n_data].reshape(n_ofdm, n_data)
    f2 = s2[: n_ofdm * n_data].reshape(n_ofdm, n_data)

    for i in range(n_ofdm):
        X1 = np.zeros(Nfft, dtype=complex)
        X2 = np.zeros(Nfft, dtype=complex)
        X1[data_idx] = f1[i] * scale
        X2[data_idx] = f2[i] * scale
        a1 = np.fft.ifft(X1)
        a2 = np.fft.ifft(X2)
        off = i * sym_len
        if cp_len > 0:
            x1[off:off + cp_len] = a1[-cp_len:]
            x1[off + cp_len:off + sym_len] = a1
            x2[off:off + cp_len] = a2[-cp_len:]
            x2[off + cp_len:off + sym_len] = a2
        else:
            x1[off:off + sym_len] = a1
            x2[off:off + sym_len] = a2

    return x1, x2, n_ofdm, s1[: n_ofdm * n_data], s2[: n_ofdm * n_data]


# -------------------------------------------------------------------
# 2. Canal MIMO 2x2 (cuatro trayectos Rayleigh independientes)
# -------------------------------------------------------------------

def generate_channel_matrix(profile, taps_L=2):
    """Genera las cuatro respuestas impulsivas h11, h12, h21, h22.

    Cada una es un canal Rayleigh independiente que representa el trayecto
    entre una antena transmisora y una antena receptora.
    """
    return [ofdm_channel.get_channel_profile(profile, taps_L) for _ in range(4)]


def apply_mimo2x2(x1, x2, channels, snr_db, velocity_kmh=0, fs=1.92e6):
    """Aplica el canal MIMO 2x2 con convolución causal y ruido AWGN.

    y1 = h11 * x1 + h12 * x2 + n1
    y2 = h21 * x1 + h22 * x2 + n2

    El canal se trata como estático durante la trama, coherente con la
    hipótesis de CSI perfecto obtenida de la FFT de las respuestas
    impulsivas. Devuelve las dos señales recibidas y la varianza de ruido,
    que el detector MMSE necesita para regularizar la inversión.
    """
    h11, h12, h21, h22 = channels
    L = len(x1)
    y1 = np.convolve(x1, h11, mode="full")[:L] + np.convolve(x2, h12, mode="full")[:L]
    y2 = np.convolve(x1, h21, mode="full")[:L] + np.convolve(x2, h22, mode="full")[:L]

    p = (np.mean(np.abs(y1) ** 2) + np.mean(np.abs(y2) ** 2)) / 2 or 1.0
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = p / snr_lin
    n1 = (np.random.randn(L) + 1j * np.random.randn(L)) * np.sqrt(noise_pow / 2)
    n2 = (np.random.randn(L) + 1j * np.random.randn(L)) * np.sqrt(noise_pow / 2)
    return y1 + n1, y2 + n2, noise_pow


# -------------------------------------------------------------------
# 3. Receptor: detector MMSE por subportadora con CSI perfecto
# -------------------------------------------------------------------

def mmse_detect(Y1, Y2, channels, Nfft, sc_map, noise_pow):
    """Detector lineal MMSE por subportadora.

    Con CSI perfecto, la matriz de canal en cada subportadora se obtiene de la
    FFT de las respuestas impulsivas. Los pesos se calculan como
    W = (H^H H + sigma^2 I)^{-1} H^H, que mitiga la interferencia entre
    antenas de forma más efectiva que Zero-Forcing al equilibrar el realce de
    ruido en las subportadoras mal condicionadas.

    Returns
    -------
    shat1, shat2 : símbolos estimados de cada capa sobre las subportadoras de
    datos, y los símbolos crudos recibidos en cada antena para la
    visualización de constelaciones.
    """
    h11, h12, h21, h22 = channels
    H11 = np.fft.fft(h11, Nfft); H12 = np.fft.fft(h12, Nfft)
    H21 = np.fft.fft(h21, Nfft); H22 = np.fft.fft(h22, Nfft)

    n_frames = Y1.shape[0]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)
    I2 = np.eye(2, dtype=complex)

    shat1 = np.zeros(n_frames * n_data, dtype=complex)
    shat2 = np.zeros(n_frames * n_data, dtype=complex)
    raw1 = np.zeros(n_frames * n_data, dtype=complex)
    raw2 = np.zeros(n_frames * n_data, dtype=complex)

    for i in range(n_frames):
        for di, k in enumerate(data_idx):
            H = np.array([[H11[k], H12[k]], [H21[k], H22[k]]], dtype=complex)
            r = np.array([Y1[i][k], Y2[i][k]], dtype=complex)
            W = np.linalg.inv(H.conj().T @ H + noise_pow * I2) @ H.conj().T
            s = W @ r
            # MMSE no sesgado: se divide cada capa por su factor de sesgo
            # (diagonal de W·H) para corregir la escala antes de la decisión,
            # lo que es crítico en modulaciones con varios niveles de amplitud.
            G = W @ H
            g0 = G[0, 0] if abs(G[0, 0]) > 1e-9 else 1e-9
            g1 = G[1, 1] if abs(G[1, 1]) > 1e-9 else 1e-9
            idx = i * n_data + di
            shat1[idx] = s[0] / g0; shat2[idx] = s[1] / g1
            raw1[idx] = r[0]; raw2[idx] = r[1]

    # Compensar el reparto de potencia 1/sqrt(2) para volver a la escala QAM
    sc = np.sqrt(2.0)
    return shat1 * sc, shat2 * sc, raw1, raw2


def mimo2x2_receive(y1, y2, channels, Nfft, cp_len, sc_map, M, noise_pow):
    """Cadena de recepción completa: FFT, detección MMSE y de-intercalado.

    Devuelve los bits recuperados y datos de constelación para la
    visualización.
    """
    Y1 = ofdm_rx.ofdm_rx_block(y1, Nfft, cp_len)
    Y2 = ofdm_rx.ofdm_rx_block(y2, Nfft, cp_len)
    shat1, shat2, raw1, raw2 = mmse_detect(Y1, Y2, channels, Nfft, sc_map, noise_pow)

    # De-intercalado: par <- capa 1, impar <- capa 2
    n = min(len(shat1), len(shat2))
    merged = np.empty(2 * n, dtype=complex)
    merged[0::2] = shat1[:n]
    merged[1::2] = shat2[:n]
    bits_rx = ofdm_rx.qam_demod(merged, M)
    return bits_rx, {"raw1": raw1, "raw2": raw2, "eq1": shat1, "eq2": shat2}


# -------------------------------------------------------------------
# 4. Cadenas de alto nivel: BER, throughput, imagen
# -------------------------------------------------------------------

def _siso_ber(bits_in, Nfft, cp_len, sc_map, pilot_value, M, profile, taps_L,
              snr_db, velocity, fs):
    s = ofdm_tx.qam_mod(bits_in, M)
    tx, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
    h = ofdm_channel.get_channel_profile(profile, taps_L)
    rx, _ = ofdm_channel.apply_channel(tx, h, snr_db, velocity, fs)
    Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
    X, _ = ofdm_rx.equalize_with_pilots(Y, sc_map, pilot_value, Nfft)
    bh = ofdm_rx.qam_demod(X, M)[: len(bits_in)]
    if len(bh) < len(bits_in):
        bh = np.pad(bh, (0, len(bits_in) - len(bh)))
    return float(np.mean(bh != bits_in))


def mimo2x2_ber(bits_in, Nfft, cp_len, sc_map, M, profile, taps_L,
                snr_db, velocity, fs):
    x1, x2, _, _, _ = mimo2x2_tx(bits_in, Nfft, cp_len, sc_map, M)
    ch = generate_channel_matrix(profile, taps_L)
    y1, y2, npow = apply_mimo2x2(x1, x2, ch, snr_db, velocity, fs)
    bits_rx, _ = mimo2x2_receive(y1, y2, ch, Nfft, cp_len, sc_map, M, npow)
    n = min(len(bits_rx), len(bits_in))
    return float(np.mean(bits_rx[:n] != bits_in[:n]))


def run_ber_siso_vs_mimo(bits_tx, Nfft, cp_len, sc_map, pilot_value, M,
                         profile, taps_L, snr_list, n_mc, velocity):
    """BER vs SNR para SISO frente a MIMO 2x2 con MMSE, con IC 95%."""
    fs = Nfft * DELTA_F
    n_data = sc_map["n_data"]
    k = int(np.log2(M))
    pad = (-len(bits_tx)) % (n_data * k)
    bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

    res = {"SISO": {"mean": [], "ci": []}, "MIMO 2x2 (MMSE)": {"mean": [], "ci": []}}
    for snr in snr_list:
        bs, bm = [], []
        for _ in range(n_mc):
            bs.append(_siso_ber(bits_in, Nfft, cp_len, sc_map, pilot_value, M,
                                profile, taps_L, snr, velocity, fs))
            bm.append(mimo2x2_ber(bits_in, Nfft, cp_len, sc_map, M,
                                  profile, taps_L, snr, velocity, fs))
        for key, vals in (("SISO", bs), ("MIMO 2x2 (MMSE)", bm)):
            m = np.mean(vals); sd = np.std(vals, ddof=1) if n_mc > 1 else 0
            res[key]["mean"].append(m)
            res[key]["ci"].append(1.96 * sd / np.sqrt(n_mc))
    return res


def throughput_data(bits_tx, Nfft, cp_len, sc_map, M):
    """Tiempo de transmisión y throughput de SISO frente a MIMO 2x2.

    Para la misma cantidad de bits, MIMO 2x2 emplea la mitad de símbolos OFDM
    porque transmite dos capas a la vez, lo que reduce a la mitad el tiempo de
    transmisión y duplica el throughput.
    """
    n_data = sc_map["n_data"]
    k = int(np.log2(M))
    fs = Nfft * DELTA_F
    sym_time = (Nfft + cp_len) / fs

    n_bits = len(bits_tx)
    bits_ofdm = n_data * k
    n_siso = int(np.ceil(n_bits / bits_ofdm))
    n_mimo = int(np.ceil(n_bits / (2 * bits_ofdm)))

    t_siso = n_siso * sym_time
    t_mimo = n_mimo * sym_time
    return {
        "n_bits": n_bits,
        "n_ofdm_siso": n_siso, "n_ofdm_mimo": n_mimo,
        "t_siso": t_siso, "t_mimo": t_mimo,
        "thr_siso": n_bits / t_siso, "thr_mimo": n_bits / t_mimo,
        "sym_time": sym_time,
    }


def transmit_image_mimo(bits_tx, img_arr, Nfft, cp_len, sc_map, M,
                        profile, taps_L, snr_db, velocity):
    """Reconstruye la imagen transmitida por MIMO 2x2 a una SNR dada."""
    fs = Nfft * DELTA_F
    n_bits = len(bits_tx)
    k = int(np.log2(M))
    n_data = sc_map["n_data"]
    bpo = n_data * k
    n_need = int(np.ceil(n_bits / bpo))
    total_cap = n_need * bpo
    bits_in = np.pad(bits_tx, (0, total_cap - n_bits)) if total_cap > n_bits else bits_tx.copy()

    x1, x2, _, _, _ = mimo2x2_tx(bits_in, Nfft, cp_len, sc_map, M)
    ch = generate_channel_matrix(profile, taps_L)
    y1, y2, npow = apply_mimo2x2(x1, x2, ch, snr_db, velocity, fs)
    bits_rx, const = mimo2x2_receive(y1, y2, ch, Nfft, cp_len, sc_map, M, npow)

    img = ofdm_utils._bits_to_image(bits_rx, n_bits, total_cap, img_arr.shape)
    mse, psnr = ofdm_utils._img_metrics(img_arr, img)
    return {"img": img, "psnr": psnr, "mse": mse, "snr_db": snr_db, "const": const}


# -------------------------------------------------------------------
# 5. Visualizaciones
# -------------------------------------------------------------------

def plot_mimo_constellations(axes, const, M):
    """Cuatro paneles: RX antena 1, RX antena 2, ambas, y salida MMSE.

    Las dos primeras muestran nubes dispersas por la mezcla de señales en el
    aire; la última muestra los puntos de la modulación ya separados por el
    detector MMSE.
    """
    raw1 = const["raw1"]; raw2 = const["raw2"]
    eq = np.concatenate([const["eq1"], const["eq2"]]) * np.sqrt(2.0)

    def scat(ax, data, title, color, lim):
        ax.scatter(np.real(data), np.imag(data), s=4, alpha=0.3, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.3); ax.set_aspect("equal", adjustable="box")

    lim_raw = 3 * (np.std(np.abs(raw1)) + 1e-9)
    scat(axes[0], raw1, "RX Antena 1 (mezcla)", "#c0392b", lim_raw)
    scat(axes[1], raw2, "RX Antena 2 (mezcla)", "#e67e22", lim_raw)
    axes[2].scatter(np.real(raw1), np.imag(raw1), s=4, alpha=0.25, color="#c0392b", label="Ant 1")
    axes[2].scatter(np.real(raw2), np.imag(raw2), s=4, alpha=0.25, color="#e67e22", label="Ant 2")
    axes[2].set_title("Ambas antenas (superpuestas)", fontsize=10)
    axes[2].set_xlabel("I"); axes[2].set_ylabel("Q")
    axes[2].set_xlim(-lim_raw, lim_raw); axes[2].set_ylim(-lim_raw, lim_raw)
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect("equal", adjustable="box")
    scat(axes[3], eq, "Salida ecualizada MMSE", "#2980b9", 1.8)


def plot_ber_siso_vs_mimo(ax, res, snr_list, M_name):
    """BER vs SNR de SISO frente a MIMO 2x2."""
    snr = np.array(snr_list)
    colors = {"SISO": "#7f8c8d", "MIMO 2x2 (MMSE)": "#c0392b"}
    markers = {"SISO": "o", "MIMO 2x2 (MMSE)": "P"}
    for t, d in res.items():
        m = np.array(d["mean"]); ci = np.array(d["ci"]); mask = m > 0
        if np.any(mask):
            ax.semilogy(snr[mask], m[mask], marker=markers[t], color=colors[t],
                        label=t, markersize=5)
            ax.fill_between(snr[mask], np.maximum(m[mask] - ci[mask], 1e-10),
                            m[mask] + ci[mask], alpha=0.15, color=colors[t])
        else:
            ax.semilogy([], [], marker=markers[t], color=colors[t], label=f"{t} (BER=0)")
    ax.set_title(f"BER vs SNR — SISO vs MIMO 2x2 ({M_name})")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)


def plot_throughput(ax, thr):
    """Tiempo de transmisión y throughput relativo de SISO frente a MIMO 2x2."""
    labels = ["SISO", "MIMO 2x2"]
    t_ms = [thr["t_siso"] * 1e3, thr["t_mimo"] * 1e3]
    bars = ax.bar(labels, t_ms, color=["#7f8c8d", "#c0392b"], alpha=0.8, width=0.6)
    ax.set_ylabel("Tiempo de transmisión (ms)")
    ax.set_title("Tiempo de transmisión para la misma imagen", pad=10, fontsize=11)

    # Margen superior para que los rótulos de las barras no toquen el título
    ax.set_ylim(0, max(t_ms) * 1.45)
    for bar, t, no in zip(bars, t_ms, [thr["n_ofdm_siso"], thr["n_ofdm_mimo"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, t + max(t_ms) * 0.02,
                f"{t:.2f} ms\n{no} OFDM", ha="center", va="bottom", fontsize=9)
    # Nota en la zona libre sobre la barra corta (MIMO), sin chocar con nada
    ax.text(0.97, 0.88, "x2 throughput\n(tiempo a la mitad)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    ax.grid(True, axis="y", alpha=0.3)
