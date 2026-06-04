# ofdm_utils.py
# Utilidades del simulador OFDM 4G LTE.
#
# Funciones principales:
# - get_nfft_cp:              Nfft y CP desde BW (regla 90% banda de guarda)
# - build_subcarrier_map:     asignación datos / pilotos / guarda / DC
# - calculate_resource_stats: conteo de subportadoras y símbolos OFDM
# - classify_channel:         selectivo vs plano, rápido vs lento
# - run_analysis:             Monte Carlo BER con IC 95% + CCDF PAPR
# - compute_papr_vs_snr:      estadísticas PAPR con IC
# - Funciones de graficación  para cada pestaña de la GUI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from ofdm_params import DELTA_F, FC


def next_pow2(n):
    """Siguiente potencia de 2 >= n."""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


# ===================================================================
# Parámetros del sistema
# ===================================================================

def get_nfft_cp(bw_mhz, cp_mode, delta_f=None):
    """Calcula Nfft, cp_len y N_used aplicando la regla del 90%.

    Solo el 90% del ancho de banda se destina a subportadoras útiles;
    el 10% restante forma la banda de guarda para atenuar la
    interferencia con bandas adyacentes (estándar LTE).

    CP Normal  ~ Nfft/8 muestras (~4.7 us)
    CP Extendido ~ Nfft/4 muestras (~16.7 us)
    """
    if delta_f is None:
        delta_f = DELTA_F
    bw_hz = bw_mhz * 1e6
    bw_useful = 0.9 * bw_hz  # 90% del BW
    N_used = int(round(bw_useful / delta_f))
    if N_used < 8:
        N_used = 8
    Nfft = next_pow2(N_used)
    if N_used > Nfft:
        N_used = Nfft
    cp_len = Nfft // 8 if cp_mode == "Normal" else Nfft // 4
    return Nfft, cp_len, N_used


def build_subcarrier_map(Nfft, N_used, pilot_spacing):
    """Construye el mapa de asignación de subportadoras estilo LTE.

    Distribución en el espectro (orden FFT de numpy):
      Indice 0            : DC (nula — no se transmite)
      Indices 1..half     : subportadoras útiles (frecuencias positivas)
      Indices Nfft-half..Nfft-1 : subportadoras útiles (freq negativas)
      Resto               : banda de guarda (ceros)

    Dentro de las útiles, cada ``pilot_spacing`` subportadoras se
    designa como piloto; las demás llevan datos.
    """
    half = N_used // 2

    pos_indices = np.arange(1, half + 1)
    neg_indices = np.arange(Nfft - half, Nfft)
    used_indices = np.concatenate([neg_indices, pos_indices])

    # Pilotos uniformes dentro de las usadas
    pilot_mask = np.zeros(len(used_indices), dtype=bool)
    pilot_mask[::pilot_spacing] = True
    pilot_indices = used_indices[pilot_mask]
    data_indices = used_indices[~pilot_mask]

    used_set = set(used_indices.tolist())
    guard_indices = np.array(sorted(set(range(Nfft)) - used_set - {0}))

    return {
        "data_indices": data_indices,
        "pilot_indices": pilot_indices,
        "guard_indices": guard_indices,
        "dc_index": 0,
        "used_indices": used_indices,
        "n_data": len(data_indices),
        "n_pilots": len(pilot_indices),
        "n_guard": len(guard_indices) + 1,  # +1 por DC
        "n_used": len(used_indices),
        "Nfft": Nfft,
    }


def calculate_resource_stats(n_bits, sc_map, M):
    """Estadísticas de recursos OFDM considerando pilotos y banda de guarda."""
    k = int(np.log2(M))
    bits_per_ofdm = sc_map["n_data"] * k
    n_ofdm = int(np.ceil(n_bits / bits_per_ofdm))
    total_capacity = n_ofdm * bits_per_ofdm
    padding = total_capacity - n_bits
    return {
        "bits_per_ofdm": bits_per_ofdm,
        "total_ofdm_symbols": n_ofdm,
        "padding_zeros": padding,
        "n_data_sc": sc_map["n_data"],
        "n_pilot_sc": sc_map["n_pilots"],
        "n_guard_sc": sc_map["n_guard"],
        "n_total_sc": sc_map["Nfft"],
    }


# ===================================================================
# Clasificación del canal
# ===================================================================

def classify_channel(h, fs, N_used, velocity_kmh, delta_f=None):
    """Clasifica el tipo de desvanecimiento del canal.

    Frecuencia:
      Bc (coherence bandwidth) ≈ 1 / (5 · sigma_tau)
      Si BW_señal > Bc -> selectivo en frecuencia; sino -> plano

    Tiempo:
      fd (Doppler spread)  = v · fc / c
      Tc (coherence time)  ≈ 0.423 / fd
      Si T_simbolo > Tc   -> desvanecimiento rápido; sino -> lento
    """
    if delta_f is None:
        delta_f = DELTA_F

    power_profile = np.abs(h) ** 2
    total_power = np.sum(power_profile)

    if total_power == 0 or len(h) <= 1:
        return {
            "rms_delay_us": 0,
            "Bc_kHz": float("inf"),
            "BW_kHz": N_used * delta_f / 1e3,
            "freq_type": "Frecuencia plana (canal ideal)",
            "fd_Hz": 0,
            "Tc_ms": float("inf"),
            "Tsym_us": 1 / delta_f * 1e6,
            "time_type": "Desvanecimiento lento (estático)",
        }

    delays = np.arange(len(h)) / fs
    mean_delay = np.sum(delays * power_profile) / total_power
    rms_delay = np.sqrt(
        np.sum((delays - mean_delay) ** 2 * power_profile) / total_power
    )

    Bc = 1 / (5 * rms_delay) if rms_delay > 0 else float("inf")
    BW = N_used * delta_f

    if Bc == float("inf"):
        freq_type = "Frecuencia plana"
    elif BW > Bc:
        freq_type = "Selectivo en frecuencia"
    else:
        freq_type = "Frecuencia plana"

    v_ms = velocity_kmh / 3.6
    fd = (v_ms * FC) / 3e8 if velocity_kmh > 0 else 0
    Tc = 0.423 / fd if fd > 0 else float("inf")
    T_sym = 1 / delta_f

    if fd == 0:
        time_type = "Desvanecimiento lento (estático)"
    elif T_sym > Tc:
        time_type = "Desvanecimiento rápido"
    else:
        time_type = "Desvanecimiento lento"

    return {
        "rms_delay_us": rms_delay * 1e6,
        "Bc_kHz": Bc / 1e3 if Bc != float("inf") else float("inf"),
        "BW_kHz": BW / 1e3,
        "freq_type": freq_type,
        "fd_Hz": fd,
        "Tc_ms": Tc * 1e3 if Tc != float("inf") else float("inf"),
        "Tsym_us": T_sym * 1e6,
        "time_type": time_type,
    }


# ===================================================================
# Análisis Monte Carlo
# ===================================================================

def run_analysis(
    bits_tx, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_list, n_mc, velocity_kmh=0,
):
    """Simulación Monte Carlo de BER vs SNR para QPSK, 16QAM y 64QAM.

    Para cada modulación y valor de SNR se ejecutan ``n_mc`` realizaciones
    independientes del canal.  Se calcula:
    - BER con ecualización por pilotos  (cadena realista)
    - BER sin ecualización              (para comparar)
    - Intervalo de confianza del 95%:   mean ± 1.96 · std / sqrt(n_mc)

    También se genera la CCDF del PAPR con 1000 símbolos aleatorios.

    Returns
    -------
    ber_data   : dict  {mod: {'mean': [...], 'ci': [...]}}  con pilotos
    ber_no_eq  : dict  {mod: {'mean': [...], 'ci': [...]}}  sin ecualización
    ccdf_data  : dict  {mod: (papr_sorted, ccdf)}
    """
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    ber_data = {}
    ber_no_eq = {}
    ccdf_data = {}

    fs = Nfft * DELTA_F
    n_data = sc_map["n_data"]

    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = n_data * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = (
            np.pad(bits_tx, (0, pad), constant_values=0) if pad else bits_tx.copy()
        )

        # ----- CCDF del PAPR -----
        n_papr_syms = 1000
        bits_rand = np.random.randint(0, 2, n_papr_syms * n_data * k).astype(
            np.uint8
        )
        s_rand = ofdm_tx.qam_mod(bits_rand, M)
        _, papr_vals, _ = ofdm_tx.ofdm_tx_block(
            s_rand, Nfft, cp_len, sc_map, pilot_value
        )
        papr_sorted = np.sort(papr_vals)
        ccdf = 1.0 - np.arange(1, len(papr_sorted) + 1) / len(papr_sorted)
        ccdf_data[mod_name] = (papr_sorted, ccdf)

        # ----- BER Monte Carlo con IC 95% -----
        ber_means, ber_cis = [], []
        ber_noeq_means, ber_noeq_cis = [], []

        for snr_db in snr_list:
            ber_iters = []
            ber_noeq_iters = []

            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)
                tx, _, _ = ofdm_tx.ofdm_tx_block(
                    s, Nfft, cp_len, sc_map, pilot_value
                )

                h = ofdm_channel.get_channel_profile(chan_profile, taps_L)
                rx, _ = ofdm_channel.apply_channel(
                    tx, h, snr_db, velocity_kmh, fs=fs
                )

                Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)

                # Con pilotos
                Xhat, _ = ofdm_rx.equalize_with_pilots(
                    Y, sc_map, pilot_value, Nfft
                )
                bits_hat = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
                if len(bits_hat) < len(bits_in):
                    bits_hat = np.pad(
                        bits_hat, (0, len(bits_in) - len(bits_hat))
                    )
                ber_iters.append(np.mean(bits_hat != bits_in))

                # Sin ecualización
                Xraw = ofdm_rx.demod_without_equalization(Y, sc_map)
                bits_raw = ofdm_rx.qam_demod(Xraw, M)[: len(bits_in)]
                if len(bits_raw) < len(bits_in):
                    bits_raw = np.pad(
                        bits_raw, (0, len(bits_in) - len(bits_raw))
                    )
                ber_noeq_iters.append(np.mean(bits_raw != bits_in))

            # Media e IC 95%
            m_eq = np.mean(ber_iters)
            s_eq = np.std(ber_iters, ddof=1) if n_mc > 1 else 0
            ber_means.append(m_eq)
            ber_cis.append(1.96 * s_eq / np.sqrt(n_mc))

            m_noeq = np.mean(ber_noeq_iters)
            s_noeq = np.std(ber_noeq_iters, ddof=1) if n_mc > 1 else 0
            ber_noeq_means.append(m_noeq)
            ber_noeq_cis.append(1.96 * s_noeq / np.sqrt(n_mc))

        ber_data[mod_name] = {"mean": ber_means, "ci": ber_cis}
        ber_no_eq[mod_name] = {"mean": ber_noeq_means, "ci": ber_noeq_cis}

    return ber_data, ber_no_eq, ccdf_data


def compute_papr_vs_snr(
    Nfft, cp_len, M, sc_map, pilot_value, snr_list, n_mc=10, n_symbols=100
):
    """PAPR medio e IC 95 % para cada punto de la lista SNR.

    Nota: el PAPR es propiedad de la señal TX y no depende del SNR;
    la iteración sirve para obtener la varianza estadística.
    """
    import ofdm_tx

    k = int(np.log2(M))
    n_data = sc_map["n_data"]
    papr_means, papr_stds = [], []

    for _ in snr_list:
        papr_vals = []
        for _ in range(n_mc):
            bits = np.random.randint(0, 2, n_symbols * n_data * k).astype(
                np.uint8
            )
            sym = ofdm_tx.qam_mod(bits, M)
            _, papr_list, _ = ofdm_tx.ofdm_tx_block(
                sym, Nfft, cp_len, sc_map, pilot_value
            )
            papr_vals.append(np.max(papr_list))
        papr_means.append(np.mean(papr_vals))
        papr_stds.append(np.std(papr_vals))

    conf_intervals = [1.96 * s / np.sqrt(n_mc) for s in papr_stds]
    return papr_means, conf_intervals


# ===================================================================
# Funciones de graficación
# ===================================================================

def plot_subcarrier_map(ax_map, ax_bar, sc_map, Nfft):
    """Visualiza la asignación de subportadoras.

    ax_map: mapa de color (datos=verde, pilotos=rojo, guarda=gris)
    ax_bar: gráfico de barras con el conteo de cada tipo
    """
    type_arr = np.zeros(Nfft, dtype=int)
    type_arr[sc_map["data_indices"]] = 1
    type_arr[sc_map["pilot_indices"]] = 2

    freq_order = np.fft.fftshift(type_arr)
    cmap = ListedColormap(["#95a5a6", "#2ecc71", "#e74c3c"])
    ax_map.imshow(
        freq_order.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        extent=[-Nfft // 2, Nfft // 2, -0.5, 0.5],
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    legend_el = [
        Patch(facecolor="#95a5a6", label=f'Guarda+DC ({sc_map["n_guard"]})'),
        Patch(facecolor="#2ecc71", label=f'Datos ({sc_map["n_data"]})'),
        Patch(facecolor="#e74c3c", label=f'Pilotos ({sc_map["n_pilots"]})'),
    ]
    ax_map.legend(handles=legend_el, loc="upper right", fontsize=7)
    ax_map.set_title(f"Asignación de Subportadoras (Nfft={Nfft})")
    ax_map.set_xlabel("Subportadora (centrada en DC)")
    ax_map.set_yticks([])

    # Barras resumen
    labels = ["Datos", "Pilotos", "Guarda+DC", "Total"]
    vals = [sc_map["n_data"], sc_map["n_pilots"], sc_map["n_guard"], Nfft]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#3498db"]
    bars = ax_bar.bar(labels, vals, color=colors)
    ax_bar.set_title("Conteo de Subportadoras")
    ax_bar.set_ylabel("Cantidad")
    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            str(v),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )


def plot_subcarrier_spacing(ax, delta_f=None):
    """Demuestra la ortogonalidad: nulos de una sinc coinciden con picos de vecinas."""
    if delta_f is None:
        delta_f = DELTA_F
    n_sc = 5
    f = np.linspace(-1.5 * delta_f, (n_sc + 0.5) * delta_f, 2000)
    palette = ["#2980b9", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad"]
    for i in range(n_sc):
        sinc_i = np.abs(np.sinc((f - i * delta_f) / delta_f))
        ax.plot(
            f / 1e3, sinc_i,
            label=f"SC {i}", color=palette[i % len(palette)], alpha=0.8,
        )
        ax.axvline(i * delta_f / 1e3, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"Ortogonalidad OFDM ($\\Delta f$ = {delta_f/1e3:.0f} kHz)")
    ax.set_xlabel("Frecuencia (kHz)")
    ax.set_ylabel("Amplitud |sinc|")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_channel_response(ax_time, ax_freq, h, fs, Nfft, channel_name,
                          chan_class=None):
    """Respuesta al impulso y en frecuencia del canal, con clasificación."""
    # Tiempo
    t_axis = np.arange(len(h)) / fs * 1e6
    ax_time.stem(t_axis, np.abs(h), basefmt=" ")
    ax_time.set_title(f"Respuesta al impulso ({channel_name})")
    ax_time.set_xlabel("Tiempo (μs)")
    ax_time.set_ylabel("|h(t)|")
    ax_time.grid(True, alpha=0.3)

    # Frecuencia
    H_f = np.fft.fft(h, n=Nfft)
    f_axis = np.fft.fftfreq(Nfft, 1 / fs) / 1e3
    f_shift = np.fft.fftshift(f_axis)
    H_shift = np.fft.fftshift(H_f)
    ax_freq.plot(f_shift, 20 * np.log10(np.abs(H_shift) + 1e-12), color="orange")
    ax_freq.set_title("Respuesta en frecuencia |H(f)|")
    ax_freq.set_xlabel("Frecuencia (kHz)")
    ax_freq.set_ylabel("|H(f)| (dB)")
    ax_freq.grid(True, alpha=0.3)

    # Clasificación como texto superpuesto
    if chan_class:
        bc = (
            "∞" if chan_class["Bc_kHz"] == float("inf")
            else f"{chan_class['Bc_kHz']:.1f}"
        )
        tc = (
            "∞" if chan_class["Tc_ms"] == float("inf")
            else f"{chan_class['Tc_ms']:.2f}"
        )
        info = (
            f"RMS Delay: {chan_class['rms_delay_us']:.3f} μs\n"
            f"Bc: {bc} kHz  |  BW: {chan_class['BW_kHz']:.1f} kHz\n"
            f"→ {chan_class['freq_type']}\n"
            f"fd: {chan_class['fd_Hz']:.1f} Hz  |  Tc: {tc} ms\n"
            f"→ {chan_class['time_type']}"
        )
        ax_freq.text(
            0.02, 0.02, info,
            transform=ax_freq.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )


def plot_cp_analysis(ax, cp_normal, cp_extended, Nfft, fs):
    """Compara CP Normal vs Extendido en muestras y duración."""
    labels = ["CP Normal", "CP Extendido"]
    values = [cp_normal, cp_extended]
    colors = ["#3498db", "#9b59b6"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Longitud del Prefijo Cíclico")
    ax.set_ylabel("Muestras")
    for i, v in enumerate(values):
        dur = v / fs * 1e6
        ax.text(
            i, v + 0.5, f"{v} ({dur:.1f} μs)",
            ha="center", fontweight="bold", fontsize=9,
        )
    ax.grid(True, alpha=0.3, axis="y")


def plot_papr_time_domain(ax, tx_signal, Nfft, cp_len):
    """Potencia instantánea vs promedio de un símbolo OFDM (visualización PAPR)."""
    sym_len = Nfft + cp_len
    if len(tx_signal) < sym_len:
        return
    symbol = tx_signal[:sym_len]
    power = np.abs(symbol) ** 2
    mean_power = np.mean(power)
    if mean_power == 0:
        return

    papr = 10 * np.log10(np.max(power) / mean_power)
    power_dB = 10 * np.log10(power + 1e-20)
    mean_dB = 10 * np.log10(mean_power)

    t = np.arange(len(symbol))
    ax.plot(t, power_dB, color="#3498db", alpha=0.8, label="Potencia instantánea")
    ax.axhline(
        mean_dB, color="#e74c3c", linestyle="--", linewidth=2,
        label=f"Pot. promedio ({mean_dB:.1f} dB)",
    )
    ax.axhline(
        mean_dB + papr, color="#f39c12", linestyle=":", linewidth=1.5,
        label=f"Pot. pico ({mean_dB + papr:.1f} dB)",
    )

    ax.annotate(
        f"PAPR = {papr:.2f} dB",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_title("Potencia de un Símbolo OFDM")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Potencia (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Zona del CP sombreada
    if cp_len > 0:
        ax.axvspan(0, cp_len, alpha=0.12, color="gray")
        ymin, ymax = ax.get_ylim()
        ax.text(
            cp_len / 2, ymax - (ymax - ymin) * 0.05, "CP",
            ha="center", fontsize=8, color="gray",
        )
