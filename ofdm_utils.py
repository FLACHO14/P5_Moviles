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


# ===================================================================
# SC-FDMA: Análisis comparativo
# ===================================================================

def build_scfdma_map(sc_map, M_dft):
    """Subcarrier map para SC-FDMA: solo M_dft subportadoras de datos."""
    sc = dict(sc_map)
    all_data = sc_map["data_indices"]
    sc["data_indices"] = all_data[:M_dft]
    sc["n_data"] = M_dft
    unused = all_data[M_dft:]
    sc["n_guard"] = sc_map["n_guard"] + len(unused)
    return sc


def run_analysis_scfdma(
    bits_tx, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_list, n_mc, velocity_kmh, M_dft,
):
    """Monte Carlo BER + CCDF PAPR para SC-FDMA."""
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    sc_map_sc = build_scfdma_map(sc_map, M_dft)
    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    ber_data = {}
    ccdf_data = {}

    fs = Nfft * DELTA_F
    n_data = M_dft

    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = n_data * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = np.pad(bits_tx, (0, pad), constant_values=0) if pad else bits_tx.copy()

        n_papr_syms = 1000
        bits_rand = np.random.randint(0, 2, n_papr_syms * n_data * k).astype(np.uint8)
        s_rand = ofdm_tx.qam_mod(bits_rand, M)
        _, papr_vals, _ = ofdm_tx.scfdma_tx_block(
            s_rand, Nfft, cp_len, sc_map, pilot_value, M_dft
        )
        papr_sorted = np.sort(papr_vals)
        ccdf = 1.0 - np.arange(1, len(papr_sorted) + 1) / len(papr_sorted)
        ccdf_data[mod_name] = (papr_sorted, ccdf)

        ber_means, ber_cis = [], []
        for snr_db in snr_list:
            ber_iters = []
            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)
                tx, _, _ = ofdm_tx.scfdma_tx_block(
                    s, Nfft, cp_len, sc_map, pilot_value, M_dft
                )

                h = ofdm_channel.get_channel_profile(chan_profile, taps_L)
                rx, _ = ofdm_channel.apply_channel(tx, h, snr_db, velocity_kmh, fs=fs)

                Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
                Xhat, _ = ofdm_rx.equalize_with_pilots(Y, sc_map_sc, pilot_value, Nfft)
                Xhat = ofdm_rx.scfdma_despread(Xhat, M_dft)

                bits_hat = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
                if len(bits_hat) < len(bits_in):
                    bits_hat = np.pad(bits_hat, (0, len(bits_in) - len(bits_hat)))
                ber_iters.append(np.mean(bits_hat != bits_in))

            m_eq = np.mean(ber_iters)
            s_eq = np.std(ber_iters, ddof=1) if n_mc > 1 else 0
            ber_means.append(m_eq)
            ber_cis.append(1.96 * s_eq / np.sqrt(n_mc))

        ber_data[mod_name] = {"mean": ber_means, "ci": ber_cis}

    return ber_data, ccdf_data


def compute_papr_vs_snr_scfdma(
    Nfft, cp_len, M_mod, sc_map, pilot_value, snr_list, M_dft, n_mc=10, n_symbols=100
):
    """PAPR medio e IC 95% para SC-FDMA."""
    import ofdm_tx

    k = int(np.log2(M_mod))
    n_data = M_dft
    papr_means, papr_stds = [], []

    for _ in snr_list:
        papr_vals = []
        for _ in range(n_mc):
            bits = np.random.randint(0, 2, n_symbols * n_data * k).astype(np.uint8)
            sym = ofdm_tx.qam_mod(bits, M_mod)
            _, papr_list, _ = ofdm_tx.scfdma_tx_block(
                sym, Nfft, cp_len, sc_map, pilot_value, M_dft
            )
            papr_vals.append(np.max(papr_list))
        papr_means.append(np.mean(papr_vals))
        papr_stds.append(np.std(papr_vals))

    conf_intervals = [1.96 * s / np.sqrt(n_mc) for s in papr_stds]
    return papr_means, conf_intervals


def plot_comparative_subcarrier_maps(ax_ofdm, ax_scfdma, sc_map, sc_map_sc, Nfft, M_dft):
    """Mapas de subportadoras comparativos OFDM vs SC-FDMA."""
    cmap = ListedColormap(["#95a5a6", "#2ecc71", "#e74c3c"])

    for ax, smap, title in [
        (ax_ofdm, sc_map, f"OFDM (Nfft={Nfft})"),
        (ax_scfdma, sc_map_sc, f"SC-FDMA (DFT={M_dft}, IFFT={Nfft})"),
    ]:
        type_arr = np.zeros(Nfft, dtype=int)
        type_arr[smap["data_indices"]] = 1
        type_arr[smap["pilot_indices"]] = 2
        freq_order = np.fft.fftshift(type_arr)
        ax.imshow(
            freq_order.reshape(1, -1), aspect="auto", cmap=cmap,
            extent=[-Nfft // 2, Nfft // 2, -0.5, 0.5],
            vmin=0, vmax=2, interpolation="nearest",
        )
        legend_el = [
            Patch(facecolor="#95a5a6", label=f'Guarda+DC ({smap["n_guard"]})'),
            Patch(facecolor="#2ecc71", label=f'Datos ({smap["n_data"]})'),
            Patch(facecolor="#e74c3c", label=f'Pilotos ({smap["n_pilots"]})'),
        ]
        ax.legend(handles=legend_el, loc="upper right", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Subportadora (centrada en DC)")
        ax.set_yticks([])


def plot_comparative_subcarrier_count(ax, sc_map, sc_map_sc, Nfft, M_dft):
    """Barras agrupadas de conteo de subportadoras OFDM vs SC-FDMA."""
    labels = ["Datos", "Pilotos", "Guarda+DC", "Total"]
    ofdm_vals = [sc_map["n_data"], sc_map["n_pilots"], sc_map["n_guard"], Nfft]
    sc_vals = [sc_map_sc["n_data"], sc_map_sc["n_pilots"], sc_map_sc["n_guard"], Nfft]

    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, ofdm_vals, w, label="OFDM", color="#3498db", alpha=0.85)
    b2 = ax.bar(x + w / 2, sc_vals, w, label="SC-FDMA", color="#e67e22", alpha=0.85)

    ax.set_title("Conteo de Subportadoras")
    ax.set_ylabel("Cantidad")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    str(int(h)), ha="center", fontsize=8, fontweight="bold")


def plot_comparative_bandwidth(ax, sc_map, sc_map_sc, delta_f, M_dft):
    """Ancho de banda efectivo OFDM vs SC-FDMA."""
    bw_ofdm = sc_map["n_data"] * delta_f / 1e6
    bw_sc = sc_map_sc["n_data"] * delta_f / 1e6
    bw_total = sc_map["n_used"] * delta_f / 1e6

    labels = ["BW Total\n(usado)", "BW Datos\nOFDM", "BW Datos\nSC-FDMA"]
    vals = [bw_total, bw_ofdm, bw_sc]
    colors = ["#95a5a6", "#3498db", "#e67e22"]

    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("Ancho de Banda Efectivo (MHz)")
    ax.set_ylabel("MHz")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")


def plot_comparative_ber(ax, snr_list, ber_ofdm, ber_scfdma, mod_colors):
    """BER OFDM (sólido) vs SC-FDMA (punteado) para cada modulación."""
    snr_arr = np.array(snr_list)
    for mod_name in ber_ofdm:
        color = mod_colors.get(mod_name, "#333")
        m_ofdm = np.array(ber_ofdm[mod_name]["mean"])
        m_sc = np.array(ber_scfdma[mod_name]["mean"])

        mask_o = m_ofdm > 0
        if np.any(mask_o):
            ax.semilogy(snr_arr[mask_o], m_ofdm[mask_o],
                        marker="o", color=color, label=f"{mod_name} OFDM")
        mask_s = m_sc > 0
        if np.any(mask_s):
            ax.semilogy(snr_arr[mask_s], m_sc[mask_s],
                        marker="s", linestyle="--", color=color,
                        label=f"{mod_name} SC-FDMA", alpha=0.8)

    if not any(np.any(np.array(ber_ofdm[m]["mean"]) > 0) for m in ber_ofdm):
        ax.text(0.5, 0.5, "BER = 0 en todos los puntos",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("BER: OFDM vs SC-FDMA")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


def plot_comparative_ccdf(ax, ccdf_ofdm, ccdf_scfdma, mod_colors):
    """CCDF del PAPR comparativa."""
    for mod_name in ccdf_ofdm:
        color = mod_colors.get(mod_name, "#333")
        p_o, c_o = ccdf_ofdm[mod_name]
        p_s, c_s = ccdf_scfdma[mod_name]
        ax.semilogy(p_o, c_o, color=color, label=f"{mod_name} OFDM")
        ax.semilogy(p_s, c_s, color=color, linestyle="--",
                    label=f"{mod_name} SC-FDMA", alpha=0.8)
    ax.set_title("CCDF PAPR: OFDM vs SC-FDMA")
    ax.set_xlabel("PAPR (dB)")
    ax.set_ylabel("Prob{PAPR > x}")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


def plot_comparative_symbol_power(ax, tx_ofdm, tx_scfdma, Nfft, cp_len):
    """Potencia instantánea de un símbolo OFDM vs SC-FDMA (superpuestas)."""
    sym_len = Nfft + cp_len
    if len(tx_ofdm) < sym_len or len(tx_scfdma) < sym_len:
        return

    for signal, label, color, alpha in [
        (tx_ofdm, "OFDM", "#3498db", 0.9),
        (tx_scfdma, "SC-FDMA", "#e67e22", 0.85),
    ]:
        symbol = signal[:sym_len]
        power = np.abs(symbol) ** 2
        mean_pow = np.mean(power)
        if mean_pow == 0:
            continue
        papr = 10 * np.log10(np.max(power) / mean_pow)
        power_dB = 10 * np.log10(power + 1e-20)
        t = np.arange(len(symbol))
        ax.plot(t, power_dB, color=color, alpha=alpha,
                label=f"{label} (PAPR={papr:.2f} dB)")

    ax.set_title("Potencia Instantánea: OFDM vs SC-FDMA")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Potencia (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if cp_len > 0:
        ax.axvspan(0, cp_len, alpha=0.1, color="gray")
        ymin, ymax = ax.get_ylim()
        ax.text(cp_len / 2, ymax - (ymax - ymin) * 0.05, "CP",
                ha="center", fontsize=8, color="gray")


# ===================================================================
# Diversidad RX: análisis Monte Carlo y graficación
# ===================================================================

DIVERSITY_COLORS = {
    "SISO": "#7f8c8d",
    "MRC": "#2ecc71",
    "SC": "#e67e22",
    "MMSE": "#9b59b6",
}
DIVERSITY_MARKERS = {"SISO": "o", "MRC": "s", "SC": "^", "MMSE": "D"}


def run_analysis_diversity(
    bits_tx, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_list, n_mc, velocity_kmh, NR,
    use_scfdma=False, M_dft=None,
):
    """Monte Carlo BER con IC 95% para SISO vs MRC vs SC vs MMSE.

    Todas las técnicas comparten la misma realización de canal en cada
    iteración para una comparación justa. SISO usa solo la primera antena.
    Los algoritmos de combinación operan en el dominio de la frecuencia
    (después de la FFT-N del RX, antes de la IDFT-M de SC-FDMA).
    """
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    if use_scfdma and M_dft:
        sc_map_eff = build_scfdma_map(sc_map, M_dft)
        n_data = M_dft
    else:
        sc_map_eff = sc_map
        n_data = sc_map["n_data"]

    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    techniques = ["SISO", "MRC", "SC", "MMSE"]
    results = {t: {} for t in techniques}
    fs = Nfft * DELTA_F

    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = n_data * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

        tech_means = {t: [] for t in techniques}
        tech_cis = {t: [] for t in techniques}

        for snr_db in snr_list:
            tech_iters = {t: [] for t in techniques}

            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)
                if use_scfdma and M_dft:
                    tx, _, _ = ofdm_tx.scfdma_tx_block(
                        s, Nfft, cp_len, sc_map, pilot_value, M_dft
                    )
                else:
                    tx, _, _ = ofdm_tx.ofdm_tx_block(
                        s, Nfft, cp_len, sc_map_eff, pilot_value
                    )

                channels = ofdm_channel.generate_mimo_channels(NR, chan_profile, taps_L)
                rx_list = ofdm_channel.apply_channel_mimo(
                    tx, channels, snr_db, velocity_kmh, fs
                )

                Y_list = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rx_list]
                H_all = [
                    ofdm_rx.estimate_channel_from_pilots(Y, sc_map_eff, pilot_value, Nfft)
                    for Y in Y_list
                ]

                combos = {}
                siso_eq, _ = ofdm_rx.equalize_with_pilots(
                    Y_list[0], sc_map_eff, pilot_value, Nfft
                )
                combos["SISO"] = siso_eq
                combos["MRC"], _ = ofdm_rx.combine_mrc(Y_list, H_all, sc_map_eff)
                combos["SC"] = ofdm_rx.combine_sc(Y_list, H_all, sc_map_eff)
                combos["MMSE"] = ofdm_rx.combine_mmse(
                    Y_list, H_all, sc_map_eff, snr_db
                )

                for t in techniques:
                    Xhat = combos[t]
                    if use_scfdma and M_dft:
                        Xhat = ofdm_rx.scfdma_despread(Xhat, M_dft)
                    bh = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
                    if len(bh) < len(bits_in):
                        bh = np.pad(bh, (0, len(bits_in) - len(bh)))
                    tech_iters[t].append(np.mean(bh != bits_in))

            for t in techniques:
                m = np.mean(tech_iters[t])
                s = np.std(tech_iters[t], ddof=1) if n_mc > 1 else 0
                tech_means[t].append(m)
                tech_cis[t].append(1.96 * s / np.sqrt(n_mc))

        for t in techniques:
            results[t][mod_name] = {"mean": tech_means[t], "ci": tech_cis[t]}

    return results


def generate_diversity_power_data(
    Nfft, cp_len, sc_map, pilot_value, chan_profile, taps_L,
    snr_db, velocity_kmh, NR, M_mod=16,
    use_scfdma=False, M_dft=None,
):
    """Genera datos de potencia de canal para visualizar ganancia de diversidad.

    Retorna |H_1[k]|² (una antena) y sum_r|H_r[k]|² (NR antenas combinadas)
    sobre todas las subportadoras, mostrando cómo MRC suaviza los deep fades.
    """
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    if use_scfdma and M_dft:
        sc_map_eff = build_scfdma_map(sc_map, M_dft)
        n_data = M_dft
    else:
        sc_map_eff = sc_map
        n_data = sc_map["n_data"]

    k = int(np.log2(M_mod))
    fs = Nfft * DELTA_F
    bits = np.random.randint(0, 2, 10 * n_data * k).astype(np.uint8)
    syms = ofdm_tx.qam_mod(bits, M_mod)

    if use_scfdma and M_dft:
        tx, _, _ = ofdm_tx.scfdma_tx_block(syms, Nfft, cp_len, sc_map, pilot_value, M_dft)
    else:
        tx, _, _ = ofdm_tx.ofdm_tx_block(syms, Nfft, cp_len, sc_map_eff, pilot_value)

    channels = ofdm_channel.generate_mimo_channels(NR, chan_profile, taps_L)
    rx_list = ofdm_channel.apply_channel_mimo(tx, channels, snr_db, velocity_kmh, fs)

    Y_list = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rx_list]
    H_all = [
        ofdm_rx.estimate_channel_from_pilots(Y, sc_map_eff, pilot_value, Nfft)
        for Y in Y_list
    ]

    H_single = np.abs(H_all[0][0]) ** 2
    H_combined = sum(np.abs(H_all[r][0]) ** 2 for r in range(NR))

    return H_single, H_combined


def plot_diversity_ber(ax, snr_list, div_results, mod_name, NR):
    """BER vs SNR para SISO vs MRC vs SC vs MMSE con IC 95%."""
    snr_arr = np.array(snr_list)
    for tech in ["SISO", "MRC", "SC", "MMSE"]:
        if mod_name not in div_results[tech]:
            continue
        means = np.array(div_results[tech][mod_name]["mean"])
        cis = np.array(div_results[tech][mod_name]["ci"])
        color = DIVERSITY_COLORS[tech]
        marker = DIVERSITY_MARKERS[tech]
        mask = means > 0
        label = f"{tech}" if tech == "SISO" else f"{tech} (NR={NR})"
        if np.any(mask):
            ax.semilogy(
                snr_arr[mask], means[mask],
                marker=marker, color=color, label=label, markersize=5,
            )
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


def plot_diversity_power_stability(ax, H_single, H_combined, Nfft, NR):
    """Potencia del canal |H|² por subportadora: 1 antena vs NR combinadas.

    Demuestra cómo la diversidad suaviza los deep fades del canal.
    Solo muestra subportadoras activas (excluye banda de guarda y DC nulo).
    """
    active = np.where((H_single > 1e-10) & (H_combined > 1e-10))[0]
    if len(active) == 0:
        ax.text(0.5, 0.5, "Sin datos activos", ha="center", va="center",
                transform=ax.transAxes)
        return

    sc_centered = active - Nfft // 2
    H1_dB = 10 * np.log10(H_single[active] + 1e-20)
    Hc_dB = 10 * np.log10(H_combined[active] + 1e-20)

    from scipy.ndimage import uniform_filter1d
    win = max(3, len(active) // 80)
    H1_smooth = uniform_filter1d(H1_dB, size=win)
    Hc_smooth = uniform_filter1d(Hc_dB, size=win)

    ax.fill_between(sc_centered, H1_smooth, alpha=0.15, color="#e74c3c")
    ax.plot(sc_centered, H1_smooth, color="#e74c3c", alpha=0.8, linewidth=0.9,
            label="1 antena (SISO)")
    ax.fill_between(sc_centered, Hc_smooth, alpha=0.15, color="#2ecc71")
    ax.plot(sc_centered, Hc_smooth, color="#2ecc71", alpha=0.9, linewidth=1.3,
            label=f"MRC combinado (NR={NR})")

    mean1 = np.mean(H1_dB)
    meanc = np.mean(Hc_dB)
    ax.axhline(mean1, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(meanc, color="#2ecc71", linestyle="--", alpha=0.5, linewidth=0.8)

    gain = meanc - mean1
    var_single = np.std(H1_dB)
    var_combined = np.std(Hc_dB)
    ax.annotate(
        f"Ganancia diversidad: {gain:.1f} dB\n"
        f"Var SISO: {var_single:.1f} dB  |  Var MRC: {var_combined:.1f} dB",
        xy=(0.02, 0.98), xycoords="axes fraction",
        ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    y_min = max(np.min(H1_smooth) - 3, -30)
    y_max = np.max(Hc_smooth) + 3
    ax.set_ylim(y_min, y_max)

    ax.set_title(f"Ganancia de Diversidad vs Fading (NR={NR})")
    ax.set_xlabel("Subportadora (centrada en DC)")
    ax.set_ylabel("|H|² (dB)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)


# ===================================================================
# Diversidad en TRANSMISIÓN (MISO - SFBC) y comparativa SISO/SIMO/MISO
# ===================================================================

TXDIV_COLORS = {
    "SISO": "#7f8c8d",
    "SIMO-MRC": "#2ecc71",
    "MISO-SFBC": "#e67e22",
}
TXDIV_MARKERS = {"SISO": "o", "SIMO-MRC": "s", "MISO-SFBC": "D"}


def run_analysis_diversity_tx(
    bits_tx, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_list, n_mc, velocity_kmh,
):
    """Monte Carlo BER con IC 95%: SISO vs SIMO-MRC (2 RX) vs MISO-SFBC (2 TX).

    Las tres técnicas se evalúan sobre la misma rejilla OFDM para una
    comparación justa. SISO es 1 TX y 1 RX; SIMO-MRC usa 1 TX y 2 RX con
    combinación MRC en frecuencia; MISO-SFBC usa 2 TX y 1 RX con código
    Alamouti espacio-frecuencia y decodificación en el dominio de la frecuencia.
    """
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    n_data = sc_map["n_data"]
    fs = Nfft * DELTA_F

    # Subconjuntos ortogonales de pilotos por antena TX (para estimar H1, H2)
    pilot_idx = sc_map["pilot_indices"]
    sc_map_a1 = dict(sc_map); sc_map_a1["pilot_indices"] = pilot_idx[::2]
    sc_map_a2 = dict(sc_map); sc_map_a2["pilot_indices"] = pilot_idx[1::2]
    data_idx = sc_map["data_indices"]

    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    techniques = ["SISO", "SIMO-MRC", "MISO-SFBC"]
    results = {t: {} for t in techniques}

    for mod_name, M in mods.items():
        k = int(np.log2(M))
        bits_per_ofdm = n_data * k
        pad = (-len(bits_tx)) % bits_per_ofdm
        bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

        tech_means = {t: [] for t in techniques}
        tech_cis = {t: [] for t in techniques}

        for snr_db in snr_list:
            tech_iters = {t: [] for t in techniques}

            for _ in range(n_mc):
                s = ofdm_tx.qam_mod(bits_in, M)

                # --- SISO: 1 TX, 1 RX ---
                tx_siso, _, _ = ofdm_tx.ofdm_tx_block(
                    s, Nfft, cp_len, sc_map, pilot_value
                )
                h_siso = ofdm_channel.get_channel_profile(chan_profile, taps_L)
                rx_siso, _ = ofdm_channel.apply_channel(
                    tx_siso, h_siso, snr_db, velocity_kmh, fs
                )
                Y_siso = ofdm_rx.ofdm_rx_block(rx_siso, Nfft, cp_len)
                Xs, _ = ofdm_rx.equalize_with_pilots(Y_siso, sc_map, pilot_value, Nfft)
                tech_iters["SISO"].append(
                    _ber_from_symbols(Xs, M, bits_in, ofdm_rx)
                )

                # --- SIMO-MRC: 1 TX, 2 RX ---
                channels = ofdm_channel.generate_mimo_channels(2, chan_profile, taps_L)
                rx_list = ofdm_channel.apply_channel_mimo(
                    tx_siso, channels, snr_db, velocity_kmh, fs
                )
                Y_list = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rx_list]
                H_all = [
                    ofdm_rx.estimate_channel_from_pilots(Y, sc_map, pilot_value, Nfft)
                    for Y in Y_list
                ]
                Xmrc, _ = ofdm_rx.combine_mrc(Y_list, H_all, sc_map)
                tech_iters["SIMO-MRC"].append(
                    _ber_from_symbols(Xmrc, M, bits_in, ofdm_rx)
                )

                # --- MISO-SFBC: 2 TX, 1 RX ---
                tx1, tx2, _, _, _, _ = ofdm_tx.sfbc_tx_2ant(
                    s, Nfft, cp_len, sc_map, pilot_value
                )
                h_miso = ofdm_channel.generate_miso_channels(2, chan_profile, taps_L)
                rx_miso, _ = ofdm_channel.apply_channel_miso(
                    [tx1, tx2], h_miso, snr_db, velocity_kmh, fs
                )
                Y_miso = ofdm_rx.ofdm_rx_block(rx_miso, Nfft, cp_len)
                H1 = ofdm_rx.estimate_channel_from_pilots(Y_miso, sc_map_a1, pilot_value, Nfft)
                H2 = ofdm_rx.estimate_channel_from_pilots(Y_miso, sc_map_a2, pilot_value, Nfft)
                Xsfbc = ofdm_rx.sfbc_decode_2ant(Y_miso, H1, H2, sc_map, data_idx)
                tech_iters["MISO-SFBC"].append(
                    _ber_from_symbols(Xsfbc, M, bits_in, ofdm_rx)
                )

            for t in techniques:
                m = np.mean(tech_iters[t])
                sd = np.std(tech_iters[t], ddof=1) if n_mc > 1 else 0
                tech_means[t].append(m)
                tech_cis[t].append(1.96 * sd / np.sqrt(n_mc))

        for t in techniques:
            results[t][mod_name] = {"mean": tech_means[t], "ci": tech_cis[t]}

    return results


def _ber_from_symbols(Xhat, M, bits_in, ofdm_rx):
    """Demodula símbolos QAM y calcula BER contra los bits de referencia."""
    bh = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
    if len(bh) < len(bits_in):
        bh = np.pad(bh, (0, len(bits_in) - len(bh)))
    return np.mean(bh != bits_in)


def compute_diversity_gain_by_modulation(txdiv_results, snr_list, op_snr=None,
                                         floor=1e-4):
    """Ganancia de diversidad relativa de MISO-SFBC sobre SISO por modulación.

    Se evalúa al SNR de operación como la reducción de BER que aporta SFBC
    respecto a SISO, en dB: 10·log10(BER_SISO / BER_SFBC). Un piso de BER
    representa el punto en que la imagen ya es visualmente limpia y evita
    valores divergentes cuando el BER tiende a cero.

    Esta métrica evidencia que, a un SNR fijo, SFBC es más eficaz en
    modulaciones de bajo orden (QPSK, 16QAM) —que ya operan en su zona de
    cascada y quedan casi sin errores— que en 64QAM, donde la sensibilidad
    intrínseca de la constelación al ruido mantiene un BER residual mayor y
    limita la mejora de calidad utilizable.
    """
    if op_snr is None:
        op_snr = max(snr_list)
    idx = int(np.argmin(np.abs(np.array(snr_list, float) - op_snr)))

    gains = {}
    for mod in ["QPSK", "16QAM", "64QAM"]:
        if mod not in txdiv_results.get("SISO", {}):
            continue
        bs = max(txdiv_results["SISO"][mod]["mean"][idx], floor)
        bf = max(txdiv_results["MISO-SFBC"][mod]["mean"][idx], floor)
        gains[mod] = 10 * np.log10(bs / bf)
    return gains


def diversity_gain_summary(gains):
    """Texto cualitativo de la sensibilidad de SFBC por orden de modulación."""
    lines = []
    for mod in ["QPSK", "16QAM", "64QAM"]:
        if mod in gains:
            lines.append(f"  {mod:6s}: {gains[mod]:+.1f} dB")
    note = (
        "SFBC es más eficaz en modulaciones de bajo orden (QPSK/16QAM);\n"
        "en 64QAM el beneficio se limita por la sensibilidad de la\n"
        "constelación al ruido."
    )
    return "\n".join(lines), note


def plot_diversity_tx_ber(ax, snr_list, txdiv_results, mod_name):
    """BER vs SNR para SISO vs SIMO-MRC vs MISO-SFBC con IC 95%.

    Anota la ganancia de diversidad relativa de SFBC sobre SISO para la
    modulación mostrada.
    """
    snr_arr = np.array(snr_list)
    for tech in ["SISO", "SIMO-MRC", "MISO-SFBC"]:
        if mod_name not in txdiv_results[tech]:
            continue
        means = np.array(txdiv_results[tech][mod_name]["mean"])
        cis = np.array(txdiv_results[tech][mod_name]["ci"])
        color = TXDIV_COLORS[tech]
        marker = TXDIV_MARKERS[tech]
        mask = means > 0
        if np.any(mask):
            ax.semilogy(
                snr_arr[mask], means[mask],
                marker=marker, color=color, label=tech, markersize=5,
            )
            upper = means[mask] + cis[mask]
            lower = np.maximum(means[mask] - cis[mask], 1e-10)
            ax.fill_between(snr_arr[mask], lower, upper, alpha=0.15, color=color)
        else:
            ax.semilogy([], [], marker=marker, color=color, label=f"{tech} (BER=0)")

    gains = compute_diversity_gain_by_modulation(txdiv_results, snr_list)
    if mod_name in gains:
        ax.annotate(
            f"Ganancia SFBC: {gains[mod_name]:+.1f} dB",
            xy=(0.03, 0.04), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
        )

    ax.set_title(f"{mod_name} — BER vs SNR + IC 95%")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


def generate_mrc_constellation_data(
    Nfft, cp_len, sc_map, pilot_value, chan_profile, taps_L,
    snr_db, velocity_kmh, M_mod=16, n_syms_plot=2000,
):
    """Genera símbolos recibidos antes (SISO) y después de MRC (2 RX).

    Sirve para visualizar la reducción de dispersión del ruido por la
    combinación de diversidad en recepción.
    """
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    n_data = sc_map["n_data"]
    k = int(np.log2(M_mod))
    fs = Nfft * DELTA_F
    bits = np.random.randint(0, 2, 12 * n_data * k).astype(np.uint8)
    syms = ofdm_tx.qam_mod(bits, M_mod)

    tx, _, _ = ofdm_tx.ofdm_tx_block(syms, Nfft, cp_len, sc_map, pilot_value)

    channels = ofdm_channel.generate_mimo_channels(2, chan_profile, taps_L)
    rx_list = ofdm_channel.apply_channel_mimo(tx, channels, snr_db, velocity_kmh, fs)

    Y_list = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rx_list]
    H_all = [
        ofdm_rx.estimate_channel_from_pilots(Y, sc_map, pilot_value, Nfft)
        for Y in Y_list
    ]

    # Antes: una sola antena con ecualización ZF
    before, _ = ofdm_rx.equalize_with_pilots(Y_list[0], sc_map, pilot_value, Nfft)
    # Después: combinación MRC de 2 antenas
    after, _ = ofdm_rx.combine_mrc(Y_list, H_all, sc_map)

    return before[:n_syms_plot], after[:n_syms_plot]


def plot_mrc_constellation(ax, data, title, M_mod=16):
    """Dibuja una constelación recibida (scatter I/Q)."""
    ax.scatter(np.real(data), np.imag(data), s=4, alpha=0.35, color="#2980b9")
    lim = 1.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("En fase (I)")
    ax.set_ylabel("Cuadratura (Q)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def generate_papr_instant_data(
    Nfft, cp_len, sc_map, pilot_value, M_mod, M_dft,
    n_show=3,
):
    """Genera potencia instantánea de OFDM-SFBC y SC-FDMA en multi-antena.

    Compara la envolvente de potencia de una antena SFBC (OFDM puro) contra
    la cadena SC-FDMA, evidenciando el menor PAPR de SC-FDMA incluso con
    diversidad en transmisión.
    """
    import ofdm_tx

    n_data = sc_map["n_data"]
    k = int(np.log2(M_mod))
    bits = np.random.randint(0, 2, n_show * 2 * n_data * k).astype(np.uint8)
    syms = ofdm_tx.qam_mod(bits, M_mod)

    # OFDM con SFBC (antena 1)
    tx1, tx2, papr_sfbc, _, _, _ = ofdm_tx.sfbc_tx_2ant(
        syms, Nfft, cp_len, sc_map, pilot_value
    )
    # SC-FDMA single-antenna
    tx_sc, papr_sc, _ = ofdm_tx.scfdma_tx_block(
        syms, Nfft, cp_len, sc_map, pilot_value, M_dft
    )

    sym_len = Nfft + cp_len
    n_keep = min(n_show * sym_len, len(tx1), len(tx_sc))

    p_sfbc = np.abs(tx1[:n_keep]) ** 2
    p_sc = np.abs(tx_sc[:n_keep]) ** 2

    # Normalizar a potencia media unitaria para comparar envolventes
    p_sfbc = p_sfbc / np.mean(p_sfbc)
    p_sc = p_sc / np.mean(p_sc)

    return p_sfbc, p_sc, float(np.mean(papr_sfbc)), float(np.mean(papr_sc))


def plot_papr_instant(ax, p_sfbc, p_sc, papr_sfbc, papr_sc):
    """Dibuja la potencia instantánea normalizada de OFDM-SFBC vs SC-FDMA."""
    n = np.arange(len(p_sfbc))
    ax.plot(n, 10 * np.log10(p_sfbc + 1e-12), color="#e74c3c", alpha=0.8,
            linewidth=0.9, label=f"OFDM-SFBC (PAPR={papr_sfbc:.1f} dB)")
    ax.plot(n, 10 * np.log10(p_sc + 1e-12), color="#2980b9", alpha=0.8,
            linewidth=0.9, label=f"SC-FDMA (PAPR={papr_sc:.1f} dB)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6,
               label="Potencia media")
    ax.set_title("Potencia Instantánea: OFDM-SFBC vs SC-FDMA")
    ax.set_xlabel("Muestra temporal")
    ax.set_ylabel("Potencia normalizada (dB)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


# ===================================================================
# Transmisión de imagen en paralelo: SISO vs MISO-SFBC
# ===================================================================

def _bits_to_image(bits_rx, n_bits, total_bits_cap, img_shape):
    """Reconstruye una imagen desde bits de forma robusta ante errores.

    Ajusta longitud (rellena o recorta) en cada etapa para que nunca
    falle aunque haya demasiados bits erróneos o faltantes.
    """
    bits_rx = np.asarray(bits_rx).astype(np.uint8).ravel()
    if len(bits_rx) < total_bits_cap:
        bits_rx = np.pad(bits_rx, (0, total_bits_cap - len(bits_rx)))
    else:
        bits_rx = bits_rx[:total_bits_cap]

    bits_img = bits_rx[:n_bits]
    img_size = int(np.prod(img_shape))
    img_bytes = np.packbits(bits_img)
    if len(img_bytes) > img_size:
        img_bytes = img_bytes[:img_size]
    elif len(img_bytes) < img_size:
        img_bytes = np.pad(img_bytes, (0, img_size - len(img_bytes)))

    return img_bytes.reshape(img_shape).astype(np.uint8)


def _img_metrics(orig, rec):
    """Retorna (MSE, PSNR en dB) entre la imagen original y la recibida."""
    mse = np.mean((orig.astype(float) - rec.astype(float)) ** 2)
    if mse <= 1e-9:
        return 0.0, 100.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return float(mse), float(psnr)


def transmit_image_siso_vs_sfbc(
    bits_tx, img_arr, Nfft, cp_len, sc_map, pilot_value,
    chan_profile, taps_L, snr_db, velocity_kmh, M, nr_rx=1,
):
    """Transmite la misma imagen por hilos paralelos: SISO y MISO-SFBC.

    Hebra A (SISO): transmisión estándar de una antena, sin redundancia.
    Hebra B (MISO-SFBC): código Alamouti espacio-frecuencia con 2 antenas TX,
    mapeando (a0, a1) en la antena 1 y (-a1*, a0*) en la antena 2 sobre
    subportadoras adyacentes, según el estándar LTE.
    Hebra C (SFBC 2x2): solo si nr_rx >= 2, decodifica el mismo bloque SFBC
    combinando las observaciones de 2 antenas receptoras (orden de
    diversidad 4) para máxima robustez ante desvanecimientos profundos.

    Las cadenas se ejecutan en hilos de procesamiento distintos y se
    sincronizan al final. La reconstrucción de bits a píxeles es robusta
    ante errores de recepción para que el proceso no se detenga.

    El canal Rayleigh se configura con taps_L caminos; con taps_L = 2 se
    recrea una selectividad en frecuencia moderada pero real, ideal para
    observar cómo SFBC protege los datos cuando un tap cae en un hueco.

    Returns
    -------
    dict con imágenes reconstruidas, métricas (MSE, PSNR, BER) y las
    respuestas de canal en frecuencia de cada antena TX.
    """
    import threading
    import ofdm_tx
    import ofdm_channel
    import ofdm_rx

    fs = Nfft * DELTA_F
    n_bits = len(bits_tx)
    k = int(np.log2(M))
    n_data = sc_map["n_data"]
    bits_per_ofdm = n_data * k
    n_ofdm_needed = int(np.ceil(n_bits / bits_per_ofdm))
    total_bits_cap = n_ofdm_needed * bits_per_ofdm
    pad = total_bits_cap - n_bits
    bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

    # Pilotos ortogonales por antena TX para estimar H1 y H2 por separado
    pilot_idx = sc_map["pilot_indices"]
    sc_map_a1 = dict(sc_map); sc_map_a1["pilot_indices"] = pilot_idx[::2]
    sc_map_a2 = dict(sc_map); sc_map_a2["pilot_indices"] = pilot_idx[1::2]
    data_idx = sc_map["data_indices"]

    out = {}

    def hebra_siso():
        syms = ofdm_tx.qam_mod(bits_in, M)
        tx, _, _ = ofdm_tx.ofdm_tx_block(syms, Nfft, cp_len, sc_map, pilot_value)
        h = ofdm_channel.get_channel_profile(chan_profile, taps_L)
        rx, h_used = ofdm_channel.apply_channel(tx, h, snr_db, velocity_kmh, fs)
        Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
        Xhat, _ = ofdm_rx.equalize_with_pilots(Y, sc_map, pilot_value, Nfft)
        out["siso_bits"] = ofdm_rx.qam_demod(Xhat, M)
        out["H_siso"] = np.fft.fft(h_used, Nfft)

    def hebra_sfbc():
        syms = ofdm_tx.qam_mod(bits_in, M)
        tx1, tx2, _, _, _, _ = ofdm_tx.sfbc_tx_2ant(
            syms, Nfft, cp_len, sc_map, pilot_value
        )

        # --- 1 RX (MISO 2x1) ---
        h_miso = ofdm_channel.generate_miso_channels(2, chan_profile, taps_L)
        rx, _ = ofdm_channel.apply_channel_miso(
            [tx1, tx2], h_miso, snr_db, velocity_kmh, fs
        )
        Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
        H1 = ofdm_rx.estimate_channel_from_pilots(Y, sc_map_a1, pilot_value, Nfft)
        H2 = ofdm_rx.estimate_channel_from_pilots(Y, sc_map_a2, pilot_value, Nfft)
        Xhat = ofdm_rx.sfbc_decode_2ant(Y, H1, H2, sc_map, data_idx)
        out["sfbc_bits"] = ofdm_rx.qam_demod(Xhat, M)
        out["H1"] = np.fft.fft(h_miso[0], Nfft)
        out["H2"] = np.fft.fft(h_miso[1], Nfft)

        # --- 2 RX (MIMO 2x2): combinación de dos observaciones del bloque ---
        if nr_rx >= 2:
            Y_list, H1_list, H2_list = [], [], []
            for _r in range(2):
                ch_r = ofdm_channel.generate_miso_channels(2, chan_profile, taps_L)
                rx_r, _ = ofdm_channel.apply_channel_miso(
                    [tx1, tx2], ch_r, snr_db, velocity_kmh, fs
                )
                Y_r = ofdm_rx.ofdm_rx_block(rx_r, Nfft, cp_len)
                Y_list.append(Y_r)
                H1_list.append(
                    ofdm_rx.estimate_channel_from_pilots(Y_r, sc_map_a1, pilot_value, Nfft)
                )
                H2_list.append(
                    ofdm_rx.estimate_channel_from_pilots(Y_r, sc_map_a2, pilot_value, Nfft)
                )
            Xhat2 = ofdm_rx.sfbc_decode_2ant_2rx(
                Y_list, H1_list, H2_list, sc_map, data_idx
            )
            out["sfbc2x2_bits"] = ofdm_rx.qam_demod(Xhat2, M)

    tA = threading.Thread(target=hebra_siso)
    tB = threading.Thread(target=hebra_sfbc)
    tA.start(); tB.start()
    tA.join(); tB.join()

    img_shape = img_arr.shape
    img_siso = _bits_to_image(out["siso_bits"], n_bits, total_bits_cap, img_shape)
    img_sfbc = _bits_to_image(out["sfbc_bits"], n_bits, total_bits_cap, img_shape)

    mse_siso, psnr_siso = _img_metrics(img_arr, img_siso)
    mse_sfbc, psnr_sfbc = _img_metrics(img_arr, img_sfbc)

    def _ber(bits_rx):
        b = np.asarray(bits_rx).astype(np.uint8).ravel()
        if len(b) < total_bits_cap:
            b = np.pad(b, (0, total_bits_cap - len(b)))
        return float(np.mean(b[:n_bits] != bits_tx))

    result = {
        "img_orig": img_arr,
        "img_siso": img_siso,
        "img_sfbc": img_sfbc,
        "mse_siso": mse_siso, "psnr_siso": psnr_siso,
        "mse_sfbc": mse_sfbc, "psnr_sfbc": psnr_sfbc,
        "ber_siso": _ber(out["siso_bits"]),
        "ber_sfbc": _ber(out["sfbc_bits"]),
        "H_siso": out["H_siso"],
        "H1": out["H1"], "H2": out["H2"],
        "used_indices": sc_map["used_indices"],
        "Nfft": Nfft,
        "snr_db": snr_db,
        "taps_L": taps_L,
        "nr_rx": nr_rx,
    }

    if "sfbc2x2_bits" in out:
        img_2x2 = _bits_to_image(out["sfbc2x2_bits"], n_bits, total_bits_cap, img_shape)
        mse_2x2, psnr_2x2 = _img_metrics(img_arr, img_2x2)
        result["img_sfbc2x2"] = img_2x2
        result["mse_sfbc2x2"] = mse_2x2
        result["psnr_sfbc2x2"] = psnr_2x2
        result["ber_sfbc2x2"] = _ber(out["sfbc2x2_bits"])

    return result


def plot_image_panel(ax, img, title, subtitle=None):
    """Muestra una imagen en escala de grises con título y subtítulo.

    El subtítulo se coloca centrado debajo de la imagen; usar saltos de
    línea (\\n) en subtitle para evitar que el texto se solape con los
    paneles vecinos.
    """
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if subtitle:
        ax.text(
            0.5, -0.04, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, linespacing=1.4,
        )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_sfbc_channel_redundancy(ax, H1, H2, used_indices, Nfft):
    """Magnitud del canal de las 2 antenas TX, resaltando la redundancia SFBC.

    Resalta las subportadoras donde la antena 1 cae en un hueco de
    desvanecimiento profundo pero la antena 2 mantiene buena ganancia
    (y viceversa), que es donde la diversidad SFBC recupera el símbolo.
    """
    sc = np.sort(used_indices)
    sc_centered = np.where(sc > Nfft // 2, sc - Nfft, sc)
    o = np.argsort(sc_centered)
    x = sc_centered[o]

    h1_dB = 20 * np.log10(np.abs(H1[sc][o]) + 1e-12)
    h2_dB = 20 * np.log10(np.abs(H2[sc][o]) + 1e-12)

    ax.plot(x, h1_dB, color="#2980b9", alpha=0.85, linewidth=1.0, label="Antena TX 1")
    ax.plot(x, h2_dB, color="#e67e22", alpha=0.85, linewidth=1.0, label="Antena TX 2")

    # Huecos de desvanecimiento: una antena cae muy por debajo de su mediana
    # pero la otra mantiene mejor ganancia → la diversidad cubre el hueco.
    med1, med2 = np.median(h1_dB), np.median(h2_dB)
    thr = 6.0
    redund = ((h1_dB < med1 - thr) & (h2_dB > h1_dB + thr)) | \
             ((h2_dB < med2 - thr) & (h1_dB > h2_dB + thr))

    y_min = min(np.min(h1_dB), np.min(h2_dB)) - 2
    y_max = max(np.max(h1_dB), np.max(h2_dB)) + 2
    ax.fill_between(x, y_min, y_max, where=redund, color="gold", alpha=0.35,
                    label="Redundancia SFBC (hueco cubierto)")
    ax.set_ylim(y_min, y_max)

    n_redund = int(np.sum(redund))
    ax.annotate(
        f"Subportadoras con hueco cubierto por diversidad: {n_redund}",
        xy=(0.02, 0.04), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=8, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_title("Respuesta de Canal en Frecuencia — Antenas TX 1 y 2 (SFBC)")
    ax.set_xlabel("Subportadora (centrada en DC)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
