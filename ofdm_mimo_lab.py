# ofdm_mimo_lab.py
# Laboratorio integral de sistemas multi-antena (MIMO) sobre la cadena OFDM.
#
# Reúne y compara, de forma modular y activable por parámetros, las técnicas:
#   - SISO            : una antena, sin diversidad (referencia).
#   - SIMO-MRC        : NR antenas RX, combinación de máxima razón.
#   - SIMO-IRC        : NR antenas RX, rechazo de un interferente fuerte.
#   - MISO-SFBC       : diversidad en TX con código Alamouti.
#   - Beamforming     : precodificación con CSI ideal (ganancia de array).
#   - Multiplexación  : MIMO 2x2 con receptor Zero-Forcing (doble tasa).
#
# Cada técnica aporta una ganancia distinta según la teoría de LTE: la
# diversidad (MRC, SFBC) reduce el desvanecimiento; el beamforming concentra
# potencia hacia el receptor; el IRC suprime interferencia co-canal; y la
# multiplexación espacial multiplica la tasa de datos a costa de diversidad.

import time
import numpy as np

import ofdm_tx
import ofdm_rx
import ofdm_channel
import ofdm_utils
import ofdm_beamforming
from ofdm_params import DELTA_F


# ===================================================================
# Canal con correlación espacial (Rayleigh selectivo, L taps)
# ===================================================================

def generate_correlated_channels(n, profile_name, taps_L=2, correlation="low"):
    """Genera n canales con la correlación espacial indicada.

    Correlación baja: canales independientes, lo que habilita ganancia de
    diversidad. Correlación alta: los canales comparten una componente común
    dominante, por lo que se desvanecen de forma conjunta y la diversidad casi
    desaparece. Con taps_L=2 el canal es selectivo en frecuencia de forma
    moderada, como pide el estudio.
    """
    rho = 0.95 if correlation == "high" else 0.0
    h_common = ofdm_channel.get_channel_profile(profile_name, taps_L)
    out = []
    for _ in range(n):
        h_ind = ofdm_channel.get_channel_profile(profile_name, taps_L)
        L = max(len(h_common), len(h_ind))
        hc = np.pad(h_common, (0, L - len(h_common)))
        hi = np.pad(h_ind, (0, L - len(h_ind)))
        h = np.sqrt(rho) * hc + np.sqrt(1.0 - rho) * hi
        norm = np.sqrt(np.sum(np.abs(h) ** 2))
        if norm > 0:
            h = h / norm
        out.append(h)
    return out


# ===================================================================
# SIMO - IRC (Interference Rejection Combining)
# ===================================================================

def apply_simo_with_interference(tx_desired, ch_desired, tx_interf, ch_interf,
                                 snr_db, sir_db, velocity_kmh=0, fs=1.92e6):
    """Recibe en NR antenas la señal deseada más un interferente co-canal.

    Cada antena recibe: y_r = h_d,r * x_d + alpha * h_i,r * x_i + ruido.
    alpha fija la relación señal a interferencia (SIR). El ruido se calibra
    con la SNR respecto a la potencia de la señal deseada.
    """
    NR = len(ch_desired)
    sig_ref = None
    rx_list = []
    alpha = 10 ** (-sir_db / 20)
    for r in range(NR):
        yd = np.convolve(tx_desired, ch_desired[r], mode="full")[: len(tx_desired)]
        yi = np.convolve(tx_interf, ch_interf[r], mode="full")[: len(tx_interf)]
        yd = ofdm_channel.apply_doppler(yd, velocity_kmh, fs)
        if sig_ref is None:
            sig_ref = np.mean(np.abs(yd) ** 2) or 1.0
        snr_lin = 10 ** (snr_db / 10)
        noise_pow = sig_ref / snr_lin
        noise = (np.random.randn(len(yd)) + 1j * np.random.randn(len(yd))) * np.sqrt(noise_pow / 2)
        rx_list.append(yd + alpha * yi + noise)
    return rx_list


def irc_combine(Y_list, Hd_list, Hi_list, sc_map, snr_db, sir_db):
    """Combinación con rechazo de interferencia (IRC) por subportadora.

    Para cada subportadora se forma la matriz de covarianza de interferencia
    más ruido R = P_i · h_i h_i^H + sigma^2 I, y se calculan los pesos
    w = R^{-1} h_d. El producto w^H h_i tiende a cero, de modo que el
    interferente se suprime, a diferencia del MRC que solo maximiza la SNR
    sin tener en cuenta la interferencia. La salida se normaliza para que la
    estimación del símbolo sea no sesgada.
    """
    NR = len(Y_list)
    n_frames = Y_list[0].shape[0]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)
    sigma2 = 10 ** (-snr_db / 10)
    Pi = 10 ** (-sir_db / 10)

    out = np.zeros(n_frames * n_data, dtype=complex)
    I = np.eye(NR, dtype=complex)

    for i in range(n_frames):
        for di, k in enumerate(data_idx):
            hd = np.array([Hd_list[r][i][k] for r in range(NR)], dtype=complex)
            hi = np.array([Hi_list[r][i][k] for r in range(NR)], dtype=complex)
            r_vec = np.array([Y_list[r][i][k] for r in range(NR)], dtype=complex)

            R = Pi * np.outer(hi, np.conj(hi)) + sigma2 * I
            try:
                w = np.linalg.solve(R, hd)
            except np.linalg.LinAlgError:
                w = hd
            denom = np.vdot(w, hd)  # w^H h_d
            if abs(denom) < 1e-12:
                denom = 1e-12
            out[i * n_data + di] = np.vdot(w, r_vec) / denom

    return out


# ===================================================================
# Multiplexación Espacial MIMO 2x2 con receptor Zero-Forcing
# ===================================================================

def spatial_mux_tx(bits_in, Nfft, cp_len, sc_map, pilot_value, M):
    """Transmisor de multiplexación espacial 2x2.

    Divide el flujo en dos capas independientes que se envían de forma
    simultánea por dos antenas sobre las mismas subportadoras, lo que duplica
    la tasa de datos. Cada antena usa pilotos ortogonales para permitir la
    estimación del canal 2x2 en el receptor. La potencia se reparte entre las
    dos antenas para una comparación justa frente a SISO.
    """
    n_data = sc_map["n_data"]
    k = int(np.log2(M))
    bits_per_ofdm = 2 * n_data * k
    n_ofdm = int(np.ceil(len(bits_in) / bits_per_ofdm))
    total = n_ofdm * bits_per_ofdm
    if total > len(bits_in):
        bits_in = np.pad(bits_in, (0, total - len(bits_in)))

    syms = ofdm_tx.qam_mod(bits_in, M)  # 2*n_data*n_ofdm símbolos
    syms = syms.reshape(n_ofdm, 2 * n_data)

    data_idx = sc_map["data_indices"]
    pilot_idx = sc_map["pilot_indices"]
    p1 = pilot_idx[::2]
    p2 = pilot_idx[1::2]
    scale = 1.0 / np.sqrt(2.0)
    sym_len = Nfft + cp_len

    tx1 = np.zeros(n_ofdm * sym_len, dtype=complex)
    tx2 = np.zeros(n_ofdm * sym_len, dtype=complex)

    for i in range(n_ofdm):
        layer1 = syms[i, :n_data]
        layer2 = syms[i, n_data:2 * n_data]
        X1 = np.zeros(Nfft, dtype=complex)
        X2 = np.zeros(Nfft, dtype=complex)
        X1[data_idx] = layer1
        X2[data_idx] = layer2
        X1[p1] = pilot_value
        X2[p2] = pilot_value
        X1 *= scale
        X2 *= scale
        x1 = np.fft.ifft(X1)
        x2 = np.fft.ifft(X2)
        off = i * sym_len
        if cp_len > 0:
            tx1[off:off + cp_len] = x1[-cp_len:]
            tx1[off + cp_len:off + sym_len] = x1
            tx2[off:off + cp_len] = x2[-cp_len:]
            tx2[off + cp_len:off + sym_len] = x2
        else:
            tx1[off:off + sym_len] = x1
            tx2[off:off + sym_len] = x2

    return tx1, tx2, n_ofdm


def spatial_mux_zf_decode(Y_r0, Y_r1, H, sc_map, M):
    """Receptor Zero-Forcing para MIMO 2x2.

    Para cada subportadora resuelve s_hat = H^{-1} r, donde H es la matriz de
    canal 2x2 (filas = antenas RX, columnas = capas TX) y r contiene las dos
    observaciones recibidas. De este modo separa las dos capas y recupera
    ambos flujos a la vez. El precio de la mayor tasa es la ausencia de
    diversidad y un realce del ruido en subportadoras mal condicionadas.

    H se entrega como H[t][r] -> arreglo (frame) por capa TX t y antena RX r.
    """
    n_frames = Y_r0.shape[0]
    data_idx = sc_map["data_indices"]
    n_data = len(data_idx)

    layer1 = np.zeros(n_frames * n_data, dtype=complex)
    layer2 = np.zeros(n_frames * n_data, dtype=complex)

    for i in range(n_frames):
        for di, k in enumerate(data_idx):
            Hmat = np.array([
                [H[0][0][i][k], H[1][0][i][k]],
                [H[0][1][i][k], H[1][1][i][k]],
            ], dtype=complex)
            r_vec = np.array([Y_r0[i][k], Y_r1[i][k]], dtype=complex)
            try:
                s = np.linalg.solve(Hmat, r_vec)
            except np.linalg.LinAlgError:
                s = np.array([0j, 0j])
            layer1[i * n_data + di] = s[0]
            layer2[i * n_data + di] = s[1]

    # Reconstituir el orden de símbolos por símbolo OFDM: capa1 luego capa2
    syms = np.zeros(n_frames * 2 * n_data, dtype=complex)
    for i in range(n_frames):
        syms[i * 2 * n_data: i * 2 * n_data + n_data] = layer1[i * n_data:(i + 1) * n_data]
        syms[i * 2 * n_data + n_data: (i + 1) * 2 * n_data] = layer2[i * n_data:(i + 1) * n_data]

    return ofdm_rx.qam_demod(syms, M)


def run_spatial_mux(bits_in, Nfft, cp_len, sc_map, pilot_value, M,
                    profile, taps_L, snr_db, velocity_kmh, correlation="low"):
    """Cadena completa de multiplexación espacial 2x2 y BER resultante."""
    fs = Nfft * DELTA_F
    tx1, tx2, n_ofdm = spatial_mux_tx(bits_in, Nfft, cp_len, sc_map, pilot_value, M)

    # Canal 2x2: ch[t][r] desde antena TX t hacia antena RX r
    ch = [generate_correlated_channels(2, profile, taps_L, correlation) for _ in range(2)]

    len_sig = len(tx1)
    snr_lin = 10 ** (snr_db / 10)
    rx = []
    for r in range(2):
        y = np.convolve(tx1, ch[0][r], mode="full")[:len_sig] + \
            np.convolve(tx2, ch[1][r], mode="full")[:len_sig]
        y = ofdm_channel.apply_doppler(y, velocity_kmh, fs)
        sig_pow = np.mean(np.abs(y) ** 2) or 1.0
        noise = (np.random.randn(len_sig) + 1j * np.random.randn(len_sig)) * \
            np.sqrt((sig_pow / snr_lin) / 2)
        rx.append(y + noise)

    Y0 = ofdm_rx.ofdm_rx_block(rx[0], Nfft, cp_len)
    Y1 = ofdm_rx.ofdm_rx_block(rx[1], Nfft, cp_len)

    pilot_idx = sc_map["pilot_indices"]
    sc_a1 = dict(sc_map); sc_a1["pilot_indices"] = pilot_idx[::2]
    sc_a2 = dict(sc_map); sc_a2["pilot_indices"] = pilot_idx[1::2]

    # H[t][r]: capa TX t estimada en antena RX r desde sus pilotos ortogonales
    H = [[None, None], [None, None]]
    H[0][0] = ofdm_rx.estimate_channel_from_pilots(Y0, sc_a1, pilot_value, Nfft)
    H[0][1] = ofdm_rx.estimate_channel_from_pilots(Y1, sc_a1, pilot_value, Nfft)
    H[1][0] = ofdm_rx.estimate_channel_from_pilots(Y0, sc_a2, pilot_value, Nfft)
    H[1][1] = ofdm_rx.estimate_channel_from_pilots(Y1, sc_a2, pilot_value, Nfft)

    bits_rx = spatial_mux_zf_decode(Y0, Y1, H, sc_map, M)
    n = min(len(bits_rx), len(bits_in))
    return float(np.mean(bits_rx[:n] != bits_in[:n]))


# ===================================================================
# Análisis comparativo unificado
# ===================================================================

ALL_TECHNIQUES = ["SISO", "SIMO-MRC", "SIMO-IRC", "MISO-SFBC", "Beamforming", "MIMO"]

LAB_COLORS = {
    "SISO": "#7f8c8d", "SIMO-MRC": "#2ecc71", "SIMO-IRC": "#16a085",
    "MISO-SFBC": "#e67e22", "Beamforming": "#8e44ad", "MIMO": "#c0392b",
}
LAB_MARKERS = {
    "SISO": "o", "SIMO-MRC": "s", "SIMO-IRC": "v",
    "MISO-SFBC": "D", "Beamforming": "^", "MIMO": "P",
}


def run_mimo_lab_analysis(
    bits_tx, Nfft, cp_len, sc_map, pilot_value,
    profile, taps_L, snr_list, n_mc, velocity, M,
    NR=2, NT=4, correlation="low", sir_db=3.0, techniques=None,
):
    """BER vs SNR e instrumentación de tiempo para todas las técnicas activas.

    Devuelve, por técnica, la media de BER con intervalo de confianza del
    95 por ciento y el tiempo de procesamiento medio por punto de SNR. Las
    técnicas se pueden activar o desactivar con la lista techniques.
    """
    if techniques is None:
        techniques = ALL_TECHNIQUES
    fs = Nfft * DELTA_F
    n_data = sc_map["n_data"]
    k = int(np.log2(M))
    bits_per_ofdm = n_data * k
    pad = (-len(bits_tx)) % bits_per_ofdm
    bits_in = np.pad(bits_tx, (0, pad)) if pad else bits_tx.copy()

    pilot_idx = sc_map["pilot_indices"]
    sc_a1 = dict(sc_map); sc_a1["pilot_indices"] = pilot_idx[::2]
    sc_a2 = dict(sc_map); sc_a2["pilot_indices"] = pilot_idx[1::2]
    data_idx = sc_map["data_indices"]

    results = {t: {"mean": [], "ci": []} for t in techniques}
    times = {t: [] for t in techniques}

    def ber(Xhat):
        bh = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]
        if len(bh) < len(bits_in):
            bh = np.pad(bh, (0, len(bits_in) - len(bh)))
        return float(np.mean(bh != bits_in))

    for snr_db in snr_list:
        acc = {t: [] for t in techniques}
        tacc = {t: 0.0 for t in techniques}

        for _ in range(n_mc):
            s = ofdm_tx.qam_mod(bits_in, M)

            if "SISO" in techniques:
                t0 = time.perf_counter()
                tx, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
                h = ofdm_channel.get_channel_profile(profile, taps_L)
                rx, _ = ofdm_channel.apply_channel(tx, h, snr_db, velocity, fs)
                Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
                X, _ = ofdm_rx.equalize_with_pilots(Y, sc_map, pilot_value, Nfft)
                acc["SISO"].append(ber(X)); tacc["SISO"] += time.perf_counter() - t0

            if "SIMO-MRC" in techniques:
                t0 = time.perf_counter()
                tx, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
                chs = generate_correlated_channels(NR, profile, taps_L, correlation)
                rxl = ofdm_channel.apply_channel_mimo(tx, chs, snr_db, velocity, fs)
                Yl = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rxl]
                Hl = [ofdm_rx.estimate_channel_from_pilots(Y, sc_map, pilot_value, Nfft) for Y in Yl]
                Xm, _ = ofdm_rx.combine_mrc(Yl, Hl, sc_map)
                acc["SIMO-MRC"].append(ber(Xm)); tacc["SIMO-MRC"] += time.perf_counter() - t0

            if "SIMO-IRC" in techniques:
                t0 = time.perf_counter()
                tx, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
                bi = np.random.randint(0, 2, len(bits_in)).astype(np.uint8)
                txi, _, _ = ofdm_tx.ofdm_tx_block(ofdm_tx.qam_mod(bi, M),
                                                  Nfft, cp_len, sc_map, pilot_value)
                chd = generate_correlated_channels(NR, profile, taps_L, correlation)
                chi = generate_correlated_channels(NR, profile, taps_L, "low")
                rxl = apply_simo_with_interference(tx, chd, txi, chi, snr_db, sir_db, velocity, fs)
                Yl = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rxl]
                Hd = [np.array([np.fft.fft(chd[r], Nfft)] * Yl[0].shape[0]) for r in range(NR)]
                Hi = [np.array([np.fft.fft(chi[r], Nfft)] * Yl[0].shape[0]) for r in range(NR)]
                Xi = irc_combine(Yl, Hd, Hi, sc_map, snr_db, sir_db)
                acc["SIMO-IRC"].append(ber(Xi)); tacc["SIMO-IRC"] += time.perf_counter() - t0

            if "MISO-SFBC" in techniques:
                t0 = time.perf_counter()
                tx1, tx2, _, _, _, _ = ofdm_tx.sfbc_tx_2ant(s, Nfft, cp_len, sc_map, pilot_value)
                hm = ofdm_channel.generate_miso_channels(2, profile, taps_L)
                rxm, _ = ofdm_channel.apply_channel_miso([tx1, tx2], hm, snr_db, velocity, fs)
                Ym = ofdm_rx.ofdm_rx_block(rxm, Nfft, cp_len)
                H1 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a1, pilot_value, Nfft)
                H2 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a2, pilot_value, Nfft)
                Xs = ofdm_rx.sfbc_decode_2ant(Ym, H1, H2, sc_map, data_idx)
                acc["MISO-SFBC"].append(ber(Xs)); tacc["MISO-SFBC"] += time.perf_counter() - t0

            if "Beamforming" in techniques:
                t0 = time.perf_counter()
                chb = ofdm_beamforming.generate_beamforming_channels(NT, profile, taps_L, correlation)
                txb, _, _, _ = ofdm_beamforming.beamforming_tx_block(s, Nfft, cp_len, sc_map, pilot_value, chb)
                rxb = ofdm_beamforming.apply_channel_beamforming(txb, chb, snr_db, velocity, fs)
                Yb = ofdm_rx.ofdm_rx_block(rxb, Nfft, cp_len)
                Xb, _ = ofdm_rx.equalize_with_pilots(Yb, sc_map, pilot_value, Nfft)
                acc["Beamforming"].append(ber(Xb)); tacc["Beamforming"] += time.perf_counter() - t0

            if "Mux-Espacial" in techniques:
                t0 = time.perf_counter()
                b = run_spatial_mux(bits_in, Nfft, cp_len, sc_map, pilot_value, M,
                                    profile, taps_L, snr_db, velocity, correlation)
                acc["Mux-Espacial"].append(b); tacc["Mux-Espacial"] += time.perf_counter() - t0

        for t in techniques:
            m = np.mean(acc[t]); sd = np.std(acc[t], ddof=1) if n_mc > 1 else 0
            results[t]["mean"].append(m)
            results[t]["ci"].append(1.96 * sd / np.sqrt(n_mc))
            times[t].append(tacc[t] / max(1, n_mc))

    return {"ber": results, "times": times, "snr_list": list(snr_list),
            "NR": NR, "NT": NT, "correlation": correlation, "sir_db": sir_db}


# ===================================================================
# Imagen recuperada por cada modo a SNR bajo
# ===================================================================

def transmit_image_all_modes(
    bits_tx, img_arr, Nfft, cp_len, sc_map, pilot_value,
    profile, taps_L, snr_db, velocity, M, NR=2, NT=4, correlation="low", sir_db=3.0,
):
    """Reconstruye la imagen por cada modo a una SNR baja para comparar.

    Devuelve, por técnica, la imagen recuperada y sus métricas, evidenciando
    cómo la diversidad espacial preserva mejor la estructura de la imagen.
    """
    fs = Nfft * DELTA_F
    n_bits = len(bits_tx)
    k = int(np.log2(M))
    n_data = sc_map["n_data"]
    bpo = n_data * k
    n_need = int(np.ceil(n_bits / bpo))
    total_cap = n_need * bpo
    bits_in = np.pad(bits_tx, (0, total_cap - n_bits)) if total_cap > n_bits else bits_tx.copy()
    shape = img_arr.shape

    pilot_idx = sc_map["pilot_indices"]
    sc_a1 = dict(sc_map); sc_a1["pilot_indices"] = pilot_idx[::2]
    sc_a2 = dict(sc_map); sc_a2["pilot_indices"] = pilot_idx[1::2]
    data_idx = sc_map["data_indices"]
    s = ofdm_tx.qam_mod(bits_in, M)

    out = {"img_orig": img_arr, "snr_db": snr_db}

    def store(name, bits_rx):
        img = ofdm_utils._bits_to_image(bits_rx, n_bits, total_cap, shape)
        mse, psnr = ofdm_utils._img_metrics(img_arr, img)
        b = np.asarray(bits_rx).astype(np.uint8).ravel()
        if len(b) < total_cap:
            b = np.pad(b, (0, total_cap - len(b)))
        out[name] = {"img": img, "psnr": psnr, "mse": mse,
                     "ber": float(np.mean(b[:n_bits] != bits_tx))}

    # SISO
    tx, _, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)
    h = ofdm_channel.get_channel_profile(profile, taps_L)
    rx, _ = ofdm_channel.apply_channel(tx, h, snr_db, velocity, fs)
    Y = ofdm_rx.ofdm_rx_block(rx, Nfft, cp_len)
    X, _ = ofdm_rx.equalize_with_pilots(Y, sc_map, pilot_value, Nfft)
    store("SISO", ofdm_rx.qam_demod(X, M))

    # SIMO-MRC
    chs = generate_correlated_channels(NR, profile, taps_L, correlation)
    rxl = ofdm_channel.apply_channel_mimo(tx, chs, snr_db, velocity, fs)
    Yl = [ofdm_rx.ofdm_rx_block(r, Nfft, cp_len) for r in rxl]
    Hl = [ofdm_rx.estimate_channel_from_pilots(Y, sc_map, pilot_value, Nfft) for Y in Yl]
    Xm, _ = ofdm_rx.combine_mrc(Yl, Hl, sc_map)
    store("SIMO-MRC", ofdm_rx.qam_demod(Xm, M))

    # MISO-SFBC
    tx1, tx2, _, _, _, _ = ofdm_tx.sfbc_tx_2ant(s, Nfft, cp_len, sc_map, pilot_value)
    hm = ofdm_channel.generate_miso_channels(2, profile, taps_L)
    rxm, _ = ofdm_channel.apply_channel_miso([tx1, tx2], hm, snr_db, velocity, fs)
    Ym = ofdm_rx.ofdm_rx_block(rxm, Nfft, cp_len)
    H1 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a1, pilot_value, Nfft)
    H2 = ofdm_rx.estimate_channel_from_pilots(Ym, sc_a2, pilot_value, Nfft)
    Xsf = ofdm_rx.sfbc_decode_2ant(Ym, H1, H2, sc_map, data_idx)
    store("MISO-SFBC", ofdm_rx.qam_demod(Xsf, M))

    # Beamforming
    chb = ofdm_beamforming.generate_beamforming_channels(NT, profile, taps_L, correlation)
    txb, _, _, _ = ofdm_beamforming.beamforming_tx_block(s, Nfft, cp_len, sc_map, pilot_value, chb)
    rxb = ofdm_beamforming.apply_channel_beamforming(txb, chb, snr_db, velocity, fs)
    Yb = ofdm_rx.ofdm_rx_block(rxb, Nfft, cp_len)
    Xb, _ = ofdm_rx.equalize_with_pilots(Yb, sc_map, pilot_value, Nfft)
    store("Beamforming", ofdm_rx.qam_demod(Xb, M))

    # MIMO (multiplexación espacial 2x2 con receptor Zero-Forcing)
    tx1m, tx2m, _ = spatial_mux_tx(bits_in, Nfft, cp_len, sc_map, pilot_value, M)
    ch_mux = [generate_correlated_channels(2, profile, taps_L, correlation) for _ in range(2)]
    len_sig = len(tx1m)
    snr_lin = 10 ** (snr_db / 10)
    rx_mux = []
    for r in range(2):
        y = np.convolve(tx1m, ch_mux[0][r], mode="full")[:len_sig] + \
            np.convolve(tx2m, ch_mux[1][r], mode="full")[:len_sig]
        y = ofdm_channel.apply_doppler(y, velocity, fs)
        sp = np.mean(np.abs(y) ** 2) or 1.0
        noise = (np.random.randn(len_sig) + 1j * np.random.randn(len_sig)) * \
            np.sqrt((sp / snr_lin) / 2)
        rx_mux.append(y + noise)
    Y0m = ofdm_rx.ofdm_rx_block(rx_mux[0], Nfft, cp_len)
    Y1m = ofdm_rx.ofdm_rx_block(rx_mux[1], Nfft, cp_len)
    Hm = [[None, None], [None, None]]
    Hm[0][0] = ofdm_rx.estimate_channel_from_pilots(Y0m, sc_a1, pilot_value, Nfft)
    Hm[0][1] = ofdm_rx.estimate_channel_from_pilots(Y1m, sc_a1, pilot_value, Nfft)
    Hm[1][0] = ofdm_rx.estimate_channel_from_pilots(Y0m, sc_a2, pilot_value, Nfft)
    Hm[1][1] = ofdm_rx.estimate_channel_from_pilots(Y1m, sc_a2, pilot_value, Nfft)
    store("MIMO", spatial_mux_zf_decode(Y0m, Y1m, Hm, sc_map, M))

    return out


# ===================================================================
# PAPR / potencia instantánea
# ===================================================================

def papr_distribution_data(Nfft, cp_len, sc_map, pilot_value, M, NT=4, n_sym=200):
    """Distribución de PAPR de OFDM frente a Beamforming y multiplexación.

    Muestra cómo cambia la relación entre potencia pico y promedio al pasar de
    una sola antena a configuraciones multi-antena.
    """
    n_data = sc_map["n_data"]
    k = int(np.log2(M))
    bits = np.random.randint(0, 2, n_sym * 2 * n_data * k).astype(np.uint8)
    s = ofdm_tx.qam_mod(bits, M)

    _, papr_ofdm, _ = ofdm_tx.ofdm_tx_block(s, Nfft, cp_len, sc_map, pilot_value)

    chb = ofdm_beamforming.generate_beamforming_channels(NT, "Rayleigh (NLoS)", 2, "low")
    txb, papr_bf, _, _ = ofdm_beamforming.beamforming_tx_block(s, Nfft, cp_len, sc_map, pilot_value, chb)

    tx1, tx2, _ = spatial_mux_tx(bits, Nfft, cp_len, sc_map, pilot_value, M)
    sym_len = Nfft + cp_len
    n_ofdm = len(tx1) // sym_len
    papr_mux = []
    for i in range(n_ofdm):
        seg = tx1[i * sym_len + cp_len: (i + 1) * sym_len]
        if len(seg):
            papr_mux.append(ofdm_tx.calculate_papr(seg))

    return {"OFDM (SISO)": papr_ofdm, "Beamforming": papr_bf, "Mux-Espacial": papr_mux}


# ===================================================================
# Visualizaciones
# ===================================================================

def plot_lab_ber(ax, lab, M_name):
    """BER vs SNR de todas las técnicas activas en una sola figura."""
    snr = np.array(lab["snr_list"])
    for t, d in lab["ber"].items():
        means = np.array(d["mean"]); cis = np.array(d["ci"])
        mask = means > 0
        color = LAB_COLORS.get(t, None); marker = LAB_MARKERS.get(t, "o")
        if np.any(mask):
            ax.semilogy(snr[mask], means[mask], marker=marker, color=color, label=t, markersize=5)
            ax.fill_between(snr[mask], np.maximum(means[mask] - cis[mask], 1e-10),
                            means[mask] + cis[mask], alpha=0.12, color=color)
        else:
            ax.semilogy([], [], marker=marker, color=color, label=f"{t} (BER=0)")
    ax.set_title(f"BER vs SNR — Comparativa MIMO ({M_name})")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)


def plot_processing_time(ax, lab):
    """Tiempo de procesamiento por punto de SNR para cada técnica."""
    snr = np.array(lab["snr_list"])
    for t, ts in lab["times"].items():
        ax.plot(snr, np.array(ts) * 1e3, marker=LAB_MARKERS.get(t, "o"),
                color=LAB_COLORS.get(t, None), label=t, markersize=4)
    ax.set_title("Tiempo de procesamiento vs SNR")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Tiempo por iteración (ms)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)


def plot_papr_distribution(ax, papr_dict):
    """CCDF de PAPR de OFDM frente a las configuraciones multi-antena."""
    colors = {"OFDM (SISO)": "#7f8c8d", "Beamforming": "#8e44ad", "Mux-Espacial": "#c0392b"}
    for name, papr in papr_dict.items():
        p = np.sort(np.array(papr))
        ccdf = 1.0 - np.arange(len(p)) / len(p)
        ax.semilogy(p, ccdf, label=name, color=colors.get(name, None), linewidth=1.5)
    ax.set_title("Distribución de PAPR (CCDF)")
    ax.set_xlabel("PAPR (dB)"); ax.set_ylabel("P(PAPR > abscisa)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
