# ofdm_mcw_sic.py
# MIMO N x N con transmisión Multi-Codeword (MCW) adaptativa y receptor SIC.
#
# Implementa la variante avanzada de multiplexación espacial de la práctica de
# MIMO, con un número de antenas configurable de 2 hasta 8, como admite LTE
# Advanced:
#
#   - Transmisión Multi-Codeword (MCW): cada capa espacial se codifica y modula
#     de forma independiente. La modulación de cada capa se elige por adaptación
#     de enlace en función de los eigenvalores del canal (CSI ideal en el TX):
#     64QAM si el eigenvalor efectivo supera T1, 16QAM entre T1 y T2, y QPSK por
#     debajo de T2. Los flujos se mapean a las N antenas transmisoras.
#
#   - Transmisión Single-Codeword (SCW): una única fuente de bits con modulación
#     fija (16QAM) que se reparte entre las N antenas, como la multiplexación
#     espacial pura de la Práctica 10.
#
#   - Receptor SIC (Successive Interference Cancellation): detecta primero el
#     flujo con la modulación más robusta, lo recodifica y lo resta de la señal
#     recibida, y detecta los siguientes flujos con la interferencia mitigada.
#
# El canal es una matriz de Rayleigh N x N independiente por cada subportadora
# OFDM, con conocimiento perfecto del canal en el transmisor y el receptor.

import time

import numpy as np
import matplotlib.pyplot as plt

import ofdm_tx
import ofdm_rx
import ofdm_utils
import ofdm_params
import LTE_TURBO


# ===================================================================
# Canal MIMO N x N por subportadora y eigenvalores
# ===================================================================

def gen_channel(n_sc, N):
    """Genera una matriz de canal Rayleigh N x N por cada subportadora."""
    return (np.random.randn(n_sc, N, N) + 1j * np.random.randn(n_sc, N, N)) / np.sqrt(2.0)


def channel_eigenvalues(H):
    """Eigenvalores de H^H H por subportadora, ordenados de mayor a menor."""
    Hh = np.conj(np.transpose(H, (0, 2, 1)))
    HhH = Hh @ H
    ev = np.linalg.eigvalsh(HhH)          # ascendente (n_sc, N)
    return np.sort(ev, axis=1)[:, ::-1]   # descendente


def assign_modulation(eff_snr_db, T1=15.0, T2=8.0):
    """Adaptación de enlace: elige la modulación según el eigenvalor efectivo."""
    if eff_snr_db > T1:
        return 64
    if eff_snr_db > T2:
        return 16
    return 4


def _demod_layer(syms, M):
    return ofdm_rx.qam_demod(np.asarray(syms), M)


# ===================================================================
# Receptor SIC (cancelación sucesiva de interferencia) para N capas
# ===================================================================

def sic_receiver(r, H, M_layers, sigma2, order=None):
    """Receptor SIC de cancelación sucesiva para MIMO N x N.

    Detecta las capas en orden de modulación más robusta primero. En cada paso
    estima el flujo con un filtro MMSE que anula solo las capas aún no
    canceladas, recodifica (remodula) las decisiones y las resta de la señal
    residual antes de detectar el siguiente flujo.
    """
    N = H.shape[2]
    n_sc = r.shape[0]
    if order is None:
        order = sorted(range(N), key=lambda i: M_layers[i])  # robusta primero

    r_res = r.copy()
    shat = [None] * N

    for step, layer in enumerate(order):
        remaining = order[step:]
        Hsub = H[:, :, remaining]                       # (n_sc, N, n_rem)
        Hh = np.conj(np.transpose(Hsub, (0, 2, 1)))      # (n_sc, n_rem, N)
        n_rem = len(remaining)
        A = Hh @ Hsub + sigma2 * np.eye(n_rem)[None]     # (n_sc, n_rem, n_rem)
        W = np.linalg.solve(A, Hh)                       # (n_sc, n_rem, N)
        s_est = np.einsum('nij,nj->ni', W, r_res)        # (n_sc, n_rem)

        li = remaining.index(layer)
        bits_hat = _demod_layer(s_est[:, li], M_layers[layer])
        x_hat = ofdm_tx.qam_mod(bits_hat, M_layers[layer])[:n_sc]
        shat[layer] = x_hat
        r_res = r_res - H[:, :, layer] * x_hat[:, None]  # cancelación

    return shat, order


def mmse_joint(r, H, sigma2):
    """Detección conjunta MMSE de todas las capas, sin cancelación (referencia)."""
    N = H.shape[2]
    Hh = np.conj(np.transpose(H, (0, 2, 1)))
    A = Hh @ H + sigma2 * np.eye(N)[None]
    W = np.linalg.solve(A, Hh)
    return np.einsum('nij,nj->ni', W, r)   # (n_sc, N)


# ===================================================================
# Cadena MCW y SCW por bloque
# ===================================================================

def _run_block_mcw(H, snr_db, T1, T2):
    """Un bloque MCW: adaptación por eigenvalores, TX independiente y RX SIC."""
    n_sc, N = H.shape[0], H.shape[2]
    snr_lin = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_lin

    ev = channel_eigenvalues(H)
    ev_mean = np.mean(ev, axis=0)
    eff_snr_db = 10 * np.log10(snr_lin * ev_mean + 1e-12)
    M_layers = [assign_modulation(eff_snr_db[i], T1, T2) for i in range(N)]

    bits_tx = []
    X = np.zeros((n_sc, N), dtype=complex)
    for layer in range(N):
        k = int(np.log2(M_layers[layer]))
        b = np.random.randint(0, 2, n_sc * k).astype(np.uint8)
        bits_tx.append(b)
        X[:, layer] = ofdm_tx.qam_mod(b, M_layers[layer])[:n_sc]

    noise = (np.random.randn(n_sc, N) + 1j * np.random.randn(n_sc, N)) * np.sqrt(sigma2 / 2)
    r = np.einsum('nij,nj->ni', H, X) + noise

    shat_sic, order = sic_receiver(r, H, M_layers, sigma2)
    shat_mmse = mmse_joint(r, H, sigma2)
    return {"M_layers": M_layers, "order": order, "bits_tx": bits_tx,
            "shat_sic": shat_sic, "shat_mmse": shat_mmse}


def _ber_layer(bits_tx, shat, M):
    bhat = _demod_layer(shat, M)[:len(bits_tx)]
    if len(bhat) < len(bits_tx):
        bhat = np.pad(bhat, (0, len(bits_tx) - len(bhat)))
    return float(np.mean(bhat != bits_tx))


def _run_block_scw(H, snr_db, M=16):
    """Un bloque SCW: una fuente con modulación fija repartida en N antenas."""
    n_sc, N = H.shape[0], H.shape[2]
    snr_lin = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    k = int(np.log2(M))

    bits = np.random.randint(0, 2, N * n_sc * k).astype(np.uint8)
    syms = ofdm_tx.qam_mod(bits, M)
    X = syms[:N * n_sc].reshape(n_sc, N)

    noise = (np.random.randn(n_sc, N) + 1j * np.random.randn(n_sc, N)) * np.sqrt(sigma2 / 2)
    r = np.einsum('nij,nj->ni', H, X) + noise
    shat = mmse_joint(r, H, sigma2)

    bhat = _demod_layer(shat.reshape(-1), M)[:len(bits)]
    if len(bhat) < len(bits):
        bhat = np.pad(bhat, (0, len(bits) - len(bhat)))
    return float(np.mean(bhat != bits))


# ===================================================================
# Simulación: throughput y BER por capa
# ===================================================================

def run_mcw_vs_scw(snr_list, N=2, n_frames=12, n_sc=300, T1=15.0, T2=8.0, M_scw=16):
    """Throughput y BER por capa de MCW adaptativo frente a SCW fijo.

    El throughput se mide en bits por uso de canal considerando solo los bits
    recibidos correctamente. MCW adapta la modulación de cada capa a la calidad
    del canal, mientras que SCW usa una modulación fija. Con N antenas, el
    número de capas espaciales crece y con él la tasa máxima alcanzable.
    """
    res = {
        "snr": list(snr_list), "N": N,
        "thr_mcw": [], "thr_scw": [],
        "ber_first": [], "ber_last_sic": [], "ber_last_mmse": [],
        "mod_first": [], "mod_last": [],
        "sim_time_ms": [],           # tiempo de simulación por nivel de SNR
        "ber_layers": [],            # BER de cada capa (índice de autovalor) por SNR
        "mod_layers": [],            # modulación de cada capa por SNR
    }
    for snr_db in snr_list:
        t0 = time.perf_counter()
        thr_mcw = thr_scw = 0.0
        bf = bls = blm = 0.0
        mf_acc = []; ml_acc = []
        ber_lay = np.zeros(N)               # BER acumulada por capa (con SIC)
        mod_lay = [[] for _ in range(N)]    # modulaciones por capa a lo largo de las tramas
        for _ in range(n_frames):
            H = gen_channel(n_sc, N)
            blk = _run_block_mcw(H, snr_db, T1, T2)
            M_layers = blk["M_layers"]
            order = blk["order"]
            l_first, l_last = order[0], order[-1]

            # Throughput y BER de cada capa espacial (capa i = i-ésimo autovalor)
            for layer in range(N):
                ber = _ber_layer(blk["bits_tx"][layer], blk["shat_sic"][layer], M_layers[layer])
                thr_mcw += np.log2(M_layers[layer]) * (1 - ber)
                ber_lay[layer] += ber
                mod_lay[layer].append(M_layers[layer])

            # BER de la primera capa detectada y de la última con y sin SIC
            bf += _ber_layer(blk["bits_tx"][l_first], blk["shat_sic"][l_first], M_layers[l_first])
            bls += _ber_layer(blk["bits_tx"][l_last], blk["shat_sic"][l_last], M_layers[l_last])
            blm += _ber_layer(blk["bits_tx"][l_last], blk["shat_mmse"][:, l_last], M_layers[l_last])
            mf_acc.append(M_layers[l_first]); ml_acc.append(M_layers[l_last])

            # SCW en el mismo canal
            ber_scw = _run_block_scw(H, snr_db, M_scw)
            thr_scw += N * np.log2(M_scw) * (1 - ber_scw)

        res["thr_mcw"].append(thr_mcw / n_frames)
        res["thr_scw"].append(thr_scw / n_frames)
        res["ber_first"].append(bf / n_frames)
        res["ber_last_sic"].append(bls / n_frames)
        res["ber_last_mmse"].append(blm / n_frames)
        res["mod_first"].append(int(np.median(mf_acc)))
        res["mod_last"].append(int(np.median(ml_acc)))
        res["sim_time_ms"].append((time.perf_counter() - t0) * 1e3)
        res["ber_layers"].append((ber_lay / n_frames).tolist())
        res["mod_layers"].append([int(np.median(m)) for m in mod_lay])
    return res


def constellation_data(snr_db, N=2, n_sc=1500, T1=15.0, T2=8.0):
    """Constelaciones recibidas de dos capas representativas del sistema MCW.

    Se eligen la capa de mayor y la de menor modulación para evidenciar la
    naturaleza heterogénea del sistema. La estimación MMSE conjunta se usa para
    la visualización antes de la decisión.
    """
    H = gen_channel(n_sc, N)
    blk = _run_block_mcw(H, snr_db, T1, T2)
    M_layers = blk["M_layers"]
    s = blk["shat_mmse"]
    i_hi = int(np.argmax(M_layers))   # capa con mayor modulación
    i_lo = int(np.argmin(M_layers))   # capa con menor modulación
    if i_hi == i_lo and N > 1:
        i_lo = (i_hi + 1) % N
    return {"layer1": s[:, i_hi], "layer2": s[:, i_lo],
            "M1": M_layers[i_hi], "M2": M_layers[i_lo], "snr_db": snr_db, "N": N}


# ===================================================================
# Datos para las visualizaciones avanzadas de monitoreo
# ===================================================================

def eigenvalue_data(N=2, n_sc=220):
    """Magnitud de los eigenvalores de H^H H por subportadora.

    Permite identificar visualmente los desvanecimientos profundos: cuando el
    eigenvalor más pequeño se hunde, ese modo espacial casi no puede transportar
    información y la multiplexación espacial pierde efectividad en esa
    subportadora.
    """
    H = gen_channel(n_sc, N)
    ev = channel_eigenvalues(H)          # (n_sc, N) descendente
    return {"ev": ev, "N": N}


def papr_ccdf_data(n_symbols=300, bw_mhz=10.0, M=16, pilot_spacing=6):
    """CCDF del PAPR para OFDM frente a SC-FDM (precodificación DFT).

    La precodificación DFT del SC-FDM concentra la energía y reduce los picos de
    potencia instantánea, por lo que su curva CCDF queda desplazada hacia PAPR
    más bajos que la de OFDM.
    """
    Nfft, cp_len, N_used = ofdm_utils.get_nfft_cp(bw_mhz, "normal")
    sc_map = ofdm_utils.build_subcarrier_map(Nfft, N_used, pilot_spacing)
    pilot_value = ofdm_params.PILOT_AMPLITUDE
    n_data = sc_map["n_data"]
    M_dft = ofdm_tx.get_best_dft_size(n_data, Nfft)

    k = int(np.log2(M))
    sym_ofdm = ofdm_tx.qam_mod(np.random.randint(0, 2, n_symbols * n_data * k), M)
    _, papr_ofdm, _ = ofdm_tx.ofdm_tx_block(sym_ofdm, Nfft, cp_len, sc_map, pilot_value)

    sym_sc = ofdm_tx.qam_mod(np.random.randint(0, 2, n_symbols * M_dft * k), M)
    _, papr_sc, _ = ofdm_tx.scfdma_tx_block(sym_sc, Nfft, cp_len, sc_map, pilot_value, M_dft)

    papr_ofdm = np.asarray(papr_ofdm); papr_sc = np.asarray(papr_sc)
    gamma = np.linspace(0, max(papr_ofdm.max(), papr_sc.max()) + 0.5, 120)
    ccdf_ofdm = np.mean(papr_ofdm[:, None] > gamma[None, :], axis=0)
    ccdf_sc = np.mean(papr_sc[:, None] > gamma[None, :], axis=0)
    return {"gamma": gamma, "ccdf_ofdm": ccdf_ofdm, "ccdf_sc": ccdf_sc}


def llr_hist_data(K=512, snr_db=1.5, n_iter=6):
    """Distribución de las métricas suaves LLR antes y después de Turbo.

    A la salida del demodulador (canal) los LLR forman dos campanas solapadas:
    hay decisiones poco fiables cerca de cero. Tras las iteraciones del
    decodificador Turbo el LLR total se vuelve marcadamente bimodal y de gran
    magnitud, señal de que las decisiones se han 'limpiado'.
    """
    bits = np.random.randint(0, 2, K).astype(np.uint8)
    enc = LTE_TURBO.turbo_encode(bits)
    R = LTE_TURBO.code_rate(K)
    ebno = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / (2.0 * R * ebno)
    Lc = 2.0 / sigma2
    sigma = np.sqrt(sigma2)

    def tx(b):
        x = LTE_TURBO._bpsk(b)
        return x + sigma * np.random.randn(len(x))

    rs = tx(enc["sys"]); rp1 = tx(enc["par1"]); rp2 = tx(enc["par2"])
    rt1s = tx(enc["tail1_sys"]); rt1p = tx(enc["tail1_par"])
    rt2s = tx(enc["tail2_sys"]); rt2p = tx(enc["tail2_par"])

    _, L_total = LTE_TURBO.turbo_decode(
        Lc * rs, Lc * rp1, Lc * rp2,
        Lc * rt1s, Lc * rt1p, Lc * rt2s, Lc * rt2p,
        enc["interleaver"], n_iter, return_llr=True,
    )
    return {"llr_channel": Lc * rs, "llr_turbo": np.asarray(L_total), "snr_db": snr_db}


def _mimo_image_chain(bits, N, snr_db, M_layers):
    """Transmite un flujo de bits por N capas MCW con receptor SIC.

    Cada capa lleva su propia modulación (M_layers). Los bits se reparten en
    bloques contiguos, una capa detrás de otra, y se recuperan en el mismo orden
    tras la cancelación sucesiva de interferencia.
    """
    k_lay = [int(np.log2(m)) for m in M_layers]
    bits_per_sc = sum(k_lay)
    n_sc = int(np.ceil(len(bits) / bits_per_sc))
    cap = n_sc * bits_per_sc
    b = np.zeros(cap, dtype=np.uint8); b[:len(bits)] = bits

    H = gen_channel(n_sc, N)
    snr_lin = 10 ** (snr_db / 10.0); sigma2 = 1.0 / snr_lin

    X = np.zeros((n_sc, N), dtype=complex)
    off = 0
    for layer in range(N):
        nb = n_sc * k_lay[layer]
        X[:, layer] = ofdm_tx.qam_mod(b[off:off + nb], M_layers[layer])[:n_sc]
        off += nb

    noise = (np.random.randn(n_sc, N) + 1j * np.random.randn(n_sc, N)) * np.sqrt(sigma2 / 2)
    r = np.einsum('nij,nj->ni', H, X) + noise
    shat, _ = sic_receiver(r, H, M_layers, sigma2)

    rx = []
    for layer in range(N):
        rx.append(_demod_layer(shat[layer], M_layers[layer]))
    return np.concatenate(rx)[:len(bits)]


def _mimo_image_chain_scw(bits, N, snr_db, M=16):
    """Transmite un flujo de bits por N capas con modulación fija (SCW, MMSE)."""
    k = int(np.log2(M))
    n_sc = int(np.ceil(len(bits) / (N * k)))
    cap = n_sc * N * k
    b = np.zeros(cap, dtype=np.uint8); b[:len(bits)] = bits

    H = gen_channel(n_sc, N)
    snr_lin = 10 ** (snr_db / 10.0); sigma2 = 1.0 / snr_lin
    syms = ofdm_tx.qam_mod(b, M)[:n_sc * N]
    X = syms.reshape(n_sc, N)
    noise = (np.random.randn(n_sc, N) + 1j * np.random.randn(n_sc, N)) * np.sqrt(sigma2 / 2)
    r = np.einsum('nij,nj->ni', H, X) + noise
    shat = mmse_joint(r, H, sigma2)
    return _demod_layer(shat.reshape(-1), M)[:len(bits)]


def transmit_image_mcw_scw(img_arr, N=2, snr_db=15.0, T1=15.0, T2=8.0, M_scw=16):
    """Transmite la misma imagen por MCW adaptativo y por SCW fijo.

    Devuelve las imágenes recuperadas por cada esquema para compararlas
    visualmente junto con sus métricas de calidad (MSE y PSNR).
    """
    img_arr = np.asarray(img_arr, dtype=np.uint8)
    bits = np.unpackbits(img_arr.flatten())
    n_bits = len(bits)

    # Modulación por capa según un canal de sondeo (adaptación de enlace)
    ev_mean = np.mean(channel_eigenvalues(gen_channel(400, N)), axis=0)
    snr_lin = 10 ** (snr_db / 10.0)
    eff_snr_db = 10 * np.log10(snr_lin * ev_mean + 1e-12)
    M_layers = [assign_modulation(eff_snr_db[i], T1, T2) for i in range(N)]

    rx_mcw = _mimo_image_chain(bits, N, snr_db, M_layers)
    rx_scw = _mimo_image_chain_scw(bits, N, snr_db, M_scw)

    img_mcw = ofdm_utils._bits_to_image(rx_mcw, n_bits, n_bits, img_arr.shape)
    img_scw = ofdm_utils._bits_to_image(rx_scw, n_bits, n_bits, img_arr.shape)
    mse_mcw, psnr_mcw = ofdm_utils._img_metrics(img_arr, img_mcw)
    mse_scw, psnr_scw = ofdm_utils._img_metrics(img_arr, img_scw)
    return {
        "orig": img_arr, "mcw": img_mcw, "scw": img_scw,
        "psnr_mcw": psnr_mcw, "mse_mcw": mse_mcw,
        "psnr_scw": psnr_scw, "mse_scw": mse_scw,
        "M_layers": M_layers, "N": N, "snr_db": snr_db,
    }


def image_time_data(img_arr, N=2, M=16, n_data=600, t_sym_us=71.4):
    """Tiempo de transmisión al aire de una imagen: SISO frente a MIMO N x N.

    La multiplexación espacial envía N flujos simultáneos sobre las mismas
    subportadoras, así que transmite la imagen en aproximadamente 1/N del tiempo
    (menos símbolos OFDM al aire). Con N=2 la velocidad efectiva se duplica. Se
    modela el número de símbolos OFDM necesarios con una duración de símbolo de
    LTE (CP normal, t_sym_us) sobre n_data subportadoras de datos.
    """
    n_bits = int(np.asarray(img_arr).size) * 8
    k = int(np.log2(M))
    n_ofdm_siso = int(np.ceil(n_bits / (k * n_data)))
    n_ofdm_mimo = int(np.ceil(n_bits / (k * n_data * N)))
    t_siso = n_ofdm_siso * t_sym_us / 1e3      # ms
    t_mimo = n_ofdm_mimo * t_sym_us / 1e3
    speedup = t_siso / t_mimo if t_mimo > 0 else float(N)
    return {"t_siso_ms": t_siso, "t_mimo_ms": t_mimo, "speedup": speedup, "N": N}


# ===================================================================
# Visualización
# ===================================================================

_MODNAME = {4: "QPSK", 16: "16QAM", 64: "64QAM"}


def plot_throughput(ax, res):
    """Throughput total frente a la SNR: MCW adaptativo vs SCW fijo."""
    snr = np.array(res["snr"])
    N = res["N"]
    ax.plot(snr, res["thr_mcw"], "s-", color="#c0392b", linewidth=2, markersize=6,
            label=f"MCW adaptativo (SIC, {N} capas)")
    ax.plot(snr, res["thr_scw"], "o--", color="#2980b9", linewidth=2, markersize=6,
            label="SCW fijo (16QAM)")
    ax.set_title(f"Throughput total frente a SNR (MIMO {N}x{N})")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Throughput (bits/uso de canal)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)


def plot_ber_layers(ax, res):
    """BER por capa: la última capa mejora al aplicar SIC frente a MMSE."""
    snr = np.array(res["snr"])
    ax.semilogy(snr, np.maximum(res["ber_first"], 1e-6), "o-", color="#27ae60",
                label="Primera capa (robusta, SIC)", linewidth=2)
    ax.semilogy(snr, np.maximum(res["ber_last_mmse"], 1e-6), "^--", color="#e67e22",
                label="Última capa (sin SIC, MMSE)", linewidth=2)
    ax.semilogy(snr, np.maximum(res["ber_last_sic"], 1e-6), "s-", color="#c0392b",
                label="Última capa (con SIC)", linewidth=2)
    ax.set_title("BER por capa — mejora del último flujo con SIC")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)


def plot_constellations(ax1, ax2, cd):
    """Constelaciones recibidas independientes de dos capas heterogéneas."""
    for ax, key, M, num in ((ax1, "layer1", cd["M1"], "mayor"),
                            (ax2, "layer2", cd["M2"], "menor")):
        d = cd[key]
        ax.scatter(np.real(d), np.imag(d), s=4, alpha=0.3, color="#2980b9")
        lim = 1.8 if M <= 16 else 1.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_title(f"Capa de modulación {num} recibida ({_MODNAME.get(M, '')})",
                     fontsize=10)
        ax.set_xlabel("En fase"); ax.set_ylabel("Cuadratura")
        ax.grid(True, alpha=0.3); ax.set_aspect("equal", adjustable="box")


_MODCOLOR = {4: "#2980b9", 16: "#e67e22", 64: "#c0392b"}


def _pick_snr_index(res):
    """Índice de SNR donde las capas usan modulaciones distintas (o la mayor)."""
    for i in range(len(res["snr"]) - 1, -1, -1):
        if len(set(res["mod_layers"][i])) > 1:
            return i
    return len(res["snr"]) - 1


def plot_per_antenna(ax, res, snr_idx=None):
    """Modulación y BER de cada antena: qué envía y cómo se comporta cada flujo."""
    if snr_idx is None:
        snr_idx = _pick_snr_index(res)
    N = res["N"]
    mods = res["mod_layers"][snr_idx]
    bers = np.maximum(res["ber_layers"][snr_idx], 1e-6)
    x = np.arange(N)
    colors = [_MODCOLOR.get(m, "#7f8c8d") for m in mods]
    ax.bar(x, bers, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1.0)
    for xi, b, m in zip(x, bers, mods):
        ax.text(xi, b * 1.4, _MODNAME.get(m, ""), ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ant {i+1}\n(λ{i+1})" for i in range(N)], fontsize=8)
    ax.set_title(f"Modulación y BER por antena — SNR {res['snr'][snr_idx]} dB",
                 fontsize=10)
    ax.set_ylabel("BER de la capa (con SIC)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=_MODCOLOR[m]) for m in (4, 16, 64)]
    ax.legend(handles, ["QPSK", "16QAM", "64QAM"], fontsize=8, title="Modulación")
    ax.grid(True, which="both", axis="y", alpha=0.3)


def plot_eigenvalues(ax, ed):
    """Eigenvalores por subportadora: identifica los desvanecimientos profundos."""
    ev = ed["ev"]; N = ed["N"]
    n_sc = ev.shape[0]
    x = np.arange(n_sc)
    ev_db = 10 * np.log10(ev + 1e-12)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, N))
    for i in range(N):
        ax.plot(x, ev_db[:, i], color=cmap[i], linewidth=1.1,
                label=f"λ{i+1}" + (" (más fuerte)" if i == 0 else
                                    " (más débil)" if i == N - 1 else ""))
    # Resaltar los desvanecimientos profundos del modo más débil
    weak = ev_db[:, -1]
    thr = np.percentile(weak, 12)
    deep = weak < thr
    ax.scatter(x[deep], weak[deep], s=14, color="#c0392b", zorder=5,
               label="Desvanecimiento profundo")
    ax.set_title(f"Eigenvalores del canal MIMO {N}x{N} por subportadora", fontsize=10)
    ax.set_xlabel("Subportadora"); ax.set_ylabel("Magnitud del eigenvalor (dB)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)


def plot_sim_time(ax, res):
    """Tiempo total de simulación por nivel de SNR (carga computacional)."""
    snr = np.array(res["snr"])
    t = np.array(res["sim_time_ms"])
    ax.bar(snr, t, width=3.0, color="#16a085", edgecolor="black", linewidth=0.6)
    ax.set_title("Tiempo de simulación por nivel de SNR", fontsize=10)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Tiempo (ms)")
    ax.grid(True, axis="y", alpha=0.3)


def plot_image_time(ax, td):
    """Tiempo de transmisión de imagen: SISO frente a MIMO N x N."""
    N = td["N"]
    labels = ["SISO (1 capa)", f"MIMO {N}x{N} ({N} capas)"]
    vals = [td["t_siso_ms"], td["t_mimo_ms"]]
    colors = ["#2980b9", "#c0392b"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Tiempo de transmisión de imagen — MIMO ×{td['speedup']:.1f} más rápido",
                 fontsize=10)
    ax.set_ylabel("Tiempo (ms)")
    ax.grid(True, axis="y", alpha=0.3)


def plot_papr_ccdf(ax, pd):
    """CCDF del PAPR: la precodificación DFT del SC-FDM reduce los picos."""
    g = pd["gamma"]
    ax.semilogy(g, np.maximum(pd["ccdf_ofdm"], 1e-4), color="#c0392b", linewidth=2,
                label="OFDM")
    ax.semilogy(g, np.maximum(pd["ccdf_sc"], 1e-4), color="#2980b9", linewidth=2,
                label="SC-FDM (precod. DFT)")
    ax.set_ylim(1e-3, 1.0)
    ax.set_title("Distribución de potencia (CCDF del PAPR)", fontsize=10)
    ax.set_xlabel("PAPR γ (dB)"); ax.set_ylabel("Probabilidad  P(PAPR > γ)")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)


def plot_llr_hist(ax, ld):
    """Histograma de LLR: el decodificador Turbo 'limpia' las decisiones."""
    ch = ld["llr_channel"]; tb = ld["llr_turbo"]
    lim = np.percentile(np.abs(np.concatenate([ch, tb])), 99)
    bins = np.linspace(-lim, lim, 60)
    ax.hist(ch, bins=bins, density=True, alpha=0.55, color="#7f8c8d",
            label="Salida del demodulador (canal)")
    ax.hist(tb, bins=bins, density=True, alpha=0.55, color="#c0392b",
            label="Después de Turbo (LLR total)")
    ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_title(f"Métrica de confianza LLR — SNR {ld['snr_db']} dB", fontsize=10)
    ax.set_xlabel("LLR"); ax.set_ylabel("Densidad de probabilidad")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
