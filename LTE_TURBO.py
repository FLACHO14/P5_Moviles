# LTE_TURBO.py
# Codificador y decodificador de canal Turbo del estándar LTE (3GPP TS 36.212).
#
# Estructura PCCC (Parallel Concatenated Convolutional Code) formada por dos
# codificadores RSC (Recursive Systematic Convolutional) idénticos de 8 estados
# unidos por un entrelazador QPP (Quadratic Permutation Polynomial). La tasa
# base es 1/3: por cada bit de información se transmiten el bit sistemático y
# dos bits de paridad, uno de cada codificador constituyente.
#
# Polinomios generadores del estándar (octal 13 y 15):
#   g0(D) = 1 + D^2 + D^3   (realimentación)
#   g1(D) = 1 + D + D^3     (paridad)
#
# El decodificador es iterativo y emplea el algoritmo Max-Log-MAP (BCJR
# simplificado) intercambiando información extrínseca entre los dos
# decodificadores SISO a través del entrelazador.

import numpy as np


NSTATES = 8  # trellis de 8 estados (3 elementos de memoria)


# -------------------------------------------------------------------
# Trellis del codificador RSC (8 estados)
# -------------------------------------------------------------------

def _build_trellis():
    """Construye las tablas de transición del RSC de LTE.

    Estado codificado como entero a partir de los registros (m1, m2, m3):
    state = m1 + 2*m2 + 4*m3, con m1 el registro más reciente.

    Returns
    -------
    next_state[s, u] : estado siguiente desde s con entrada u.
    out_parity[s, u] : bit de paridad generado en esa transición.
    """
    next_state = np.zeros((NSTATES, 2), dtype=int)
    out_parity = np.zeros((NSTATES, 2), dtype=int)
    for s in range(NSTATES):
        m1, m2, m3 = s & 1, (s >> 1) & 1, (s >> 2) & 1
        for u in (0, 1):
            fb = u ^ m2 ^ m3          # realimentación g0 = 1 + D^2 + D^3
            p = fb ^ m1 ^ m3          # paridad g1 = 1 + D + D^3
            ns = fb | (m1 << 1) | (m2 << 2)
            next_state[s, u] = ns
            out_parity[s, u] = p
    return next_state, out_parity


NEXT_STATE, OUT_PARITY = _build_trellis()


# -------------------------------------------------------------------
# Entrelazador QPP del estándar LTE
# -------------------------------------------------------------------

# Subconjunto de la tabla 5.1.3-3 de 3GPP TS 36.212 (parámetros f1, f2).
QPP_TABLE = {
    40: (3, 10), 64: (7, 16), 128: (15, 32), 256: (15, 32),
    512: (31, 64), 1024: (31, 64), 2048: (31, 64),
}


def qpp_interleaver(K):
    """Entrelazador QPP de LTE para un tamaño de bloque K válido.

    La permutación se define como Pi(i) = (f1*i + f2*i^2) mod K, con los
    parámetros f1 y f2 especificados por el estándar para cada tamaño K.
    """
    if K not in QPP_TABLE:
        raise ValueError(f"Tamaño de bloque K={K} no está en la tabla QPP soportada: "
                         f"{sorted(QPP_TABLE)}")
    f1, f2 = QPP_TABLE[K]
    i = np.arange(K)
    return (f1 * i + f2 * i * i) % K


def inverse_permutation(perm):
    """Permutación inversa para deshacer el entrelazado."""
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# -------------------------------------------------------------------
# Codificador RSC con terminación de trellis
# -------------------------------------------------------------------

def rsc_encode(bits):
    """Codifica una secuencia con el RSC de LTE y termina el trellis.

    Devuelve los bits sistemáticos y de paridad de la parte de información y
    de los tres bits de cola que llevan el codificador al estado cero.
    """
    s = 0
    par = np.zeros(len(bits), dtype=np.uint8)
    for k, u in enumerate(bits):
        par[k] = OUT_PARITY[s, u]
        s = NEXT_STATE[s, u]

    tail_sys = np.zeros(3, dtype=np.uint8)
    tail_par = np.zeros(3, dtype=np.uint8)
    for k in range(3):
        m1, m2, m3 = s & 1, (s >> 1) & 1, (s >> 2) & 1
        u_t = m2 ^ m3            # entrada que fuerza realimentación nula
        tail_sys[k] = u_t
        tail_par[k] = OUT_PARITY[s, u_t]
        s = NEXT_STATE[s, u_t]
    return par, tail_sys, tail_par


def turbo_encode(bits, interleaver=None):
    """Codificador Turbo PCCC de LTE (tasa 1/3 más cola).

    Returns
    -------
    dict con los flujos sistemático y de paridad de información y de cola, y el
    entrelazador empleado.
    """
    bits = np.asarray(bits).astype(np.uint8).ravel()
    K = len(bits)
    if interleaver is None:
        interleaver = qpp_interleaver(K)

    # Codificador 1 sobre los bits originales
    par1, ts1, tp1 = rsc_encode(bits)
    # Codificador 2 sobre los bits entrelazados
    bits_i = bits[interleaver]
    par2, ts2, tp2 = rsc_encode(bits_i)

    return {
        "K": K,
        "sys": bits,            # sistemático de información
        "par1": par1,           # paridad codificador 1
        "par2": par2,           # paridad codificador 2
        "tail1_sys": ts1, "tail1_par": tp1,
        "tail2_sys": ts2, "tail2_par": tp2,
        "interleaver": interleaver,
    }


# -------------------------------------------------------------------
# Decodificador SISO Max-Log-MAP (BCJR)
# -------------------------------------------------------------------

_XPAR = 1.0 - 2.0 * OUT_PARITY  # símbolo de paridad por (estado, entrada)


def siso_maxlogmap(Lc_sys, Lc_par, La):
    """Decodificador SISO Max-Log-MAP para el RSC terminado.

    Parameters
    ----------
    Lc_sys, Lc_par : LLR de canal de los bits sistemático y de paridad.
    La : LLR a priori de los bits de información (cero en la cola).

    Returns
    -------
    Le : información extrínseca de cada bit.
    """
    N = len(Lc_sys)
    INF = 1e9
    ns0 = NEXT_STATE[:, 0]; ns1 = NEXT_STATE[:, 1]
    xp0 = _XPAR[:, 0]; xp1 = _XPAR[:, 1]

    # Métrica de rama por sección: gamma_u[s'] = 0.5*xu*(La+Lcs) + 0.5*xp*Lcp
    Ls = La + Lc_sys
    g0 = 0.5 * Ls[:, None] + 0.5 * (xp0 * Lc_par[:, None])  # (N, 8) para u=0
    g1 = -0.5 * Ls[:, None] + 0.5 * (xp1 * Lc_par[:, None])  # (N, 8) para u=1

    # Recursión hacia adelante (alpha)
    alpha = np.full((N + 1, NSTATES), -INF)
    alpha[0, 0] = 0.0
    for k in range(N):
        a = alpha[k]
        newa = np.full(NSTATES, -INF)
        np.maximum.at(newa, ns0, a + g0[k])
        np.maximum.at(newa, ns1, a + g1[k])
        m = newa.max()
        alpha[k + 1] = newa - (m if m > -INF / 2 else 0.0)  # normalización

    # Recursión hacia atrás (beta) — trellis terminado en estado 0
    beta = np.full((N + 1, NSTATES), -INF)
    beta[N, 0] = 0.0
    for k in range(N - 1, -1, -1):
        b = beta[k + 1]
        beta[k] = np.maximum(g0[k] + b[ns0], g1[k] + b[ns1])
        m = beta[k].max()
        if m > -INF / 2:
            beta[k] -= m

    # LLR y extrínseca
    Le = np.zeros(N)
    for k in range(N):
        a = alpha[k]; b = beta[k + 1]
        m0 = np.max(a + g0[k] + b[ns0])  # transiciones con u=0
        m1 = np.max(a + g1[k] + b[ns1])  # transiciones con u=1
        L = m0 - m1
        Le[k] = L - La[k] - Lc_sys[k]
    return Le


def turbo_decode(Lc_sys, Lc_par1, Lc_par2,
                 Lc_tail1_sys, Lc_tail1_par, Lc_tail2_sys, Lc_tail2_par,
                 interleaver, n_iter=6):
    """Decodificador Turbo iterativo de LTE (Max-Log-MAP).

    Intercambia información extrínseca entre los dos decodificadores SISO a
    través del entrelazador durante n_iter iteraciones y devuelve los bits
    estimados.
    """
    K = len(Lc_sys)
    deint = inverse_permutation(interleaver)

    # Secuencias de información más cola para cada decodificador
    sys1 = np.concatenate([Lc_sys, Lc_tail1_sys])
    par1 = np.concatenate([Lc_par1, Lc_tail1_par])
    sys2 = np.concatenate([Lc_sys[interleaver], Lc_tail2_sys])
    par2 = np.concatenate([Lc_par2, Lc_tail2_par])

    La1 = np.zeros(K + 3)
    Le2_info = np.zeros(K)

    for _ in range(n_iter):
        La1[:K] = Le2_info[deint]
        La1[K:] = 0.0
        Le1 = siso_maxlogmap(sys1, par1, La1)
        Le1_info = Le1[:K]

        La2 = np.zeros(K + 3)
        La2[:K] = Le1_info[interleaver]
        Le2 = siso_maxlogmap(sys2, par2, La2)
        Le2_info = Le2[:K]

    # LLR total y decisión dura
    L_total = Lc_sys + Le1_info + Le2_info[deint]
    return (L_total < 0).astype(np.uint8)


# -------------------------------------------------------------------
# Cadena completa sobre canal AWGN (BPSK)
# -------------------------------------------------------------------

def _bpsk(bits):
    """Mapea bits a símbolos BPSK: 0 -> +1, 1 -> -1."""
    return 1.0 - 2.0 * np.asarray(bits, dtype=float)


def code_rate(K):
    """Tasa real del código incluyendo los bits de cola."""
    n_coded = 3 * K + 12  # K sis + K par1 + K par2 + 12 bits de cola
    return K / n_coded


def run_awgn_chain(bits, ebno_db, n_iter=6, interleaver=None):
    """Codifica, transmite por AWGN y decodifica un bloque Turbo.

    Devuelve los bits decodificados para comparar con los originales.
    """
    enc = turbo_encode(bits, interleaver)
    K = enc["K"]
    R = code_rate(K)
    ebno = 10 ** (ebno_db / 10)
    sigma2 = 1.0 / (2.0 * R * ebno)   # energía de símbolo unitaria
    Lc = 2.0 / sigma2
    sigma = np.sqrt(sigma2)

    def tx(b):
        x = _bpsk(b)
        return x + sigma * np.random.randn(len(x))

    rs = tx(enc["sys"]); rp1 = tx(enc["par1"]); rp2 = tx(enc["par2"])
    rt1s = tx(enc["tail1_sys"]); rt1p = tx(enc["tail1_par"])
    rt2s = tx(enc["tail2_sys"]); rt2p = tx(enc["tail2_par"])

    dec = turbo_decode(
        Lc * rs, Lc * rp1, Lc * rp2,
        Lc * rt1s, Lc * rt1p, Lc * rt2s, Lc * rt2p,
        enc["interleaver"], n_iter,
    )
    return dec


def run_ber_turbo_vs_uncoded(K, ebno_list, n_frames, n_iter=6):
    """BER frente a Eb/N0 del código Turbo y de BPSK sin codificar.

    Permite cuantificar la ganancia de codificación que aporta el código Turbo
    de LTE respecto a la transmisión sin protección.
    """
    inter = qpp_interleaver(K)
    ber_turbo = []
    ber_unc = []
    for ebno_db in ebno_list:
        err_t = bits_t = 0
        err_u = bits_u = 0
        ebno = 10 ** (ebno_db / 10)
        sigma_u = np.sqrt(1.0 / (2.0 * ebno))
        for _ in range(n_frames):
            bits = np.random.randint(0, 2, K).astype(np.uint8)

            dec = run_awgn_chain(bits, ebno_db, n_iter, inter)
            err_t += np.sum(dec != bits); bits_t += K

            # Sin codificar: BPSK directo de los mismos bits
            r = _bpsk(bits) + sigma_u * np.random.randn(K)
            dec_u = (r < 0).astype(np.uint8)
            err_u += np.sum(dec_u != bits); bits_u += K

        ber_turbo.append(max(err_t / bits_t, 0.0))
        ber_unc.append(max(err_u / bits_u, 0.0))
    return {"ebno": list(ebno_list), "turbo": ber_turbo, "uncoded": ber_unc,
            "K": K, "n_iter": n_iter, "rate": code_rate(K)}


def run_ber_vs_iterations(K, ebno_db, iters_list, n_frames):
    """BER del código Turbo en función del número de iteraciones del decodificador.

    Muestra la convergencia del decodificador iterativo: cada iteración
    adicional refina la información extrínseca y reduce la tasa de error hasta
    saturar.
    """
    inter = qpp_interleaver(K)
    bers = []
    for nit in iters_list:
        err = tot = 0
        for _ in range(n_frames):
            bits = np.random.randint(0, 2, K).astype(np.uint8)
            dec = run_awgn_chain(bits, ebno_db, nit, inter)
            err += np.sum(dec != bits); tot += K
        bers.append(max(err / tot, 0.0))
    return {"iters": list(iters_list), "ber": bers, "ebno_db": ebno_db, "K": K}


# -------------------------------------------------------------------
# Visualización
# -------------------------------------------------------------------

def plot_ber_turbo(ax, res):
    """BER vs Eb/N0 del Turbo frente a no codificado."""
    eb = np.array(res["ebno"])
    for key, color, label, marker in (
        ("uncoded", "#7f8c8d", "Sin codificar (BPSK)", "o"),
        ("turbo", "#c0392b", f"Turbo LTE (K={res['K']}, {res['n_iter']} iter)", "s"),
    ):
        b = np.array(res[key])
        mask = b > 0
        if np.any(mask):
            ax.semilogy(eb[mask], b[mask], marker=marker, color=color, label=label)
        if np.any(~mask):
            ax.semilogy(eb[~mask], np.full(np.sum(~mask), 1e-7), marker=marker,
                        color=color, linestyle="", alpha=0.5)
    ax.set_title("BER vs Eb/N0 — Código Turbo LTE frente a no codificado")
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("BER")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)


def plot_ber_iterations(ax, res):
    """BER en función del número de iteraciones del decodificador."""
    ax.semilogy(res["iters"], np.maximum(res["ber"], 1e-7), "o-", color="#8e44ad")
    ax.set_title(f"Convergencia (Eb/N0 = {res['ebno_db']} dB)", fontsize=10)
    ax.set_xlabel("Iteraciones"); ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3)


# -------------------------------------------------------------------
# Turbo sobre modulación QAM: comparación por modulación
# -------------------------------------------------------------------

def _qam_table(M):
    """Tabla de símbolos QAM y etiquetas de bits compatible con ofdm_tx.qam_mod."""
    from ofdm_tx import qam_constellation
    k = int(np.log2(M))
    kb = k // 2
    levels = qam_constellation(M)
    syms = np.zeros(M, dtype=complex)
    bitmat = np.zeros((M, k), dtype=int)
    for lab in range(M):
        bits = [(lab >> (k - 1 - i)) & 1 for i in range(k)]
        bi = 0
        for i in range(kb):
            bi = (bi << 1) | bits[i]
        bq = 0
        for i in range(kb):
            bq = (bq << 1) | bits[kb + i]
        gi = bi ^ (bi >> 1)
        gq = bq ^ (bq >> 1)
        syms[lab] = levels[gi] + 1j * levels[gq]
        bitmat[lab] = bits
    return syms, bitmat, k


def qam_soft_demap(rx, M, N0):
    """Demapeo suave de QAM: LLR por bit mediante aproximación Max-Log.

    Para cada símbolo recibido y cada bit, el LLR se obtiene de la diferencia
    entre la distancia mínima a un símbolo con ese bit en cero y en uno. Estos
    LLR alimentan directamente al decodificador Turbo.
    """
    syms, bitmat, k = _qam_table(M)
    rx = np.asarray(rx).ravel()
    d2 = np.abs(rx[:, None] - syms[None, :]) ** 2  # (N, M)
    llr = np.zeros((len(rx), k))
    for j in range(k):
        s0 = bitmat[:, j] == 0
        s1 = ~s0
        min0 = d2[:, s0].min(axis=1)
        min1 = d2[:, s1].min(axis=1)
        llr[:, j] = (min1 - min0) / N0   # L>0 favorece bit 0
    return llr.reshape(-1)


def _qam_mod(bits, M):
    from ofdm_tx import qam_mod
    return qam_mod(np.asarray(bits, dtype=np.uint8), M)


# -------------------------------------------------------------------
# Adaptación de tasa por puncturing (rate matching)
# -------------------------------------------------------------------

RATE_LABELS = ["1/3", "1/2", "2/3", "3/4"]


def rate_masks(K, rate):
    """Máscaras de puncturing de paridad para alcanzar la tasa deseada.

    La tasa base del Turbo de LTE es 1/3 (se transmite toda la paridad). Para
    tasas mayores se perfora parte de los bits de paridad, lo que reduce la
    redundancia y, por tanto, la capacidad de corrección, a cambio de una mayor
    eficiencia espectral. True indica que el bit de paridad se transmite.
    """
    m1 = np.zeros(K, dtype=bool)
    m2 = np.zeros(K, dtype=bool)
    if rate == "1/2":
        m1[0::2] = True; m2[1::2] = True
    elif rate == "2/3":
        m1[0::4] = True; m2[2::4] = True
    elif rate == "3/4":
        m1[0::6] = True; m2[3::6] = True
    else:  # "1/3" base, sin puncturing
        m1[:] = True; m2[:] = True
    return m1, m2


def code_rate_punctured(K, m1, m2):
    """Tasa real del código tras el puncturing, incluyendo bits de cola."""
    n_coded = K + int(m1.sum()) + int(m2.sum()) + 12
    return K / n_coded


def _turbo_chain_qam(bits, M, N0, sigma, m1, m2, inter, n_iter):
    """Codifica, perfora, modula QAM, demapea suave y decodifica un bloque.

    Los bits de paridad perforados se tratan como borrados en el decodificador
    asignándoles una información de canal nula, que el algoritmo BCJR maneja de
    forma natural.
    """
    K = len(bits)
    kbits = int(np.log2(M))
    enc = turbo_encode(bits, inter)
    coded = np.concatenate([
        enc["sys"], enc["par1"][m1], enc["par2"][m2],
        enc["tail1_sys"], enc["tail1_par"], enc["tail2_sys"], enc["tail2_par"],
    ]).astype(np.uint8)
    pc = (-len(coded)) % kbits
    coded_p = np.pad(coded, (0, pc)) if pc else coded
    symc = _qam_mod(coded_p, M)
    rc = symc + sigma * (np.random.randn(len(symc)) + 1j * np.random.randn(len(symc)))
    llr = qam_soft_demap(rc, M, N0)[:len(coded)]

    # Reconstruye los flujos completos con borrados en las posiciones perforadas
    Ls = llr[:K]
    n1 = int(m1.sum()); n2 = int(m2.sum())
    off = K
    Lp1 = np.zeros(K); Lp1[m1] = llr[off:off + n1]; off += n1
    Lp2 = np.zeros(K); Lp2[m2] = llr[off:off + n2]; off += n2
    Lt1s = llr[off:off + 3]; Lt1p = llr[off + 3:off + 6]
    Lt2s = llr[off + 6:off + 9]; Lt2p = llr[off + 9:off + 12]
    return turbo_decode(Ls, Lp1, Lp2, Lt1s, Lt1p, Lt2s, Lt2p, inter, n_iter)


def run_ber_qam_raw_vs_turbo(K, snr_list, n_frames, n_iter=6, rate="1/3"):
    """BER vs SNR para QPSK, 16QAM y 64QAM, sin codificar y con Turbo.

    Reproduce la comparación clásica en la que, para cada modulación, la curva
    con codificación Turbo queda por debajo de la curva sin codificar, lo que
    evidencia la ganancia de codificación del estándar LTE. El parámetro rate
    controla la adaptación de tasa por puncturing.
    """
    from ofdm_rx import qam_demod
    mods = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
    inter = qpp_interleaver(K)
    m1, m2 = rate_masks(K, rate)
    res = {"snr": list(snr_list), "raw": {}, "turbo": {},
           "rate": code_rate_punctured(K, m1, m2), "rate_label": rate}

    for name, M in mods.items():
        ber_raw = []
        ber_turbo = []
        for snr_db in snr_list:
            N0 = 10 ** (-snr_db / 10.0)   # Es = 1, SNR = Es/N0
            sigma = np.sqrt(N0 / 2.0)
            er = br = et = bt = 0
            for _ in range(n_frames):
                bits = np.random.randint(0, 2, K).astype(np.uint8)

                # Sin codificar: QAM directo (relleno a múltiplo de log2(M))
                kbits = int(np.log2(M))
                pad_r = (-len(bits)) % kbits
                bits_r = np.pad(bits, (0, pad_r)) if pad_r else bits
                sym = _qam_mod(bits_r, M)
                r = sym + sigma * (np.random.randn(len(sym)) + 1j * np.random.randn(len(sym)))
                bhat = qam_demod(r, M)[: len(bits)]
                if len(bhat) < len(bits):
                    bhat = np.pad(bhat, (0, len(bits) - len(bhat)))
                er += np.sum(bhat != bits); br += K

                # Con Turbo (tasa configurable por puncturing)
                dec = _turbo_chain_qam(bits, M, N0, sigma, m1, m2, inter, n_iter)
                et += np.sum(dec != bits); bt += K

            ber_raw.append(er / br)
            ber_turbo.append(et / bt)
        res["raw"][name] = ber_raw
        res["turbo"][name] = ber_turbo
    return res


def transmit_image_turbo(img_arr, M, snr_db, n_iter=6, K=256, rate="1/3"):
    """Transmite una imagen sin codificar y con código Turbo sobre AWGN.

    Segmenta los bits de la imagen en bloques de tamaño K, codifica cada bloque
    con el código Turbo, los modula en QAM, los transmite por AWGN y los
    decodifica. A una relación señal a ruido en la que la transmisión sin
    codificar presenta muchos errores, el código Turbo recupera la imagen con
    una calidad muy superior.
    """
    import ofdm_utils
    from ofdm_rx import qam_demod

    # Redimensiona a un tamaño manejable para que la demostración sea rápida,
    # ya que el decodificador Turbo procesa la imagen bloque a bloque.
    img_arr = np.asarray(img_arr, dtype=np.uint8)
    max_side = 64
    if max(img_arr.shape) > max_side:
        try:
            from PIL import Image
            h, w = img_arr.shape
            scale = max_side / max(h, w)
            im = Image.fromarray(img_arr).resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            img_arr = np.array(im, dtype=np.uint8)
        except Exception:
            step = int(np.ceil(max(img_arr.shape) / max_side))
            img_arr = img_arr[::step, ::step]

    bits = np.unpackbits(np.asarray(img_arr, dtype=np.uint8).ravel())
    n_bits = len(bits)
    kbits = int(np.log2(M))
    N0 = 10 ** (-snr_db / 10.0)
    sigma = np.sqrt(N0 / 2.0)
    inter = qpp_interleaver(K)

    # --- Sin codificar ---
    pad = (-n_bits) % kbits
    b_pad = np.pad(bits, (0, pad)) if pad else bits
    sym = _qam_mod(b_pad, M)
    r = sym + sigma * (np.random.randn(len(sym)) + 1j * np.random.randn(len(sym)))
    bits_raw = qam_demod(r, M)[:n_bits]

    # --- Con Turbo (bloque a bloque, tasa configurable por puncturing) ---
    m1, m2 = rate_masks(K, rate)
    n_blocks = int(np.ceil(n_bits / K))
    bits_turbo = np.zeros(n_blocks * K, dtype=np.uint8)
    src = np.pad(bits, (0, n_blocks * K - n_bits)) if n_blocks * K > n_bits else bits
    for bidx in range(n_blocks):
        blk = src[bidx * K:(bidx + 1) * K]
        bits_turbo[bidx * K:(bidx + 1) * K] = _turbo_chain_qam(
            blk, M, N0, sigma, m1, m2, inter, n_iter)
    bits_turbo = bits_turbo[:n_bits]

    cap = ((n_bits + 7) // 8) * 8
    img_raw = ofdm_utils._bits_to_image(bits_raw, n_bits, cap, img_arr.shape)
    img_tur = ofdm_utils._bits_to_image(bits_turbo, n_bits, cap, img_arr.shape)
    mse_r, psnr_r = ofdm_utils._img_metrics(img_arr, img_raw)
    mse_t, psnr_t = ofdm_utils._img_metrics(img_arr, img_tur)
    ber_r = float(np.mean(bits_raw[:n_bits] != bits[:n_bits]))
    ber_t = float(np.mean(bits_turbo[:n_bits] != bits[:n_bits]))
    return {
        "img_orig": img_arr, "img_raw": img_raw, "img_turbo": img_tur,
        "psnr_raw": psnr_r, "psnr_turbo": psnr_t,
        "ber_raw": ber_r, "ber_turbo": ber_t,
        "snr_db": snr_db, "M": M,
    }


_QAM_COLORS = {"QPSK": "#2741d6", "16QAM": "#1e8a2e", "64QAM": "#d62728"}


def plot_ber_qam_turbo(ax, res):
    """BER vs SNR de QPSK, 16QAM y 64QAM, sin codificar (Raw) y con Turbo."""
    snr = np.array(res["snr"])
    for name in ("QPSK", "16QAM", "64QAM"):
        c = _QAM_COLORS[name]
        raw = np.maximum(np.array(res["raw"][name]), 1e-7)
        tur = np.maximum(np.array(res["turbo"][name]), 1e-7)
        ax.semilogy(snr, raw, "o--", color=c, alpha=0.7, markersize=5,
                    label=f"{name} Raw")
        ax.semilogy(snr, tur, "s-", color=c, linewidth=2, markersize=5,
                    label=f"{name} Turbo")
    ax.set_title("BER vs SNR — QAM sin codificar frente a Turbo LTE")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_ylim(1e-5, 1.0)
    # Leyenda discreta en la esquina inferior izquierda (zona libre de curvas)
    ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.85,
              handlelength=1.6, columnspacing=1.0, borderpad=0.4)
    ax.grid(True, which="both", alpha=0.3)
