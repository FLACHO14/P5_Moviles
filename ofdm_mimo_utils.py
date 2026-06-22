# ofdm_mimo_utils.py
# Funciones de análisis y visualización para sistemas MIMO
# Incluye: comparativa SISO vs SIMO vs MISO, constelaciones, PAPR

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def run_mimo_ber_analysis(mod_type, M_qam, channel_profile, velocity_kmh,
                          snr_list, n_mc_iterations, use_scfdma=False, M_dft=None):
    """Análisis Monte Carlo comparativo: SISO vs SIMO-MRC vs MISO-SFBC.

    Retorna BER y IC 95% para cada técnica a cada SNR.
    """
    from ofdm_params import PILOT_AMPLITUDE
    from ofdm_utils import get_nfft_cp, build_subcarrier_map, generate_constellation_data
    from ofdm_tx import qam_mod, ofdm_tx_block, scfdma_tx_block, sfbc_tx_2ant
    from ofdm_rx import qam_demod, ofdm_rx_block, equalize_with_pilots
    from ofdm_channel import get_channel_profile, apply_channel, apply_channel_mimo, apply_channel_miso
    from ofdm_channel import generate_mimo_channels, generate_miso_channels

    Nfft, cp_len = get_nfft_cp(10e6)
    sc_map = build_subcarrier_map(Nfft)
    n_data = sc_map["n_data"]

    if use_scfdma and M_dft is None:
        from ofdm_tx import get_best_dft_size
        M_dft = get_best_dft_size(n_data, Nfft)

    n_bits = 10000
    bits = np.random.randint(0, 2, n_bits)

    ber_siso = {snr: [] for snr in snr_list}
    ber_simo_mrc = {snr: [] for snr in snr_list}
    ber_miso_sfbc = {snr: [] for snr in snr_list}

    for mc in range(n_mc_iterations):
        symbols_data = qam_mod(bits, M_qam)

        if use_scfdma:
            tx_siso, _, _ = scfdma_tx_block(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE, M_dft)
        else:
            tx_siso, _, _ = ofdm_tx_block(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

        tx1_sfbc, tx2_sfbc, _, _, _, _ = sfbc_tx_2ant(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

        for snr_db in snr_list:
            h_channel = get_channel_profile(channel_profile, taps_L=8)

            rx_siso, _ = apply_channel(tx_siso, h_channel, snr_db, velocity_kmh)

            h_mimo = generate_mimo_channels(2, channel_profile, taps_L=8)
            rx_simo_list = apply_channel_mimo(tx_siso, h_mimo, snr_db, velocity_kmh)

            h_miso = generate_miso_channels(2, channel_profile, taps_L=8)
            rx_miso, _ = apply_channel_miso([tx1_sfbc, tx2_sfbc], h_miso, snr_db, velocity_kmh)

            Y_siso = ofdm_rx_block(rx_siso, Nfft, cp_len)
            data_eq_siso = equalize_with_pilots(Y_siso, sc_map, PILOT_AMPLITUDE, Nfft)[0]

            Y_simo_list = [ofdm_rx_block(rx, Nfft, cp_len) for rx in rx_simo_list]
            H_simo_list = []
            for Y in Y_simo_list:
                _, H_est, _ = equalize_with_pilots(Y, sc_map, PILOT_AMPLITUDE, Nfft)
                H_simo_list.append(np.array(H_est))

            from ofdm_rx import combine_mrc_frequency
            data_combined = combine_mrc_frequency(Y_simo_list, H_simo_list, sc_map)
            data_eq_simo = data_combined

            Y_miso = ofdm_rx_block(rx_miso, Nfft, cp_len)
            data_eq_miso = equalize_with_pilots(Y_miso, sc_map, PILOT_AMPLITUDE, Nfft)[0]

            bits_siso = (np.angle(data_eq_siso) > 0).astype(int)
            bits_simo = (np.angle(data_eq_simo) > 0).astype(int)
            bits_miso = (np.angle(data_eq_miso) > 0).astype(int)

            ber_siso[snr_db].append(np.mean(bits_siso[:len(bits)] != bits[:len(bits_siso)]))
            ber_simo_mrc[snr_db].append(np.mean(bits_simo[:len(bits)] != bits[:len(bits_simo)]))
            ber_miso_sfbc[snr_db].append(np.mean(bits_miso[:len(bits)] != bits[:len(bits_miso)]))

    results = {
        'snr': snr_list,
        'siso': ber_siso,
        'simo_mrc': ber_simo_mrc,
        'miso_sfbc': ber_miso_sfbc
    }

    return results


def compute_ber_with_ci(ber_list, confidence=0.95):
    """Calcula BER promedio e intervalo de confianza.

    Retorna: (ber_mean, ci_lower, ci_upper)
    """
    if not ber_list or len(ber_list) == 0:
        return 0, 0, 0

    ber_array = np.array(ber_list)
    mean = np.mean(ber_array)
    se = stats.sem(ber_array)
    ci = se * stats.t.ppf((1 + confidence) / 2, len(ber_array) - 1)

    return mean, mean - ci, mean + ci


def plot_ber_comparison_mimo(results, modulation, channel_profile):
    """Plotea BER vs SNR comparando SISO, SIMO-MRC y MISO-SFBC.

    Incluye intervalo de confianza del 95% para cada punto.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    mods = ['QPSK', '16QAM', '64QAM']

    snr_vals = results['snr']
    techniques = ['siso', 'simo_mrc', 'miso_sfbc']
    colors = ['gray', 'green', 'orange']
    labels = ['SISO', 'SIMO-MRC (2 RX)', 'MISO-SFBC (2 TX)']

    for ax_idx, mod in enumerate(mods):
        ax = axes[ax_idx]

        for tech, color, label in zip(techniques, colors, labels):
            ber_means = []
            ber_lowers = []
            ber_uppers = []

            for snr in snr_vals:
                if snr in results[tech]:
                    mean, lower, upper = compute_ber_with_ci(results[tech][snr])
                    ber_means.append(mean)
                    ber_lowers.append(lower)
                    ber_uppers.append(upper)

            ber_means = np.array(ber_means)
            ber_lowers = np.array(ber_lowers)
            ber_uppers = np.array(ber_uppers)

            ax.semilogy(snr_vals, ber_means, 'o-', color=color, label=label, linewidth=2, markersize=6)
            ax.fill_between(snr_vals, ber_lowers, ber_uppers, alpha=0.2, color=color)

        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel('BER', fontsize=10)
        ax.set_title(f'{mod} — BER vs SNR + IC 95%', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim([1e-5, 1])

    plt.suptitle(f'Comparativa MIMO: {channel_profile} — Modulación {modulation}', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_constellation_comparison_mrc(data_before, data_after, M_qam):
    """Plotea constelación antes y después de combinación MRC.

    Muestra el efecto de suavizado del ruido por diversidad.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in zip(axes, [data_before, data_after],
                               ['Antes de MRC (SISO)', 'Después de MRC (2 RX)']):
        ax.scatter(np.real(data), np.imag(data), alpha=0.4, s=10, c='blue')
        ax.set_xlabel('I', fontsize=10)
        ax.set_ylabel('Q', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.suptitle(f'{M_qam}-QAM: Reducción de ruido por MRC', fontsize=12)
    plt.tight_layout()
    return fig


def plot_papr_scfdma_vs_ofdm_mimo(papr_ofdm, papr_scfdma, papr_sfbc):
    """Compara PAPR entre OFDM puro, SC-FDMA y SFBC.

    Demuestra que SC-FDMA mantiene PAPR bajo incluso en MIMO.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [papr_ofdm, papr_scfdma, papr_sfbc]
    labels = ['OFDM (2 TX)', 'SC-FDMA (2 TX)', 'SFBC (2 TX)']
    colors = ['red', 'blue', 'green']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('PAPR (dB)', fontsize=11)
    ax.set_title('Comparativa PAPR: OFDM vs SC-FDMA vs SFBC', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, color='red', marker='D', s=100,
               label='Promedio', zorder=3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_channel_power_stability_mimo(H_single, H_combined):
    """Compara respuesta en potencia del canal: una antena vs combinación MRC.

    Muestra cómo MRC suaviza los desvanecimientos profundos.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    subcarriers = np.arange(len(H_single))

    ax.plot(subcarriers, 10 * np.log10(np.abs(H_single) ** 2 + 1e-12),
            'r-', alpha=0.7, linewidth=2, label='1 antena (SISO)')
    ax.plot(subcarriers, 10 * np.log10(np.abs(H_combined) ** 2 + 1e-12),
            'g-', alpha=0.7, linewidth=2, label='MRC combinado (2 RX)')

    gain = np.mean(np.abs(H_combined) ** 2) / np.mean(np.abs(H_single) ** 2)
    ax.axhline(y=10 * np.log10(gain), color='k', linestyle='--', alpha=0.5,
               label=f'Ganancia MRC: {10*np.log10(gain):.1f} dB')

    ax.set_xlabel('Subportadora', fontsize=11)
    ax.set_ylabel('Potencia del Canal (dB)', fontsize=11)
    ax.set_title('Suavizado de Canal por Diversidad MRC', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def plot_diversity_gain_curves(snr_list, ber_siso, ber_mrc, M_qam):
    """Plotea ganancia de diversidad: diferencia en dB entre SISO y MRC.

    Muestra el beneficio en términos de SNR requerido.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ber_means_siso = [np.mean(ber_siso.get(snr, [1])) for snr in snr_list]
    ber_means_mrc = [np.mean(ber_mrc.get(snr, [1])) for snr in snr_list]

    ax1.semilogy(snr_list, ber_means_siso, 'o-', color='red', label='SISO', linewidth=2)
    ax1.semilogy(snr_list, ber_means_mrc, 's-', color='green', label='MRC (2 RX)', linewidth=2)
    ax1.set_xlabel('SNR (dB)', fontsize=11)
    ax1.set_ylabel('BER', fontsize=11)
    ax1.set_title('BER vs SNR', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([1e-6, 1])

    diversity_gain = []
    for target_ber in [1e-2, 1e-3, 1e-4]:
        snr_siso_at_target = None
        snr_mrc_at_target = None

        for i, snr in enumerate(snr_list):
            if ber_means_siso[i] <= target_ber and snr_siso_at_target is None:
                snr_siso_at_target = snr
            if ber_means_mrc[i] <= target_ber and snr_mrc_at_target is None:
                snr_mrc_at_target = snr

        if snr_siso_at_target and snr_mrc_at_target:
            gain = snr_siso_at_target - snr_mrc_at_target
            diversity_gain.append((f'BER={target_ber}', gain))

    if diversity_gain:
        labels_div, gains = zip(*diversity_gain)
        ax2.bar(labels_div, gains, color='steelblue', alpha=0.7)
        ax2.set_ylabel('Ganancia en SNR (dB)', fontsize=11)
        ax2.set_title('Ganancia de Diversidad en SNR', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        for i, gain in enumerate(gains):
            ax2.text(i, gain + 0.1, f'{gain:.1f} dB', ha='center', fontsize=10)

    plt.suptitle(f'{M_qam}-QAM: Ganancia de Diversidad MRC', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig
