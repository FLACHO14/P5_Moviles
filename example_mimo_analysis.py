#!/usr/bin/env python3
"""
example_mimo_analysis.py
Ejemplo de uso del simulador MIMO extendido.

Demuestra:
1. Transmisión SFBC (2 antenas TX)
2. Recepción SIMO-MRC (múltiples antenas RX)
3. Análisis comparativo SISO vs SIMO vs MISO
4. Visualizaciones: BER, constelaciones, PAPR
"""

import numpy as np
import matplotlib.pyplot as plt
from ofdm_params import PILOT_AMPLITUDE
from ofdm_utils import get_nfft_cp, build_subcarrier_map
from ofdm_tx import qam_mod, ofdm_tx_block, sfbc_tx_2ant, calculate_papr
from ofdm_rx import ofdm_rx_block, equalize_with_pilots, qam_demod, combine_mrc_frequency
from ofdm_channel import (get_channel_profile, apply_channel, apply_channel_mimo,
                           apply_channel_miso, generate_mimo_channels, generate_miso_channels)
from ofdm_mimo_utils import (plot_ber_comparison_mimo, plot_constellation_comparison_mrc,
                              plot_papr_scfdma_vs_ofdm_mimo, plot_channel_power_stability_mimo)


def example_1_sfbc_transmission():
    """Ejemplo 1: Transmisión SFBC con 2 antenas."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Transmisión SFBC (2 antenas TX)")
    print("="*60)

    Nfft, cp_len = get_nfft_cp(10e6)
    sc_map = build_subcarrier_map(Nfft)

    n_bits = 1000
    bits = np.random.randint(0, 2, n_bits)
    symbols_data = qam_mod(bits, M=16)

    print(f"\nParámetros LTE:")
    print(f"  FFT size: {Nfft}, CP length: {cp_len}")
    print(f"  Data subcarriers: {sc_map['n_data']}")
    print(f"  Símbolos QAM: {len(symbols_data)}")

    tx1, tx2, papr_list, n_ofdm, _, _ = sfbc_tx_2ant(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

    papr_avg = np.mean(papr_list)
    papr_max = np.max(papr_list)
    papr_min = np.min(papr_list)

    print(f"\nSFBC (Alamouti) Transmisión:")
    print(f"  Símbolos OFDM: {n_ofdm}")
    print(f"  PAPR (promedio): {papr_avg:.2f} dB")
    print(f"  PAPR (máximo): {papr_max:.2f} dB")
    print(f"  PAPR (mínimo): {papr_min:.2f} dB")
    print(f"  Señal TX Ant1: {len(tx1)} muestras")
    print(f"  Señal TX Ant2: {len(tx2)} muestras")


def example_2_simo_reception():
    """Ejemplo 2: Recepción SIMO con MRC."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Recepción SIMO (2 antenas RX + MRC)")
    print("="*60)

    Nfft, cp_len = get_nfft_cp(10e6)
    sc_map = build_subcarrier_map(Nfft)

    n_bits = 2000
    bits = np.random.randint(0, 2, n_bits)
    symbols_data = qam_mod(bits, M=16)

    tx_signal, _, _ = ofdm_tx_block(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

    snr_db = 15
    h_channels = generate_mimo_channels(2, 'Rayleigh', taps_L=8)

    print(f"\nParámetros:")
    print(f"  SNR: {snr_db} dB")
    print(f"  Canal: Rayleigh (8 taps)")
    print(f"  Antenas RX: 2")

    rx_signals = apply_channel_mimo(tx_signal, h_channels, snr_db, velocity_kmh=50)

    print(f"  Señales RX recibidas: {len(rx_signals)}")

    Y_frames_list = [ofdm_rx_block(rx, Nfft, cp_len) for rx in rx_signals]

    H_est_list = []
    for Y in Y_frames_list:
        _, H_est, _ = equalize_with_pilots(Y, sc_map, PILOT_AMPLITUDE, Nfft)
        H_est_list.append(np.array(H_est))

    print(f"  Símbolos OFDM recibidos por antena: {Y_frames_list[0].shape[0]}")

    data_mrc = combine_mrc_frequency(Y_frames_list, H_est_list, sc_map)
    bits_recovered = (np.angle(data_mrc) > 0).astype(int)
    ber_mrc = np.mean(bits_recovered[:len(bits)] != bits[:len(bits_recovered)])

    print(f"\nResultados MRC:")
    print(f"  Símbolos combinados: {len(data_mrc)}")
    print(f"  BER: {ber_mrc:.4e}")
    print(f"  Bits recuperados correctamente: {(1-ber_mrc)*100:.1f}%")


def example_3_miso_sfbc():
    """Ejemplo 3: MISO con SFBC."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Transmisión MISO (2 antenas TX + SFBC)")
    print("="*60)

    Nfft, cp_len = get_nfft_cp(10e6)
    sc_map = build_subcarrier_map(Nfft)

    n_bits = 2000
    bits = np.random.randint(0, 2, n_bits)
    symbols_data = qam_mod(bits, M=16)

    tx1, tx2, papr_list, n_ofdm, _, _ = sfbc_tx_2ant(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

    snr_db = 15
    h_channels = generate_miso_channels(2, 'Rayleigh', taps_L=8)

    print(f"\nParámetros:")
    print(f"  SNR: {snr_db} dB")
    print(f"  Canal: Rayleigh (8 taps) independiente por antena TX")
    print(f"  Antenas TX: 2 (SFBC)")
    print(f"  Antenas RX: 1")

    rx_signal, _ = apply_channel_miso([tx1, tx2], h_channels, snr_db, velocity_kmh=50)

    print(f"  Señal RX: {len(rx_signal)} muestras")

    Y_frames = ofdm_rx_block(rx_signal, Nfft, cp_len)
    data_eq, _, _ = equalize_with_pilots(Y_frames, sc_map, PILOT_AMPLITUDE, Nfft)

    bits_recovered = (np.angle(data_eq) > 0).astype(int)
    ber_sfbc = np.mean(bits_recovered[:len(bits)] != bits[:len(bits_recovered)])

    print(f"\nResultados SFBC:")
    print(f"  Símbolos ecualizados: {len(data_eq)}")
    print(f"  BER: {ber_sfbc:.4e}")
    print(f"  Bits recuperados correctamente: {(1-ber_sfbc)*100:.1f}%")


def example_4_papr_comparison():
    """Ejemplo 4: Comparativa PAPR entre técnicas."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Comparativa PAPR")
    print("="*60)

    Nfft, cp_len = get_nfft_cp(10e6)
    sc_map = build_subcarrier_map(Nfft)

    papr_ofdm_list = []
    papr_sfbc_list = []

    for _ in range(100):
        n_bits = 1000
        bits = np.random.randint(0, 2, n_bits)
        symbols_data = qam_mod(bits, M=16)

        tx_ofdm, papr_ofdm, _ = ofdm_tx_block(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)
        papr_ofdm_list.extend(papr_ofdm)

        tx1, tx2, papr_sfbc, _, _, _ = sfbc_tx_2ant(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)
        papr_sfbc_list.extend(papr_sfbc)

    print(f"\nPAPR Estadísticas (100 iteraciones):")
    print(f"\nOFDM:")
    print(f"  Promedio: {np.mean(papr_ofdm_list):.2f} dB")
    print(f"  Máximo: {np.max(papr_ofdm_list):.2f} dB")
    print(f"  Mínimo: {np.min(papr_ofdm_list):.2f} dB")
    print(f"  Desv. Est.: {np.std(papr_ofdm_list):.2f} dB")

    print(f"\nSFBC (Alamouti):")
    print(f"  Promedio: {np.mean(papr_sfbc_list):.2f} dB")
    print(f"  Máximo: {np.max(papr_sfbc_list):.2f} dB")
    print(f"  Mínimo: {np.min(papr_sfbc_list):.2f} dB")
    print(f"  Desv. Est.: {np.std(papr_sfbc_list):.2f} dB")

    reduction = np.mean(papr_ofdm_list) - np.mean(papr_sfbc_list)
    print(f"\nReducción PAPR (OFDM → SFBC): {reduction:.2f} dB ({reduction/np.mean(papr_ofdm_list)*100:.1f}%)")


def main():
    """Ejecuta todos los ejemplos."""
    print("\n" + "#"*60)
    print("# SIMULADOR MIMO EXTENDIDO - EJEMPLOS DE USO")
    print("#"*60)

    example_1_sfbc_transmission()
    example_2_simo_reception()
    example_3_miso_sfbc()
    example_4_papr_comparison()

    print("\n" + "="*60)
    print("GENERANDO VISUALIZACIONES...")
    print("="*60)

    fig1 = plt.figure(figsize=(12, 6))
    ax = fig1.add_subplot(111)

    Nfft, cp_len = get_nfft_cp(10e6)
    sc_map = build_subcarrier_map(Nfft)

    bits = np.random.randint(0, 2, 2000)
    symbols_data = qam_mod(bits, M=16)

    tx_signal, _, _ = ofdm_tx_block(symbols_data, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

    h_single = get_channel_profile('Rayleigh', taps_L=8)
    h_channels = generate_mimo_channels(2, 'Rayleigh', taps_L=8)

    rx_single, _ = apply_channel(tx_signal, h_single, 15, velocity_kmh=50)
    rx_mimo = apply_channel_mimo(tx_signal, h_channels, 15, velocity_kmh=50)

    Y_frames_list = [ofdm_rx_block(rx, Nfft, cp_len) for rx in rx_mimo]

    H_est_list = []
    for Y in Y_frames_list:
        _, H_est, _ = equalize_with_pilots(Y, sc_map, PILOT_AMPLITUDE, Nfft)
        H_est_list.append(np.array(H_est))

    H_combined = np.zeros(Nfft, dtype=complex)
    for H in H_est_list:
        H_combined += np.conj(H[0]) * H[0]
    H_combined = np.sqrt(H_combined)

    subcarriers = np.arange(len(h_single))
    ax.plot(subcarriers, 10*np.log10(np.abs(h_single)**2 + 1e-12), 'r-', alpha=0.7, label='1 antena (SISO)')
    ax.plot(np.arange(len(H_combined)), 10*np.log10(np.abs(H_combined)**2 + 1e-12), 'g-', alpha=0.7, label='MRC combinado')
    ax.set_xlabel('Subportadora', fontsize=11)
    ax.set_ylabel('Potencia (dB)', fontsize=11)
    ax.set_title('Respuesta de Canal: SISO vs MRC', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("\nVisualización: Respuesta de Canal guardada")
    plt.tight_layout()
    plt.savefig('mimo_channel_response.png', dpi=150)
    plt.close()

    print("\n✓ Ejemplos completados exitosamente")
    print("  Consulta los módulos ofdm_mimo_utils para más visualizaciones")


if __name__ == '__main__':
    main()
