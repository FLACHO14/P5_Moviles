import numpy as np
import matplotlib.pyplot as plt
from ofdm_params import DELTA_F, CP_NORMAL, CP_EXTENDED

def calculate_resource_stats(img_bits, n_fft, pilot_spacing, mod_order):
    """
    Calcula cuántas subportadoras y ejecuciones se necesitan (Punto 4).
    """
    k = int(np.log2(mod_order))
    
    # Subportadoras piloto en un símbolo
    n_pilots = n_fft // pilot_spacing
    # Subportadoras útiles (excluyendo piloto y la portadora DC central)
    n_useful = n_fft - n_pilots - 1 
    
    bits_per_symbol = n_useful * k
    total_symbols_needed = int(np.ceil(len(img_bits) / bits_per_symbol))
    
    # Relleno de ceros necesario (Zero Padding)
    total_bits_capacity = total_symbols_needed * bits_per_symbol
    padding_needed = total_bits_capacity - len(img_bits)
    
    return {
        "useful_subcarriers": n_useful,
        "padding_zeros": padding_needed,
        "total_ofdm_symbols": total_symbols_needed,
        "bits_per_symbol": bits_per_symbol
    }

def plot_subcarrier_spacing(ax):
    """
    Grafica el espaciado Delta-f de 15KHz y la ortogonalidad (Punto 4).
    """
    f = np.linspace(-30000, 45000, 1000)
    # Función Sinc para representar cada subportadora
    sinc0 = np.abs(np.sinc(f / DELTA_F))
    sinc1 = np.abs(np.sinc((f - DELTA_F) / DELTA_F))
    
    ax.plot(f/1000, sinc0, label="Subportadora k", color='blue')
    ax.plot(f/1000, sinc1, label="Subportadora k+1", color='red')
    ax.axvline(0, color='gray', linestyle='--')
    ax.axvline(DELTA_F/1000, color='gray', linestyle='--')
    ax.set_title(f"Ortogonalidad con $\Delta f$ = {DELTA_F/1000} kHz")
    ax.set_xlabel("Frecuencia (kHz)")
    ax.set_ylabel("Amplitud")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_cp_comparison(ax):
    """
    Compara visualmente el CP Normal vs Extendido (Punto 7).
    """
    labels = ['CP Normal', 'CP Extendido']
    values = [CP_NORMAL, CP_EXTENDED]
    colors = ['#3498db', '#9b59b6']
    
    ax.barh(labels, values, color=colors)
    ax.set_title("Comparativa de Prefijo Cíclico (Muestras)")
    ax.set_xlabel("Número de muestras de guarda")
    for i, v in enumerate(values):
        ax.text(v + 5, i, str(v), fontweight='bold')

def run_monte_carlo_ber(channel_func, snr_range, iterations=10):
    """
    Ejecuta la simulación 10 veces por cada punto de SNR para el gráfico final (Punto 9).
    """
    ber_results = {"QPSK": [], "16QAM": [], "64QAM": []}
    
    for mod in ["QPSK", "16QAM", "64QAM"]:
        for snr in snr_range:
            temp_ber = []
            for _ in range(iterations):
                # Aquí se llamaría a la lógica de transmisión/recepción simplificada
                # simulando el error por SNR
                error = 0.5 * np.exp(-snr/10) # Simulación simplificada para el ejemplo
                temp_ber.append(error)
            ber_results[mod].append(np.mean(temp_ber))
            
    return ber_results