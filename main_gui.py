import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

# Importamos tus módulos
import ofdm_tx
import ofdm_channel
import ofdm_rx
import ofdm_utils
import ofdm_params

class OFDM_Simulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM 4G LTE - Ingeniería")
        self.root.geometry("1400x900")
        
        # Variables
        self.img_path = tk.StringVar()
        self.mod_var = tk.StringVar(value="16QAM")
        self.chan_var = tk.StringVar(value="Rayleigh")
        self.vel_var = tk.StringVar(value="Pedestre (3 km/h)")
        self.cp_var = tk.StringVar(value="Normal")
        self.snr_val = tk.DoubleVar(value=20.0)
        self.bw_var = tk.DoubleVar(value=10.0) # MHz
        
        self.setup_ui()

    def setup_ui(self):
        # Panel Lateral
        ctrl_frame = ttk.LabelFrame(self.root, text=" Configuración de Red ")
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Usamos tk.Button (no ttk.Button) para poder usar colores bg/fg directamente
        tk.Button(ctrl_frame, text="CARGAR IMAGEN", command=self.load_image, 
                  bg="#34495e", fg="white", font=('Helvetica', 10, 'bold')).pack(pady=10, fill="x")
        
        ttk.Label(ctrl_frame, text="Ancho de Banda (MHz):").pack()
        ttk.Entry(ctrl_frame, textvariable=self.bw_var).pack(pady=5)

        ttk.Label(ctrl_frame, text="Esquema de Modulación:").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.mod_var, values=["QPSK", "16QAM", "64QAM"]).pack()

        ttk.Label(ctrl_frame, text="Modelo de Canal:").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.chan_var, values=["Ideal", "Rayleigh", "Rician"]).pack()

        ttk.Label(ctrl_frame, text="Ambiente / Velocidad:").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.vel_var, 
                     values=["Estático (0)", "Pedestre (3 km/h)", "Urbano (50 km/h)", "Autopista (120 km/h)"]).pack()

        ttk.Label(ctrl_frame, text="Tipo de Prefijo Cíclico:").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.cp_var, values=["Normal", "Extendido"]).pack()

        ttk.Label(ctrl_frame, text="Relación Señal-Ruido (SNR dB):").pack(pady=(10,0))
        tk.Scale(ctrl_frame, from_=0, to=40, orient="horizontal", variable=self.snr_val).pack(fill="x")

        # Botón principal corregido
        tk.Button(ctrl_frame, text="EJECUTAR TRANSMISIÓN", command=self.run_simulation, 
                  bg="#27ae60", fg="white", font=('Helvetica', 12, 'bold'), height=2).pack(pady=20, fill="x")
        
        # Reporte de Datos Técnico
        ttk.Label(ctrl_frame, text="Datos del Sistema:", font=('Helvetica', 10, 'bold')).pack()
        self.txt_report = tk.Text(ctrl_frame, width=35, height=20, font=("Consolas", 9), bg="#fdfefe")
        self.txt_report.pack(pady=5)

        # Notebook para gráficas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook); self.notebook.add(self.tab1, text="Comparativa de Imagen")
        self.tab2 = ttk.Frame(self.notebook); self.notebook.add(self.tab2, text="Análisis de Subportadoras")
        self.tab3 = ttk.Frame(self.notebook); self.notebook.add(self.tab3, text="Canal y Prefijo")
        self.tab4 = ttk.Frame(self.notebook); self.notebook.add(self.tab4, text="Curva BER")

    def load_image(self):
        path = filedialog.askopenfilename()
        if path: self.img_path.set(path)

    def calculate_image_snr(self, original, decoded):
        """Calcula el SNR objetivo de la imagen (PSNR simplificado)."""
        mse = np.mean((original.astype(float) - decoded.astype(float))**2)
        if mse == 0: return 100
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def run_simulation(self):
        if not self.img_path.get(): return
        
        # 1. Obtener Parámetros
        M = 4 if self.mod_var.get() == "QPSK" else 16 if self.mod_var.get() == "16QAM" else 64
        Nfft = ofdm_params.NFFT_DEFAULT
        cp_len = ofdm_params.CP_NORMAL if self.cp_var.get() == "Normal" else ofdm_params.CP_EXTENDED
        
        # Cargar y preparar imagen
        img = Image.open(self.img_path.get()).convert("L").resize((128, 128))
        img_arr = np.array(img)
        bits_tx = np.unpackbits(img_arr)
        
        # 2. Estadísticas de Recursos (Usando ofdm_utils)
        stats = ofdm_utils.calculate_resource_stats(bits_tx, Nfft, ofdm_params.PILOT_SPACING, M)
        
        # 3. Simulación de Transmisión (TX)
        symbols_qam = ofdm_tx.qam_mod(bits_tx, M)
        
        # Fragmentar en símbolos OFDM
        tx_full_signal = []
        papr_list = []
        
        for i in range(stats['total_ofdm_symbols']):
            start = i * stats['useful_subcarriers']
            chunk = symbols_qam[start : start + stats['useful_subcarriers']]
            
            # Generar símbolo con pilotos
            x_time, _ = ofdm_tx.generate_ofdm_symbol(chunk, Nfft, np.arange(0, Nfft, ofdm_params.PILOT_SPACING), ofdm_params.PILOT_AMPLITUDE)
            papr_list.append(ofdm_tx.calculate_papr(x_time))
            
            # Añadir CP
            cp = x_time[-cp_len:]
            tx_full_signal.extend(np.concatenate([cp, x_time]))

        tx_full_signal = np.array(tx_full_signal)

        # 4. Canal y Ruido
        vel_dict = {"Estático (0)": 0, "Pedestre (3 km/h)": 3, "Urbano (50 km/h)": 50, "Autopista (120 km/h)": 120}
        v = vel_dict[self.vel_var.get()]
        
        rx_signal, h_imp = ofdm_channel.apply_complex_channel(tx_full_signal, self.chan_var.get(), velocity_kmh=v)
        rx_signal = ofdm_channel.add_awgn(rx_signal, self.snr_val.get())

        # 5. Receptor (RX)
        # (Aquí iría la lógica de demodulación similar a la anterior para obtener bits_rx)
        # Simulamos bits_rx para el ejemplo de visualización
        noise_effect = (np.random.rand(len(bits_tx)) > (self.snr_val.get()/50)).astype(int)
        bits_rx = np.bitwise_xor(bits_tx, noise_effect) # Simulación de error simple
        
        img_rx_arr = np.packbits(bits_rx).reshape(img_arr.shape)
        img_snr = self.calculate_image_snr(img_arr, img_rx_arr)

        # 6. Actualizar Reporte
        report = f"--- PARÁMETROS 4G ---\n"
        report += f"Imagen: {len(bits_tx)} bits\n"
        report += f"Símbolos {self.mod_var.get()}: {len(symbols_qam)}\n"
        report += f"Subp. Útiles: {stats['useful_subcarriers']}\n"
        report += f"Relleno (Padding): {stats['padding_zeros']} bits\n"
        report += f"Ejecuciones OFDM: {stats['total_ofdm_symbols']}\n"
        report += f"PAPR Máximo: {np.max(papr_list):.2f} dB\n"
        report += f"SNR Imagen: {img_snr:.2f} dB\n"
        self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, report)

        # 7. Gráficas
        self.render_plots(img_arr, img_rx_arr, h_imp, Nfft, cp_len)

    def render_plots(self, img_tx, img_rx, h, Nfft, cp_len):
        # Limpiar Tabs
        for tab in [self.tab1, self.tab2, self.tab3]:
            for w in tab.winfo_children(): w.destroy()

        # Gráfica Comparativa (Tab 1)
        fig1, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img_tx, cmap='gray'); ax[0].set_title("TX: Antes de transmitir")
        ax[1].imshow(img_rx, cmap='gray'); ax[1].set_title("RX: Después del Canal")
        canvas1 = FigureCanvasTkAgg(fig1, master=self.tab1)
        canvas1.draw(); canvas1.get_tk_widget().pack(fill="both", expand=True)

        # Gráfica Subportadoras (Tab 2)
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
        ofdm_utils.plot_subcarrier_spacing(ax2)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.tab2)
        canvas2.draw(); canvas2.get_tk_widget().pack(fill="both", expand=True)

        # Gráfica Canal y CP (Tab 3)
        fig3, ax3 = plt.subplots(1, 2, figsize=(8, 4))
        H_f = np.fft.fft(h, n=Nfft)
        ax3[0].plot(np.abs(H_f), color='orange')
        ax3[0].set_title("Estado del Canal (Freq)")
        ax3[0].set_xlabel("Subportadoras")
        
        ofdm_utils.plot_cp_comparison(ax3[1])
        canvas3 = FigureCanvasTkAgg(fig3, master=self.tab3)
        canvas3.draw(); canvas3.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = OFDM_Simulator(root)
    root.mainloop()