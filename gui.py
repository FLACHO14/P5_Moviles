# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
from PIL import Image, ImageTk

from ofdm_params import OFDMParams
from ofdm_utils import image_to_bitstream, bitstream_to_image, calculate_ber, bits_to_symbols
from ofdm_tx import OFDMTransmitter
from ofdm_rx import OFDMReceiver
from ofdm_channel import OFDMChannel

class OFDM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema OFDM - Comunicaciones Digitales")
        self.root.geometry("1500x950")
        self.params = OFDMParams()
        self.image_path = None
        self.original_bits = None
        self.image_shape = None
        self.tx_signal = None
        self.rx_signal = None
        self.create_widgets()

    def create_widgets(self):
        left_frame = ttk.Frame(self.root, width=450)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left_frame, text="Parámetros OFDM", font=('Arial',12,'bold')).pack(pady=5)

        ttk.Button(left_frame, text="Cargar Imagen", command=self.load_image).pack(pady=2)
        self.img_label = ttk.Label(left_frame, text="Ninguna imagen cargada")
        self.img_label.pack()
        self.img_preview = ttk.Label(left_frame)
        self.img_preview.pack()

        self.info_bits = ttk.Label(left_frame, text="Bits originales: --")
        self.info_bits.pack()
        self.info_symbs = ttk.Label(left_frame, text="Símbolos modulados: --")
        self.info_symbs.pack()

        ttk.Label(left_frame, text="Modulación:").pack()
        self.mod_combo = ttk.Combobox(left_frame, values=['QPSK','16QAM','64QAM'], state='readonly')
        self.mod_combo.set('16QAM')
        self.mod_combo.pack()
        self.mod_combo.bind('<<ComboboxSelected>>', lambda e: self.update_bits_info())

        ttk.Label(left_frame, text="Ancho de banda (MHz):").pack()
        self.bw_entry = ttk.Entry(left_frame)
        self.bw_entry.insert(0, "5")
        self.bw_entry.pack()

        ttk.Label(left_frame, text="Espaciado subportadoras (kHz):").pack()
        self.df_entry = ttk.Entry(left_frame)
        self.df_entry.insert(0, "15")
        self.df_entry.pack()

        ttk.Label(left_frame, text="Tamaño IFFT (0=auto):").pack()
        self.fft_entry = ttk.Entry(left_frame)
        self.fft_entry.insert(0, "0")
        self.fft_entry.pack()

        ttk.Label(left_frame, text="Prefijo cíclico:").pack()
        self.cp_combo = ttk.Combobox(left_frame, values=['normal','extended'], state='readonly')
        self.cp_combo.set('normal')
        self.cp_combo.pack()

        ttk.Label(left_frame, text="Canal LoS/NLoS:").pack()
        self.los_combo = ttk.Combobox(left_frame, values=['LoS','NLoS'], state='readonly')
        self.los_combo.set('LoS')
        self.los_combo.pack()
        ttk.Label(left_frame, text="Número de multicopias:").pack()
        self.multipath_entry = ttk.Entry(left_frame)
        self.multipath_entry.insert(0, "3")
        self.multipath_entry.pack()
        ttk.Label(left_frame, text="Velocidad (km/h):").pack()
        self.speed_entry = ttk.Entry(left_frame)
        self.speed_entry.insert(0, "30")
        self.speed_entry.pack()
        ttk.Label(left_frame, text="SNR (dB):").pack()
        self.snr_entry = ttk.Entry(left_frame)
        self.snr_entry.insert(0, "20")
        self.snr_entry.pack()

        self.btn_single = ttk.Button(left_frame, text="Transmitir y recibir", command=self.run_single)
        self.btn_single.pack(pady=10)
        self.btn_mc = ttk.Button(left_frame, text="Simulación Monte Carlo", command=self.run_monte_carlo)
        self.btn_mc.pack(pady=5)

        # Barra de progreso
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate', length=300)
        self.progress.pack(pady=10)
        self.lbl_status = ttk.Label(left_frame, text="")
        self.lbl_status.pack()

        # Panel derecho con pestañas
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_tab_constelacion()
        self.create_tab_subportadoras()
        self.create_tab_prefijo()
        self.create_tab_canal()
        self.create_tab_papr()
        self.create_tab_imagen_recibida()
        self.create_tab_montecarlo()

    def create_tab_constelacion(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Constelaciones")
        fig = Figure(figsize=(6,5))
        self.ax_tx_const = fig.add_subplot(121)
        self.ax_rx_const_raw = fig.add_subplot(122)
        self.ax_tx_const.set_title("Transmitida")
        self.ax_rx_const_raw.set_title("Recibida (antes ecual.)")
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_const = fig
        self.canvas_const = canvas

    def create_tab_subportadoras(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Subportadoras")
        fig = Figure(figsize=(6,4))
        self.ax_subc = fig.add_subplot(111)
        self.ax_subc.set_xlabel("Índice de subportadora")
        self.ax_subc.set_ylabel("|Amplitud|")
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_subc = fig

    def create_tab_prefijo(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Prefijo Cíclico")
        fig = Figure(figsize=(6,4))
        self.ax_cp = fig.add_subplot(111)
        self.ax_cp.set_xlabel("Muestras")
        self.ax_cp.set_ylabel("Amplitud")
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_cp = fig

    def create_tab_canal(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Canal")
        fig = Figure(figsize=(6,5))
        self.ax_ch_time = fig.add_subplot(221)
        self.ax_ch_freq = fig.add_subplot(222)
        self.ax_ch_info = fig.add_subplot(212)
        self.ax_ch_time.set_title("Respuesta impulsional")
        self.ax_ch_freq.set_title("Respuesta en frecuencia")
        self.ax_ch_info.axis('off')
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_ch = fig

    def create_tab_papr(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="PAPR")
        fig = Figure(figsize=(6,5))
        self.ax_papr_time = fig.add_subplot(121)
        self.ax_papr_ccdf = fig.add_subplot(122)
        self.ax_papr_time.set_xlabel("Muestras")
        self.ax_papr_time.set_ylabel("Potencia")
        self.ax_papr_ccdf.set_xlabel("PAPR (dB)")
        self.ax_papr_ccdf.set_ylabel("Prob(PAPR > PAPR0)")
        self.ax_papr_ccdf.set_yscale('log')
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_papr = fig

    def create_tab_imagen_recibida(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Imagen Recibida")
        fig = Figure(figsize=(5,5))
        self.ax_img = fig.add_subplot(111)
        self.ax_img.set_title("Imagen reconstruida")
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_img = fig

    def create_tab_montecarlo(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Monte Carlo")
        fig = Figure(figsize=(6,4))
        self.ax_mc = fig.add_subplot(111)
        self.ax_mc.set_xlabel("SNR (dB)")
        self.ax_mc.set_ylabel("BER")
        self.ax_mc.set_yscale('log')
        self.ax_mc.grid(True)
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        self.fig_mc = fig

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.bmp")])
        if path:
            self.image_path = path
            self.original_bits, self.image_shape = image_to_bitstream(path)
            img = Image.open(path)
            img.thumbnail((120,120))
            self.img_tk = ImageTk.PhotoImage(img)
            self.img_preview.config(image=self.img_tk)
            self.img_label.config(text=f"Imagen: {self.image_shape[0]}x{self.image_shape[1]}\nBits: {len(self.original_bits)}")
            self.update_bits_info()

    def update_bits_info(self):
        if self.original_bits is None:
            return
        mod = self.mod_combo.get()
        bits_per_symbol = {'QPSK':2, '16QAM':4, '64QAM':6}[mod]
        num_symbols = (len(self.original_bits) + bits_per_symbol - 1) // bits_per_symbol
        self.info_bits.config(text=f"Bits originales: {len(self.original_bits)}")
        self.info_symbs.config(text=f"Símbolos después de {mod}: {num_symbols}")

    def update_params(self):
        self.params.bandwidth = float(self.bw_entry.get()) * 1e6
        self.params.subcarrier_spacing = float(self.df_entry.get()) * 1e3
        self.params.cp_type = self.cp_combo.get()
        self.params.modulation = self.mod_combo.get()
        self.params.channel_los = (self.los_combo.get() == 'LoS')
        self.params.num_multipath = int(self.multipath_entry.get())
        self.params.doppler_speed = float(self.speed_entry.get())
        self.params.snr_db = float(self.snr_entry.get())
        self.params.compute_subcarriers()
        fft_manual = int(self.fft_entry.get())
        if fft_manual > 0 and (fft_manual & (fft_manual-1) == 0):
            self.params.fft_size = fft_manual
            self.params.compute_subcarriers()
        self.update_bits_info()

    def set_busy(self, busy=True):
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_single.config(state=state)
        self.btn_mc.config(state=state)
        if busy:
            self.progress.start(10)
            self.lbl_status.config(text="Procesando...")
        else:
            self.progress.stop()
            self.lbl_status.config(text="")

    def run_single(self):
        if self.original_bits is None:
            messagebox.showerror("Error", "Cargue una imagen primero")
            return
        self.set_busy(True)
        self.update_params()
        # Ejecutar en un hilo para no bloquear la GUI
        threading.Thread(target=self._single_thread, daemon=True).start()

    def _single_thread(self):
        try:
            symbols_tx = bits_to_symbols(self.original_bits, self.params.modulation)

            # Actualizar constelación TX en el hilo principal
            self.root.after(0, lambda: self.update_tx_constellation(symbols_tx))

            tx = OFDMTransmitter(self.params)
            tx_signal = tx.create_ofdm_symbols(symbols_tx)
            self.tx_signal = tx_signal

            channel = OFDMChannel(self.params)
            rx_signal, taps, delays, (freq_axis, H_freq), channel_class = channel.apply_channel(tx_signal, self.params.snr_db)
            self.rx_signal = rx_signal

            rx = OFDMReceiver(tx, self.params)
            received_bits, rx_symbols_raw, rx_symbols_eq, _ = rx.process(rx_signal)

            ber = calculate_ber(self.original_bits, received_bits)

            # Actualizar todas las gráficas en el hilo principal
            self.root.after(0, lambda: self.update_all_plots(tx, rx_symbols_raw, rx_symbols_eq, taps, delays, freq_axis, H_freq, channel_class, ber, received_bits))
            self.root.after(0, lambda: messagebox.showinfo("Éxito", f"Transmisión completada\nBER = {ber:.6f}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def update_tx_constellation(self, symbols_tx):
        self.ax_tx_const.clear()
        self.ax_tx_const.scatter(symbols_tx.real, symbols_tx.imag, s=1, alpha=0.5)
        self.ax_tx_const.set_title(f"TX - {self.params.modulation}")
        self.ax_tx_const.grid(True)
        self.canvas_const.draw()

    def update_all_plots(self, tx, rx_symbols_raw, rx_symbols_eq, taps, delays, freq_axis, H_freq, channel_class, ber, received_bits):
        # Constelación recibida
        self.ax_rx_const_raw.clear()
        self.ax_rx_const_raw.scatter(rx_symbols_raw.real, rx_symbols_raw.imag, s=1, alpha=0.5, c='red')
        self.ax_rx_const_raw.set_title("RX antes ecual.")
        self.ax_rx_const_raw.grid(True)
        self.canvas_const.draw()

        # Subportadoras
        self.ax_subc.clear()
        subc = tx.subcarrier_grid[0]
        indices = np.arange(len(subc))
        mask = np.abs(subc) > 0
        self.ax_subc.bar(indices[mask], np.abs(subc[mask]), width=1.0, color='blue')
        self.ax_subc.set_title(f"Subportadoras (FFT={tx.fft_size}, Útiles={tx.params.N_useful_subcarriers}, Pilotos={tx.num_pilots})")
        self.ax_subc.set_xlim(0, tx.fft_size-1)
        self.fig_subc.canvas.draw()

        # Prefijo cíclico
        self.ax_cp.clear()
        sym_len = tx.fft_size + tx.cp_len
        y_plot = np.abs(tx.tx_signal[:sym_len])
        self.ax_cp.plot(y_plot)
        self.ax_cp.axvline(x=tx.cp_len, color='r', linestyle='--', label='Fin CP')
        self.ax_cp.fill_between(range(tx.cp_len), y_plot[:tx.cp_len], alpha=0.3, label='CP')
        self.ax_cp.set_title(f"Prefijo cíclico (longitud={tx.cp_len})")
        self.ax_cp.legend()
        self.fig_cp.canvas.draw()

        # Canal
        self.ax_ch_time.clear()
        self.ax_ch_time.stem(delays, np.abs(taps), basefmt=' ')
        self.ax_ch_time.set_title("Perfil de potencia (tiempo)")
        self.ax_ch_time.set_xlabel("Retardo (muestras)")
        self.ax_ch_time.set_ylabel("|Amplitud|")
        self.ax_ch_freq.clear()
        self.ax_ch_freq.plot(freq_axis/1e6, 20*np.log10(np.abs(H_freq)+1e-6))
        self.ax_ch_freq.set_title("Respuesta en frecuencia")
        self.ax_ch_freq.set_xlabel("Frecuencia (MHz)")
        self.ax_ch_freq.set_ylabel("Magnitud (dB)")
        self.ax_ch_info.clear()
        self.ax_ch_info.axis('off')
        self.ax_ch_info.text(0.1, 0.8, f"Clasificación del canal:\n{channel_class}", fontsize=10, verticalalignment='top')
        self.fig_ch.canvas.draw()

        # PAPR
        power = np.abs(tx.tx_signal)**2
        avg_power = np.mean(power)
        papr_db = 10*np.log10(np.max(power)/avg_power)
        self.ax_papr_time.clear()
        self.ax_papr_time.plot(power, alpha=0.7)
        self.ax_papr_time.axhline(y=avg_power, color='r', linestyle='--', label=f'Promedio = {avg_power:.2f}')
        self.ax_papr_time.set_title(f"PAPR = {papr_db:.2f} dB")
        self.ax_papr_time.legend()
        papr_vals = 10*np.log10(power/avg_power + 1e-9)
        hist, bins = np.histogram(papr_vals, bins=50, density=True)
        ccdf = 1 - np.cumsum(hist) * (bins[1]-bins[0])
        self.ax_papr_ccdf.clear()
        self.ax_papr_ccdf.plot(bins[:-1], ccdf, 'b-')
        self.ax_papr_ccdf.set_yscale('log')
        self.ax_papr_ccdf.set_xlabel("PAPR (dB)")
        self.ax_papr_ccdf.set_ylabel("Prob(PAPR > PAPR0)")
        self.ax_papr_ccdf.grid(True)
        self.fig_papr.canvas.draw()

        # Imagen recibida
        try:
            rec_img = bitstream_to_image(received_bits, self.image_shape)
            self.ax_img.clear()
            self.ax_img.imshow(rec_img, cmap='gray')
            self.ax_img.set_title(f"Imagen recibida - BER = {ber:.6f}")
            self.fig_img.canvas.draw()
        except Exception as e:
            self.ax_img.clear()
            self.ax_img.text(0.5,0.5, f"Error al reconstruir: {e}", ha='center')
            self.fig_img.canvas.draw()

    def run_monte_carlo(self):
        if self.original_bits is None:
            messagebox.showerror("Error", "Cargue una imagen primero")
            return
        self.set_busy(True)
        self.update_params()
        snr_range = np.arange(0, 21, 2)
        modulations = ['QPSK', '16QAM', '64QAM']
        results = {mod: {'mean': [], 'ci': []} for mod in modulations}
        # Ejecutar en hilo
        threading.Thread(target=self._mc_thread, args=(snr_range, modulations, results), daemon=True).start()

    def _mc_thread(self, snr_range, modulations, results):
        try:
            for mod in modulations:
                self.params.modulation = mod
                bits_per_sym = {'QPSK':2, '16QAM':4, '64QAM':6}[mod]
                num_sym = (len(self.original_bits) + bits_per_sym - 1) // bits_per_sym
                for snr in snr_range:
                    bers = []
                    for _ in range(10):
                        symbols_tx = bits_to_symbols(self.original_bits, mod)
                        tx = OFDMTransmitter(self.params)
                        tx_signal = tx.create_ofdm_symbols(symbols_tx)
                        channel = OFDMChannel(self.params)
                        rx_signal, _, _, _, _ = channel.apply_channel(tx_signal, snr)
                        rx = OFDMReceiver(tx, self.params)
                        received_bits, _, _, _ = rx.process(rx_signal)
                        ber = calculate_ber(self.original_bits, received_bits)
                        bers.append(ber)
                    mean_ber = np.mean(bers)
                    std_ber = np.std(bers)
                    ci = 1.96 * std_ber / np.sqrt(10)
                    results[mod]['mean'].append(mean_ber)
                    results[mod]['ci'].append(ci)
                    # Actualizar gráfica en hilo principal
                    self.root.after(0, lambda m=mod, snr_idx=len(results[mod]['mean'])-1: self.update_mc_plot(snr_range, modulations, results, m, snr_idx))
                # Pequeña pausa para no saturar
                self.root.after(100, None)
            self.root.after(0, lambda: messagebox.showinfo("Montecarlo", "Simulación completada"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def update_mc_plot(self, snr_range, modulations, results, current_mod, idx):
        self.ax_mc.clear()
        for mod in modulations:
            y = results[mod]['mean']
            yerr = results[mod]['ci']
            if len(y) > 0:
                self.ax_mc.errorbar(snr_range[:len(y)], y, yerr=yerr, label=mod, marker='o')
        self.ax_mc.set_yscale('log')
        self.ax_mc.set_xlabel("SNR (dB)")
        self.ax_mc.set_ylabel("BER")
        self.ax_mc.set_title("Curvas BER vs SNR (95% CI)")
        self.ax_mc.grid(True)
        self.ax_mc.legend()
        self.fig_mc.canvas.draw()
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = OFDM_GUI(root)
    root.mainloop()