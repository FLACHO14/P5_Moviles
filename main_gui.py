import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import traceback
from PIL import Image, ImageTk

from ofdm_params import OFDMParams
from ofdm_utils import (image_to_bitstream, bitstream_to_image, calculate_ber,
                        calculate_psnr, bits_to_symbols)
from ofdm_tx import OFDMTransmitter
from ofdm_rx import OFDMReceiver
from ofdm_channel import OFDMChannel

class OFDMSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM - Comunicaciones Digitales")
        self.root.geometry("1500x950")
        self.params = OFDMParams()
        self.original_bits = None
        self.image_shape = None
        self.image_path = None
        self.original_img_arr = None  # Guardar array de imagen redimensionada
        self.setup_ui()

    def setup_ui(self):
        left = ttk.Frame(self.root, width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(left, text="Parámetros OFDM", font=('Arial',12,'bold')).pack(pady=5)

        ttk.Button(left, text="Cargar Imagen", command=self.load_image).pack(pady=2)
        self.img_preview = ttk.Label(left)
        self.img_preview.pack()
        self.img_info = ttk.Label(left, text="Sin imagen")
        self.img_info.pack()

        ttk.Label(left, text="Modulación:").pack()
        self.mod_combo = ttk.Combobox(left, values=['QPSK','16QAM','64QAM'], state='readonly')
        self.mod_combo.set('16QAM')
        self.mod_combo.pack()
        self.mod_combo.bind('<<ComboboxSelected>>', lambda e: self.update_bits_info())

        ttk.Label(left, text="Ancho de banda (MHz):").pack()
        self.bw_entry = ttk.Entry(left)
        self.bw_entry.insert(0, "5")
        self.bw_entry.pack()

        ttk.Label(left, text="Espaciado subportadoras (kHz):").pack()
        self.df_entry = ttk.Entry(left)
        self.df_entry.insert(0, "15")
        self.df_entry.pack()

        ttk.Label(left, text="Tamaño IFFT (0=auto):").pack()
        self.fft_entry = ttk.Entry(left)
        self.fft_entry.insert(0, "0")
        self.fft_entry.pack()

        ttk.Label(left, text="Prefijo cíclico:").pack()
        self.cp_combo = ttk.Combobox(left, values=['normal','extended'], state='readonly')
        self.cp_combo.set('normal')
        self.cp_combo.pack()

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=5)

        ttk.Label(left, text="Canal LoS/NLoS:").pack()
        self.los_combo = ttk.Combobox(left, values=['LoS','NLoS'], state='readonly')
        self.los_combo.set('LoS')
        self.los_combo.pack()
        ttk.Label(left, text="Número de multicopias:").pack()
        self.multipath_entry = ttk.Entry(left)
        self.multipath_entry.insert(0, "3")
        self.multipath_entry.pack()
        ttk.Label(left, text="Velocidad (km/h):").pack()
        self.speed_entry = ttk.Entry(left)
        self.speed_entry.insert(0, "30")
        self.speed_entry.pack()
        ttk.Label(left, text="SNR (dB):").pack()
        self.snr_entry = ttk.Entry(left)
        self.snr_entry.insert(0, "20")
        self.snr_entry.pack()

        self.btn_single = ttk.Button(left, text="Transmisión única", command=self.run_single)
        self.btn_single.pack(pady=5, fill='x')
        self.btn_mc = ttk.Button(left, text="Monte Carlo (BER vs SNR)", command=self.run_monte_carlo)
        self.btn_mc.pack(pady=5, fill='x')

        self.progress = ttk.Progressbar(left, mode='indeterminate', length=300)
        self.progress.pack(pady=5)
        self.lbl_status = ttk.Label(left, text="")
        self.lbl_status.pack()

        self.bits_label = ttk.Label(left, text="Bits originales: --")
        self.bits_label.pack()
        self.symbols_label = ttk.Label(left, text="Símbolos modulados: --")
        self.symbols_label.pack()

        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.create_tabs()

    def create_tabs(self):
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. Imagen y Constelaciones")
        fig1 = Figure(figsize=(8,6))
        self.ax_tx_img = fig1.add_subplot(221)
        self.ax_tx_const = fig1.add_subplot(222)
        self.ax_rx_img = fig1.add_subplot(223)
        self.ax_rx_const = fig1.add_subplot(224)
        self.ax_tx_img.set_title("Imagen TX")
        self.ax_tx_const.set_title("Constelación TX")
        self.ax_rx_img.set_title("Imagen RX")
        self.ax_rx_const.set_title("Constelación RX (ecualizada)")
        self.canvas1 = FigureCanvasTkAgg(fig1, self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar1 = NavigationToolbar2Tk(self.canvas1, self.tab1)
        toolbar1.update()
        self.fig1 = fig1

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="2. Subportadoras")
        fig2 = Figure(figsize=(8,5))
        self.ax_subc = fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(fig2, self.tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar2 = NavigationToolbar2Tk(self.canvas2, self.tab2)
        toolbar2.update()
        self.fig2 = fig2

        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="3. Respuesta del Canal")
        fig3 = Figure(figsize=(8,5))
        self.ax_ch_time = fig3.add_subplot(121)
        self.ax_ch_freq = fig3.add_subplot(122)
        self.canvas3 = FigureCanvasTkAgg(fig3, self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar3 = NavigationToolbar2Tk(self.canvas3, self.tab3)
        toolbar3.update()
        self.fig3 = fig3
        self.ch_class_label = ttk.Label(self.tab3, text="")
        self.ch_class_label.pack()

        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="4. Prefijo Cíclico y PAPR")
        fig4 = Figure(figsize=(8,6))
        self.ax_cp = fig4.add_subplot(221)
        self.ax_papr_time = fig4.add_subplot(222)
        self.ax_ccdf = fig4.add_subplot(223)
        self.ax_papr_time.set_title("Potencia instantánea")
        self.ax_ccdf.set_title("CCDF del PAPR")
        self.ax_ccdf.set_yscale('log')
        self.canvas4 = FigureCanvasTkAgg(fig4, self.tab4)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar4 = NavigationToolbar2Tk(self.canvas4, self.tab4)
        toolbar4.update()
        self.fig4 = fig4

        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="5. Monte Carlo")
        fig5 = Figure(figsize=(8,5))
        self.ax_mc = fig5.add_subplot(111)
        self.ax_mc.set_xlabel("SNR (dB)")
        self.ax_mc.set_ylabel("BER")
        self.ax_mc.set_yscale('log')
        self.ax_mc.grid(True)
        self.canvas5 = FigureCanvasTkAgg(fig5, self.tab5)
        self.canvas5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar5 = NavigationToolbar2Tk(self.canvas5, self.tab5)
        toolbar5.update()
        self.fig5 = fig5

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.image_path = path
            self.original_bits, self.image_shape, self.original_img_arr = image_to_bitstream(path, max_side=256)
            img = Image.open(path)
            img.thumbnail((120,120))
            self.img_tk = ImageTk.PhotoImage(img)
            self.img_preview.config(image=self.img_tk)
            self.img_info.config(text=f"{self.image_shape[1]}x{self.image_shape[0]}\nBits: {len(self.original_bits)}")
            self.update_bits_info()
            # Mostrar imagen TX en la pestaña 1
            self.ax_tx_img.clear()
            self.ax_tx_img.imshow(self.original_img_arr, cmap='gray')
            self.ax_tx_img.set_title("Imagen TX")
            self.ax_tx_img.axis('off')
            self.canvas1.draw()

    def update_bits_info(self):
        if self.original_bits is None:
            return
        mod = self.mod_combo.get()
        bits_per_sym = {'QPSK':2, '16QAM':4, '64QAM':6}[mod]
        num_sym = (len(self.original_bits) + bits_per_sym - 1) // bits_per_sym
        self.bits_label.config(text=f"Bits originales: {len(self.original_bits)}")
        self.symbols_label.config(text=f"Símbolos modulados: {num_sym}")

    def update_params(self):
        self.params.bandwidth_mhz = float(self.bw_entry.get())
        self.params.subcarrier_spacing = float(self.df_entry.get()) * 1000
        self.params.cp_type = self.cp_combo.get()
        self.params.modulation = self.mod_combo.get()
        self.params.channel_los = (self.los_combo.get() == 'LoS')
        self.params.num_multipath = int(self.multipath_entry.get())
        self.params.doppler_speed_kmh = float(self.speed_entry.get())
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
            self.root.update()
        else:
            self.progress.stop()
            self.lbl_status.config(text="")

    def run_single(self):
        if self.original_bits is None:
            messagebox.showerror("Error", "Cargue una imagen primero")
            return
        self.set_busy(True)
        self.update_params()
        threading.Thread(target=self._single_thread, daemon=True).start()

    def _single_thread(self):
        try:
            mod = self.params.modulation
            M = {'QPSK':4, '16QAM':16, '64QAM':64}[mod]
            symbols_tx = bits_to_symbols(self.original_bits, M)
            self.root.after(0, lambda: self.update_tx_constellation(symbols_tx))
            tx = OFDMTransmitter(self.params)
            tx_signal = tx.create_ofdm_symbols(symbols_tx)
            channel = OFDMChannel(self.params)
            rx_signal, taps, delays, (freq, H_f), ch_class = channel.apply_channel(tx_signal, self.params.snr_db)
            rx = OFDMReceiver(tx, self.params)
            bits_rx, raw_sym, eq_sym, _ = rx.process(rx_signal)
            ber = calculate_ber(self.original_bits, bits_rx)
            rec_img_pil = bitstream_to_image(bits_rx, self.image_shape)
            rec_img_arr = np.array(rec_img_pil)
            # Calcular PSNR usando el array original redimensionado
            psnr = calculate_psnr(self.original_img_arr, rec_img_arr)
            self.root.after(0, lambda: self.update_plots(tx, raw_sym, eq_sym, taps, delays, freq, H_f, ch_class, rec_img_arr, ber, psnr))
            self.root.after(0, lambda: messagebox.showinfo("Éxito", f"BER = {ber:.6f}\nPSNR = {psnr:.2f} dB"))
        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"{str(e)}\n\n{error_msg}"))
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def update_tx_constellation(self, symbols):
        self.ax_tx_const.clear()
        lim = min(5000, len(symbols))
        self.ax_tx_const.scatter(symbols.real[:lim], symbols.imag[:lim], s=1, alpha=0.5)
        self.ax_tx_const.set_title(f"Constelación TX - {self.params.modulation}")
        self.ax_tx_const.grid(True)
        self.canvas1.draw()

    def update_plots(self, tx, raw_sym, eq_sym, taps, delays, freq, H_f, ch_class, rec_img_arr, ber, psnr):
        self.ax_rx_img.clear()
        self.ax_rx_img.imshow(rec_img_arr, cmap='gray')
        self.ax_rx_img.set_title(f"Imagen RX - BER={ber:.4f} PSNR={psnr:.1f}dB")
        self.ax_rx_img.axis('off')
        self.ax_rx_const.clear()
        lim = min(5000, len(eq_sym))
        self.ax_rx_const.scatter(eq_sym.real[:lim], eq_sym.imag[:lim], s=1, alpha=0.5, c='red')
        self.ax_rx_const.set_title("Constelación RX ecualizada")
        self.ax_rx_const.grid(True)
        self.canvas1.draw()

        self.ax_subc.clear()
        subc = tx.subcarrier_grid[0]
        indices = np.arange(len(subc))
        mask = np.abs(subc) > 0
        self.ax_subc.bar(indices[mask], np.abs(subc[mask]), width=1.0, color='blue')
        self.ax_subc.set_title(f"Subportadoras (FFT={tx.fft_size}, útiles={tx.params.N_useful_subcarriers}, pilotos={tx.num_pilots})")
        self.ax_subc.set_xlim(0, tx.fft_size-1)
        self.canvas2.draw()

        self.ax_ch_time.clear()
        self.ax_ch_time.stem(delays, np.abs(taps), basefmt=' ')
        self.ax_ch_time.set_title("Respuesta impulsional")
        self.ax_ch_time.set_xlabel("Retardo (muestras)")
        self.ax_ch_time.set_ylabel("|h|")
        self.ax_ch_freq.clear()
        self.ax_ch_freq.plot(freq/1e3, 20*np.log10(np.abs(H_f)+1e-6))
        self.ax_ch_freq.set_title("Respuesta en frecuencia")
        self.ax_ch_freq.set_xlabel("Frecuencia (kHz)")
        self.ax_ch_freq.set_ylabel("|H| (dB)")
        self.canvas3.draw()
        self.ch_class_label.config(text=f"Clasificación del canal: {ch_class}")

        self.ax_cp.clear()
        sym_len = tx.fft_size + tx.cp_len
        y_cp = np.abs(tx.tx_signal[:sym_len])
        self.ax_cp.plot(y_cp)
        self.ax_cp.axvline(x=tx.cp_len, color='r', linestyle='--', label='Fin CP')
        self.ax_cp.fill_between(range(tx.cp_len), y_cp[:tx.cp_len], alpha=0.3, label='CP')
        self.ax_cp.set_title(f"Prefijo cíclico (longitud={tx.cp_len})")
        self.ax_cp.legend()
        power = np.abs(tx.tx_signal)**2
        avg_power = np.mean(power)
        papr = 10*np.log10(np.max(power)/avg_power)
        self.ax_papr_time.clear()
        self.ax_papr_time.plot(power, alpha=0.7)
        self.ax_papr_time.axhline(y=avg_power, color='r', linestyle='--', label=f'Prom={avg_power:.2f}')
        self.ax_papr_time.set_title(f"PAPR = {papr:.2f} dB")
        self.ax_papr_time.legend()
        papr_vals = 10*np.log10(power/avg_power + 1e-12)
        hist, bins = np.histogram(papr_vals, bins=50, density=True)
        ccdf = 1 - np.cumsum(hist) * (bins[1]-bins[0])
        self.ax_ccdf.clear()
        self.ax_ccdf.plot(bins[:-1], ccdf, 'b-')
        self.ax_ccdf.set_xlabel("PAPR (dB)")
        self.ax_ccdf.set_ylabel("Prob(PAPR > x)")
        self.ax_ccdf.grid(True)
        self.canvas4.draw()

    def run_monte_carlo(self):
        if self.original_bits is None:
            messagebox.showerror("Error", "Cargue una imagen primero")
            return
        self.set_busy(True)
        self.update_params()
        snr_range = np.arange(0, 21, 2)
        modulations = ['QPSK', '16QAM', '64QAM']
        results = {mod: {'mean': [], 'ci': []} for mod in modulations}
        threading.Thread(target=self._mc_thread, args=(snr_range, modulations, results), daemon=True).start()

    def _mc_thread(self, snr_range, modulations, results):
        try:
            for mod in modulations:
                self.params.modulation = mod
                M = {'QPSK':4, '16QAM':16, '64QAM':64}[mod]
                bits_per_sym = int(np.log2(float(M)))
                total_bits = len(self.original_bits)
                if total_bits % bits_per_sym != 0:
                    bits_pad = np.pad(self.original_bits, (0, bits_per_sym - total_bits % bits_per_sym), constant_values=0)
                else:
                    bits_pad = self.original_bits
                for snr in snr_range:
                    bers = []
                    for _ in range(10):
                        symbols_tx = bits_to_symbols(bits_pad, M)
                        tx = OFDMTransmitter(self.params)
                        tx_signal = tx.create_ofdm_symbols(symbols_tx)
                        channel = OFDMChannel(self.params)
                        rx_signal, _, _, _, _ = channel.apply_channel(tx_signal, snr)
                        rx = OFDMReceiver(tx, self.params)
                        bits_rx, _, _, _ = rx.process(rx_signal)
                        bits_rx = bits_rx[:total_bits]
                        ber = calculate_ber(self.original_bits, bits_rx)
                        bers.append(ber)
                    mean_ber = np.mean(bers)
                    std_ber = np.std(bers)
                    ci = 1.96 * std_ber / np.sqrt(10)
                    results[mod]['mean'].append(mean_ber)
                    results[mod]['ci'].append(ci)
                    self.root.after(0, lambda m=mod, snr_idx=len(results[mod]['mean'])-1: self.update_mc_plot(snr_range, modulations, results, m, snr_idx))
            self.root.after(0, lambda: messagebox.showinfo("Monte Carlo", "Simulación completada"))
        except Exception as e:
            error_msg = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", f"{str(e)}\n\n{error_msg}"))
        finally:
            self.root.after(0, lambda: self.set_busy(False))

    def update_mc_plot(self, snr_range, modulations, results, current_mod, idx):
        self.ax_mc.clear()
        for mod in modulations:
            y = results[mod]['mean']
            yerr = results[mod]['ci']
            if len(y) > 0:
                self.ax_mc.errorbar(snr_range[:len(y)], y, yerr=yerr, label=mod, marker='o')
        self.ax_mc.set_xlabel("SNR (dB)")
        self.ax_mc.set_ylabel("BER")
        self.ax_mc.set_yscale('log')
        self.ax_mc.set_title("Curvas BER vs SNR (95% de confianza)")
        self.ax_mc.grid(True)
        self.ax_mc.legend()
        self.canvas5.draw()
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = OFDMSimulator(root)
    root.mainloop()