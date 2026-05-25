# main_gui.py
# Interfaz gráfica del Simulador OFDM 4G LTE.
# Orquesta la simulación completa: carga de imagen → TX → canal → RX → análisis.
#
# Cambios respecto a la versión anterior:
# - Panel lateral renombrado a "Configuración de Parámetros".
# - Slider de SNR reemplazado por lista editable: el usuario añade múltiples valores
#   y el mayor se usa para la transmisión de imagen; todos los valores se usan para
#   calcular la curva BER vs SNR.
# - Nuevos campos: Taps del Canal (L) e Iteraciones Monte Carlo.
# - Tab 1: etiqueta con conteo de bits/símbolos y diagramas de constelación TX/RX.
# - Tab 3: gráfica renombrada a "Respuesta en Frecuencia del Canal".
# - Tab 4: renombrada a "Analisis" con CCDF del PAPR y curva BER vs SNR.
# - Cadena RX corregida: anteriormente se simulaba el error con XOR aleatorio
#   (lo que producía ~60% BER sin importar el SNR); ahora se usa la cadena real
#   ofdm_rx_block → equalize → qam_demod.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

import ofdm_tx
import ofdm_channel
import ofdm_rx
import ofdm_utils
import ofdm_params

MOD_MAP = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
MOD_COLORS = {"QPSK": "#3498db", "16QAM": "#2ecc71", "64QAM": "#e74c3c"}
VEL_MAP = {
    "Estático (0)": 0,
    "Pedestre (3 km/h)": 3,
    "Urbano (50 km/h)": 50,
    "Autopista (120 km/h)": 120,
}


class OFDM_Simulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM 4G LTE - Ingeniería")
        self.root.geometry("1400x900")

        self.img_path = tk.StringVar()
        self.mod_var = tk.StringVar(value="16QAM")
        self.chan_var = tk.StringVar(value="Rayleigh")
        self.vel_var = tk.StringVar(value="Pedestre (3 km/h)")
        self.cp_var = tk.StringVar(value="Normal")
        self.bw_var = tk.DoubleVar(value=10.0)
        self.snr_entry_var = tk.StringVar(value="20")
        self.mc_var = tk.StringVar(value="5")
        self.taps_var = tk.StringVar(value="8")

        # Lista de valores SNR precargada; el máximo se usa para la simulación de imagen
        self.snr_list = [0, 5, 10, 15, 20]
        self.sim_data = {}

        self.setup_ui()
        self._refresh_snr_listbox()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def setup_ui(self):
        ctrl_frame = ttk.LabelFrame(self.root, text=" Configuración de Parámetros ")
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)

        tk.Button(
            ctrl_frame,
            text="CARGAR IMAGEN",
            command=self.load_image,
            bg="#34495e",
            fg="black",
            font=("Helvetica", 10, "bold"),
        ).pack(pady=10, fill="x")

        ttk.Label(ctrl_frame, text="Ancho de Banda (MHz):").pack()
        ttk.Entry(ctrl_frame, textvariable=self.bw_var).pack(pady=3, fill="x", padx=5)

        ttk.Label(ctrl_frame, text="Esquema de Modulación:").pack()
        ttk.Combobox(
            ctrl_frame, textvariable=self.mod_var, values=["QPSK", "16QAM", "64QAM"]
        ).pack(pady=3, fill="x", padx=5)

        ttk.Label(ctrl_frame, text="Modelo de Canal:").pack()
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.chan_var,
            values=["Ideal", "Rayleigh (NLoS)", "Rician (LoS)"],
        ).pack(pady=3, fill="x", padx=5)

        ttk.Label(ctrl_frame, text="Ambiente / Velocidad:").pack()
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.vel_var,
            values=list(VEL_MAP.keys()),
        ).pack(pady=3, fill="x", padx=5)

        ttk.Label(ctrl_frame, text="Tipo de Prefijo Cíclico:").pack()
        ttk.Combobox(
            ctrl_frame, textvariable=self.cp_var, values=["Normal", "Extendido"]
        ).pack(pady=3, fill="x", padx=5)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", padx=5, pady=6)

        # --- Lista de valores SNR ---
        # El usuario añade cada valor que desee; todos se usan en la curva BER.
        # El valor máximo de la lista se aplica a la transmisión de imagen.
        ttk.Label(
            ctrl_frame, text="Valores SNR (dB):", font=("Helvetica", 9, "bold")
        ).pack()
        snr_add_frame = ttk.Frame(ctrl_frame)
        snr_add_frame.pack(fill="x", padx=5, pady=2)
        ttk.Entry(snr_add_frame, textvariable=self.snr_entry_var, width=8).pack(
            side="left", padx=(0, 4)
        )
        tk.Button(
            snr_add_frame,
            text="Añadir",
            command=self.add_snr,
            bg="#2980b9",
            fg="black",
            font=("Helvetica", 8, "bold"),
        ).pack(side="left")

        self.snr_listbox = tk.Listbox(
            ctrl_frame, height=5, width=30, font=("Consolas", 9)
        )
        self.snr_listbox.pack(pady=2, padx=5, fill="x")

        snr_btn_frame = ttk.Frame(ctrl_frame)
        snr_btn_frame.pack(fill="x", padx=5)
        tk.Button(
            snr_btn_frame,
            text="Eliminar",
            command=self.remove_snr,
            bg="#c0392b",
            fg="black",
            font=("Helvetica", 8, "bold"),
        ).pack(side="left", padx=(0, 3))
        tk.Button(
            snr_btn_frame,
            text="Limpiar",
            command=self.clear_snr,
            bg="#7f8c8d",
            fg="black",
            font=("Helvetica", 8, "bold"),
        ).pack(side="left")

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", padx=5, pady=6)

        # --- Parámetros de análisis Monte Carlo ---
        ttk.Label(ctrl_frame, text="Taps del Canal (L):").pack()
        ttk.Entry(ctrl_frame, textvariable=self.taps_var).pack(
            pady=3, fill="x", padx=5
        )

        ttk.Label(ctrl_frame, text="Iteraciones Monte Carlo:").pack()
        ttk.Entry(ctrl_frame, textvariable=self.mc_var).pack(pady=3, fill="x", padx=5)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", padx=5, pady=6)

        tk.Button(
            ctrl_frame,
            text="EJECUTAR TRANSMISIÓN",
            command=self.run_simulation,
            bg="#27ae60",
            fg="black",
            font=("Helvetica", 12, "bold"),
            height=2,
        ).pack(pady=10, fill="x")

        ttk.Label(ctrl_frame, text="Datos del Sistema:", font=("Helvetica", 9, "bold")).pack()
        self.txt_report = tk.Text(
            ctrl_frame, width=32, height=14, font=("Consolas", 8), bg="#121313"
        )
        self.txt_report.pack(pady=4, padx=4)

        # --- Pestañas de resultados ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Comparativa de Imagen")
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Análisis de Subportadoras")
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Canal y Prefijo")
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Analisis")

    # ------------------------------------------------------------------
    # SNR list helpers
    # ------------------------------------------------------------------

    def add_snr(self):
        """Agrega un valor SNR a la lista (sin duplicados, mantiene orden ascendente)."""
        try:
            val = float(self.snr_entry_var.get())
            if val not in self.snr_list:
                self.snr_list.append(val)
                self.snr_list.sort()
                self._refresh_snr_listbox()
        except ValueError:
            pass

    def remove_snr(self):
        sel = self.snr_listbox.curselection()
        if sel:
            self.snr_list.pop(sel[0])
            self._refresh_snr_listbox()

    def clear_snr(self):
        self.snr_list.clear()
        self._refresh_snr_listbox()

    def _refresh_snr_listbox(self):
        self.snr_listbox.delete(0, tk.END)
        for v in self.snr_list:
            self.snr_listbox.insert(tk.END, f"  {v:g} dB")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            self.img_path.set(path)

    def _psnr(self, original, decoded):
        """PSNR simplificado (Peak Signal-to-Noise Ratio) para evaluar calidad de imagen."""
        mse = np.mean((original.astype(float) - decoded.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def _clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()

    def _status(self, msg):
        """Actualiza el cuadro de reporte y fuerza el redibujado de la ventana."""
        self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, msg)
        self.root.update()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_simulation(self):
        """Ejecuta la cadena completa TX → Canal → RX → Análisis → Gráficas.

        Flujo:
        1. Carga imagen y la serializa en bits.
        2. Modula con QAM y genera señal OFDM (IFFT + CP).
        3. Aplica el canal seleccionado al SNR máximo de la lista.
        4. Demodula: elimina CP → FFT → ecualización → QAM demod.
        5. Reconstruye la imagen y calcula PSNR.
        6. Ejecuta análisis Monte Carlo (BER + CCDF PAPR) para todos los SNR.
        7. Renderiza las cuatro pestañas.
        """
        if not self.img_path.get():
            messagebox.showwarning("Imagen", "Cargue una imagen primero.")
            return
        if not self.snr_list:
            messagebox.showwarning("SNR", "Añada al menos un valor de SNR.")
            return

        try:
            M = MOD_MAP[self.mod_var.get()]
            Nfft, cp_len = ofdm_utils.get_nfft_cp(self.bw_var.get(), self.cp_var.get())
            velocity = VEL_MAP[self.vel_var.get()]
            chan_type = self.chan_var.get()
            taps_L = max(1, int(self.taps_var.get()))
            n_mc = max(1, int(self.mc_var.get()))
            # El SNR máximo de la lista se usa para la simulación de imagen
            snr_sim = max(self.snr_list)

            # --- Carga y serialización de imagen ---
            img = Image.open(self.img_path.get()).convert("L").resize((128, 128))
            img_arr = np.array(img)
            bits_tx = np.unpackbits(img_arr)  # 128×128×8 = 131 072 bits

            stats = ofdm_utils.calculate_resource_stats(bits_tx, Nfft, M)

            # Zero-padding para completar el último bloque OFDM
            k = int(np.log2(M))
            pad = (-len(bits_tx)) % (Nfft * k)
            bits_in = (
                np.concatenate([bits_tx, np.zeros(pad, np.uint8)])
                if pad
                else bits_tx.copy()
            )

            # --- TX ---
            self._status("Modulando y transmitiendo...")
            symbols_tx = ofdm_tx.qam_mod(bits_in, M)
            tx_signal, papr_list = ofdm_tx.ofdm_tx_block(symbols_tx, Nfft, cp_len)

            # --- Canal ---
            if chan_type == "Rayleigh":
                h = ofdm_channel.multipath_rayleigh_channel(taps_L)
            elif chan_type == "Rician":
                h = ofdm_channel.multipath_rician_channel(taps_L)
            else:
                h = None  # Canal Ideal: solo AWGN

            rx_signal, h_used = ofdm_channel.apply_channel(
                tx_signal, chan_type, snr_sim, h, velocity_kmh=velocity
            )

            # --- RX (cadena real, reemplaza la simulación de error anterior) ---
            Y = ofdm_rx.ofdm_rx_block(rx_signal, Nfft, cp_len)
            H_freq = np.fft.fft(h_used, n=Nfft)
            n_ofdm_rx = len(Y) // Nfft
            # Se asume canal constante entre símbolos OFDM (bloque fading)
            Xhat = ofdm_rx.equalize(Y, np.tile(H_freq, n_ofdm_rx))
            bits_rx = ofdm_rx.qam_demod(Xhat, M)[: len(bits_in)]

            img_rx_arr = np.packbits(bits_rx[: len(bits_tx)]).reshape(img_arr.shape)
            psnr = self._psnr(img_arr, img_rx_arr)

            # --- Análisis BER / CCDF del PAPR ---
            self._status("Calculando BER y CCDF del PAPR...\n(puede tardar unos segundos)")
            ber_data, ccdf_data = ofdm_utils.run_analysis(
                bits_tx, Nfft, cp_len, chan_type, taps_L, self.snr_list, n_mc
            )

            # --- Reporte de parámetros ---
            mod_name = self.mod_var.get()
            report = (
                f"--- PARÁMETROS ---\n"
                f"Nfft: {Nfft}  |  CP: {cp_len}\n"
                f"Modulación: {mod_name}\n"
                f"Canal: {chan_type}\n"
                f"Taps (L): {taps_L}\n"
                f"SNR simulación: {snr_sim:g} dB\n"
                f"SNR valores: {self.snr_list}\n"
                f"MC iter.: {n_mc}\n\n"
                f"--- IMAGEN ---\n"
                f"Bits TX: {len(bits_tx)}\n"
                f"Símbolos {mod_name}: {len(symbols_tx)}\n"
                f"OFDM símbolos: {stats['total_ofdm_symbols']}\n"
                f"Padding: {stats['padding_zeros']} bits\n"
                f"PAPR máx: {np.max(papr_list):.2f} dB\n"
                f"PSNR imagen: {psnr:.2f} dB\n"
            )
            self._status(report)

            # Almacenar datos para las gráficas
            self.sim_data = {
                "img_tx": img_arr,
                "img_rx": img_rx_arr,
                "symbols_tx": symbols_tx,
                "symbols_rx": Xhat,
                "h_used": h_used,
                "Nfft": Nfft,
                "cp_len": cp_len,
                "papr_list": papr_list,
                "ber_data": ber_data,
                "ccdf_data": ccdf_data,
                "M": M,
                "mod_name": mod_name,
                "snr_list": self.snr_list.copy(),
                "bits_tx": bits_tx,
            }
            self.render_plots()

        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            raise

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def render_plots(self):
        d = self.sim_data
        if not d:
            return
        self._render_tab1(d)
        self._render_tab2(d)
        self._render_tab3(d)
        self._render_tab4(d)

    def _embed(self, fig, tab):
        """Inserta una figura matplotlib dentro de una pestaña Tkinter y la cierra
        del contexto de matplotlib para liberar memoria.
        """
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)  # libera la figura de matplotlib; el widget tkinter persiste

    # Tab 1 — Comparativa de imagen y diagramas de constelación TX/RX
    def _render_tab1(self, d):
        self._clear_tab(self.tab1)

        mod_name = d["mod_name"]
        n_bits = len(d["bits_tx"])
        n_syms = len(d["symbols_tx"])

        # Etiqueta informativa: bits de imagen y símbolos QAM resultantes
        tk.Label(
            self.tab1,
            text=f"  Bits de imagen: {n_bits}   |   Símbolos {mod_name}: {n_syms}  ",
            font=("Consolas", 11, "bold"),
            bg="#0b314d",
            relief="groove",
            padx=10,
            pady=5,
        ).pack(fill="x", padx=6, pady=(5, 0))

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.tight_layout(pad=3.0)

        # Fila 0: imágenes TX y RX
        axes[0, 0].imshow(d["img_tx"], cmap="gray")
        axes[0, 0].set_title("TX: Antes de transmitir")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(d["img_rx"], cmap="gray")
        axes[0, 1].set_title("RX: Después del Canal")
        axes[0, 1].axis("off")

        # Fila 1: constelaciones (máx 2000 puntos para rendimiento)
        npts = min(2000, len(d["symbols_tx"]))
        axes[1, 0].scatter(
            np.real(d["symbols_tx"][:npts]),
            np.imag(d["symbols_tx"][:npts]),
            s=5,
            c="#2ecc71",
            alpha=0.5,
        )
        axes[1, 0].set_title(f"Constelación TX ({mod_name})")
        axes[1, 0].set_xlabel("I")
        axes[1, 0].set_ylabel("Q")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect("equal")

        npts_rx = min(2000, len(d["symbols_rx"]))
        axes[1, 1].scatter(
            np.real(d["symbols_rx"][:npts_rx]),
            np.imag(d["symbols_rx"][:npts_rx]),
            s=5,
            c="#e74c3c",
            alpha=0.5,
        )
        axes[1, 1].set_title("Constelación RX Ecualizada")
        axes[1, 1].set_xlabel("I")
        axes[1, 1].set_ylabel("Q")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect("equal")

        self._embed(fig, self.tab1)

    # Tab 2 — Ortogonalidad de subportadoras con Delta_f = 15 kHz
    def _render_tab2(self, d):
        self._clear_tab(self.tab2)
        fig, ax = plt.subplots(figsize=(8, 4))
        ofdm_utils.plot_subcarrier_spacing(ax)
        self._embed(fig, self.tab2)

    # Tab 3 — Respuesta en frecuencia del canal y comparativa de CP
    def _render_tab3(self, d):
        self._clear_tab(self.tab3)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.tight_layout(pad=3.0)

        # Respuesta en frecuencia del canal: FFT de la respuesta al impulso h
        H_f = np.fft.fft(d["h_used"], n=d["Nfft"])
        axes[0].plot(np.abs(H_f), color="orange")
        axes[0].set_title("Respuesta en Frecuencia del Canal")
        axes[0].set_xlabel("Subportadoras")
        axes[0].set_ylabel("|H[k]|")
        axes[0].grid(True, alpha=0.3)

        # CP dinámico: valores calculados según el Nfft actual de la simulación
        Nfft = d["Nfft"]
        ofdm_utils.plot_cp_comparison(axes[1], Nfft // 8, Nfft // 4)

        self._embed(fig, self.tab3)

    # Tab 4 — CCDF del PAPR y curva BER vs SNR
    def _render_tab4(self, d):
        self._clear_tab(self.tab4)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout(pad=3.5)

        # CCDF del PAPR: eje Y en escala log para visualizar la cola de distribución
        for mod_name, (papr_sorted, ccdf) in d["ccdf_data"].items():
            axes[0].semilogy(
                papr_sorted, ccdf, label=mod_name, color=MOD_COLORS[mod_name]
            )
        axes[0].set_title("CCDF del PAPR (OFDM)")
        axes[0].set_xlabel("PAPR (dB)")
        axes[0].set_ylabel("Prob{PAPR > x}")
        axes[0].legend()
        axes[0].grid(True, which="both", alpha=0.4)

        # BER vs SNR: solo se grafican puntos con BER > 0 (semilogy no admite 0)
        snr_arr = d["snr_list"]
        any_plotted = False
        for mod_name, ber_vals in d["ber_data"].items():
            pairs = [(s, b) for s, b in zip(snr_arr, ber_vals) if b > 0]
            if pairs:
                snrs, bers = zip(*pairs)
                axes[1].semilogy(
                    snrs,
                    bers,
                    marker="o",
                    label=mod_name,
                    color=MOD_COLORS[mod_name],
                )
                any_plotted = True
        if not any_plotted:
            axes[1].text(
                0.5, 0.5,
                "BER = 0 en todos los puntos\n(SNR muy alto)",
                ha="center", va="center", transform=axes[1].transAxes,
            )
        axes[1].set_title("Desempeño BER vs SNR")
        axes[1].set_xlabel("SNR (dB)")
        axes[1].set_ylabel("BER")
        axes[1].legend()
        axes[1].grid(True, which="both", alpha=0.4)

        self._embed(fig, self.tab4)


if __name__ == "__main__":
    root = tk.Tk()
    app = OFDM_Simulator(root)
    root.mainloop()

