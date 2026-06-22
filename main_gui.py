# main_gui.py
# Interfaz gráfica del Simulador OFDM 4G LTE.
#
# Cambios principales respecto a la versión V2:
# - Cadena TX/RX con pilotos: insert_pilots en TX, equalize_with_pilots en RX.
# - Banda de guarda del 10%: solo el 90% del BW se usa (build_subcarrier_map).
# - Tab 1: muestra solo la modulación seleccionada (no las tres a la vez).
# - Tab 2: mapa de subportadoras (datos/pilotos/guarda) + ortogonalidad + PSD.
# - Tab 3: respuesta del canal + clasificación (selectivo/plano, rápido/lento).
# - Tab 5: BER con IC 95%, comparación con/sin pilotos, CCDF PAPR,
#           potencia instantánea vs promedio (PAPR en tiempo).
# - Nuevos controles: Δf editable, espaciado de pilotos.
# - Panel de reporte con estadísticas del sistema.

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
CHAN_MODELS = [
    "Ideal",
    "Rayleigh (NLoS)",
    "Rician (LoS)",
    "Suburbano (EPA)",
    "Urbano (EVA)",
    "Rural (ETU)",
]


class OFDM_Simulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM 4G LTE")
        self.root.geometry("1600x1000")

        # Variables de control
        self.img_path = tk.StringVar()
        self.mod_var = tk.StringVar(value="16QAM")
        self.chan_profile = tk.StringVar(value="Rayleigh (NLoS)")
        self.vel_var = tk.StringVar(value="Urbano (50 km/h)")
        self.cp_var = tk.StringVar(value="Normal")
        self.bw_var = tk.StringVar(value="10.0")
        self.delta_f_var = tk.StringVar(value="15000")
        self.pilot_spacing_var = tk.StringVar(value="6")
        self.snr_entry_var = tk.StringVar(value="20")
        self.mc_var = tk.StringVar(value="10")
        self.taps_var = tk.StringVar(value="8")
        self.max_side_var = tk.StringVar(value="128")
        self.nr_ant_var = tk.StringVar(value="2")

        self.snr_list = [0, 5, 10, 15, 20]
        self.sim_data = {}

        self.setup_ui()
        self._refresh_snr_listbox()

    # ==============================================================
    # UI
    # ==============================================================

    def setup_ui(self):
        # --- Panel lateral de controles ---
        ctrl_frame = ttk.LabelFrame(self.root, text=" Configuración de Parámetros ")
        ctrl_frame.pack(side="left", fill="y", padx=8, pady=8)

        tk.Button(
            ctrl_frame, text="CARGAR IMAGEN", command=self.load_image,
            bg="#34495e", fg="white", font=("Helvetica", 10, "bold"),
        ).pack(pady=(8, 4), fill="x", padx=5)

        self._row(ctrl_frame, "Lado máx. (px):", self.max_side_var)
        self._row(ctrl_frame, "BW (MHz):", self.bw_var)
        self._row(ctrl_frame, "Δf (Hz):", self.delta_f_var)
        self._combo(ctrl_frame, "Modulación:", self.mod_var, list(MOD_MAP.keys()))
        self._combo(ctrl_frame, "Canal:", self.chan_profile, CHAN_MODELS)
        self._combo(ctrl_frame, "Velocidad:", self.vel_var, list(VEL_MAP.keys()))
        self._combo(ctrl_frame, "Prefijo Cíclico:", self.cp_var, ["Normal", "Extendido"])
        self._row(ctrl_frame, "Taps canal (L):", self.taps_var)
        self._row(ctrl_frame, "Espac. pilotos:", self.pilot_spacing_var)
        self._combo(ctrl_frame, "NR Antenas:", self.nr_ant_var, ["1", "2", "3", "4"])

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", padx=5, pady=4)

        # --- Lista de valores SNR ---
        ttk.Label(ctrl_frame, text="Valores SNR (dB):", font=("Helvetica", 9, "bold")).pack(
            padx=5, anchor="w"
        )
        snr_row = ttk.Frame(ctrl_frame)
        snr_row.pack(fill="x", padx=5, pady=2)
        ttk.Entry(snr_row, textvariable=self.snr_entry_var, width=8).pack(
            side="left", padx=(0, 4)
        )
        tk.Button(
            snr_row, text="Añadir", command=self.add_snr, bg="#2980b9", fg="white",
        ).pack(side="left")

        self.snr_listbox = tk.Listbox(ctrl_frame, height=4, font=("Consolas", 9))
        self.snr_listbox.pack(pady=2, padx=5, fill="x")

        btn_snr = ttk.Frame(ctrl_frame)
        btn_snr.pack(fill="x", padx=5)
        tk.Button(btn_snr, text="Eliminar", command=self.remove_snr, bg="#c0392b", fg="white").pack(
            side="left", padx=(0, 3)
        )
        tk.Button(btn_snr, text="Limpiar", command=self.clear_snr, bg="#7f8c8d", fg="white").pack(
            side="left"
        )

        self._row(ctrl_frame, "Iteraciones MC:", self.mc_var)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", padx=5, pady=4)

        tk.Button(
            ctrl_frame, text="EJECUTAR TRANSMISIÓN", command=self.run_simulation,
            bg="#27ae60", fg="white", font=("Helvetica", 11, "bold"), height=2,
        ).pack(pady=6, fill="x", padx=5)

        # Reporte
        ttk.Label(ctrl_frame, text="Datos del Sistema:", font=("Helvetica", 9, "bold")).pack(
            padx=5, anchor="w"
        )
        self.txt_report = tk.Text(
            ctrl_frame, width=32, height=14, font=("Consolas", 8),
            bg="#1a1a2e", fg="#e0e0e0",
        )
        self.txt_report.pack(pady=4, padx=5, fill="x")

        # --- Pestañas de resultados ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="right", expand=True, fill="both", padx=8, pady=8)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="1. Imagen y Constelaciones")
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="2. Subportadoras")
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="3. Canal")
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="4. Prefijo Cíclico")
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="5. BER y PAPR")
        self.tab6 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab6, text="6. OFDM vs SC-FDMA")
        self.tab7 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab7, text="7. Diversidad RX")
        self.tab8 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab8, text="8. Diversidad TX (MISO-SFBC)")
        self.tab9 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab9, text="9. Imagen SISO vs SFBC")

    # helpers para layout compacto
    def _row(self, parent, label, var):
        f = ttk.Frame(parent)
        f.pack(fill="x", padx=5, pady=1)
        ttk.Label(f, text=label).pack(side="left")
        ttk.Entry(f, textvariable=var, width=10).pack(side="right")

    def _combo(self, parent, label, var, values):
        f = ttk.Frame(parent)
        f.pack(fill="x", padx=5, pady=1)
        ttk.Label(f, text=label).pack(side="left")
        ttk.Combobox(f, textvariable=var, values=values, width=18).pack(side="right")

    # ==============================================================
    # Helpers
    # ==============================================================

    def add_snr(self):
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

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            self.img_path.set(path)

    def _psnr(self, orig, dec):
        mse = np.mean((orig.astype(float) - dec.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def _clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()

    def _status(self, msg):
        self.txt_report.delete(1.0, tk.END)
        self.txt_report.insert(tk.END, msg)
        self.root.update()

    def _embed(self, fig, tab):
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ==============================================================
    # Simulación
    # ==============================================================

    def run_simulation(self):
        """Cadena completa TX → Canal → RX → Análisis Monte Carlo → Gráficas.

        Flujo:
        1. Calcula parámetros OFDM (Nfft, CP, mapa de subportadoras).
        2. Carga imagen y la serializa en bits.
        3. TX: modulación QAM + inserción de pilotos + IFFT + CP.
        4. Canal: convolución con h + Doppler + AWGN.
        5. RX: elimina CP + FFT + ecualización por pilotos + demod QAM.
        6. Reconstruye la imagen y calcula BER / PSNR.
        7. Monte Carlo: BER con IC 95% para las 3 modulaciones.
        8. Renderiza las 5 pestañas.
        """
        if not self.img_path.get():
            messagebox.showwarning("Imagen", "Cargue una imagen primero.")
            return
        if not self.snr_list:
            messagebox.showwarning("SNR", "Añada al menos un valor de SNR.")
            return

        try:
            # -- Parámetros --
            bw = float(self.bw_var.get())
            cp_mode = self.cp_var.get()
            delta_f = max(1000, float(self.delta_f_var.get()))
            pilot_spacing = max(2, int(self.pilot_spacing_var.get()))

            Nfft, cp_len, N_used = ofdm_utils.get_nfft_cp(bw, cp_mode, delta_f)
            sc_map = ofdm_utils.build_subcarrier_map(Nfft, N_used, pilot_spacing)
            pilot_value = ofdm_params.PILOT_AMPLITUDE
            fs = Nfft * delta_f

            velocity = VEL_MAP[self.vel_var.get()]
            profile = self.chan_profile.get()
            taps_L = max(1, int(self.taps_var.get()))
            n_mc = max(1, int(self.mc_var.get()))
            snr_sim = max(self.snr_list)

            # -- Imagen --
            max_side = max(8, int(self.max_side_var.get()))
            img = Image.open(self.img_path.get()).convert("L")
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img_arr = np.array(img)
            bits_tx = np.unpackbits(img_arr.flatten())
            n_bits = len(bits_tx)

            # -- Recursos --
            mod_name = self.mod_var.get()
            M = MOD_MAP[mod_name]
            k = int(np.log2(M))
            n_data = sc_map["n_data"]
            bits_per_ofdm = n_data * k
            n_ofdm_needed = int(np.ceil(n_bits / bits_per_ofdm))
            total_bits_cap = n_ofdm_needed * bits_per_ofdm
            pad_bits = total_bits_cap - n_bits
            bits_in = (
                np.pad(bits_tx, (0, pad_bits)) if pad_bits else bits_tx.copy()
            )

            # -- TX --
            self._status("Modulando y transmitiendo...")
            symbols_tx = ofdm_tx.qam_mod(bits_in, M)
            tx_signal, papr_list, n_ofdm = ofdm_tx.ofdm_tx_block(
                symbols_tx, Nfft, cp_len, sc_map, pilot_value
            )

            # -- Canal --
            h_impulse = ofdm_channel.get_channel_profile(profile, taps_L)
            rx_signal, h_used = ofdm_channel.apply_channel(
                tx_signal, h_impulse, snr_sim, velocity, fs=fs
            )

            # -- RX con pilotos --
            Y_frames = ofdm_rx.ofdm_rx_block(rx_signal, Nfft, cp_len)
            Xhat_pilot, H_est_list = ofdm_rx.equalize_with_pilots(
                Y_frames, sc_map, pilot_value, Nfft
            )
            bits_rx = ofdm_rx.qam_demod(Xhat_pilot, M)

            # Ajustar longitud
            if len(bits_rx) < total_bits_cap:
                bits_rx = np.pad(bits_rx, (0, total_bits_cap - len(bits_rx)))
            else:
                bits_rx = bits_rx[:total_bits_cap]

            # Reconstruir imagen
            bits_img = bits_rx[:n_bits]
            img_bytes = np.packbits(bits_img)
            if len(img_bytes) > img_arr.size:
                img_bytes = img_bytes[: img_arr.size]
            elif len(img_bytes) < img_arr.size:
                img_bytes = np.pad(img_bytes, (0, img_arr.size - len(img_bytes)))
            img_rx = img_bytes.reshape(img_arr.shape)

            psnr = self._psnr(img_arr, img_rx)
            ber_img = np.mean(bits_img != bits_tx)

            # Muestras de constelación
            const_tx = symbols_tx[: min(2000, len(symbols_tx))]
            const_rx = Xhat_pilot[: min(2000, len(Xhat_pilot))]

            # -- Clasificación del canal --
            chan_class = ofdm_utils.classify_channel(
                h_used, fs, N_used, velocity, delta_f
            )

            # -- Monte Carlo BER + CCDF PAPR --
            self._status("Ejecutando Monte Carlo...\n(puede tardar)")
            ber_data, ber_no_eq, ccdf_data = ofdm_utils.run_analysis(
                bits_tx, Nfft, cp_len, sc_map, pilot_value,
                profile, taps_L, self.snr_list, n_mc, velocity,
            )

            # PAPR con IC
            papr_means, papr_ci = {}, {}
            mods_all = {"QPSK": 4, "16QAM": 16, "64QAM": 64}
            for mn, Mv in mods_all.items():
                means, cis = ofdm_utils.compute_papr_vs_snr(
                    Nfft, cp_len, Mv, sc_map, pilot_value, self.snr_list, n_mc=10,
                )
                papr_means[mn] = means
                papr_ci[mn] = cis

            # -- SC-FDMA --
            self._status("Ejecutando SC-FDMA...\n(comparación con OFDM)")
            M_dft = ofdm_tx.get_best_dft_size(sc_map["n_data"], Nfft)
            sc_map_sc = ofdm_utils.build_scfdma_map(sc_map, M_dft)

            # TX SC-FDMA (misma imagen, misma modulación)
            k_sc = int(np.log2(M))
            n_data_sc = M_dft
            bits_per_ofdm_sc = n_data_sc * k_sc
            n_ofdm_sc_needed = int(np.ceil(n_bits / bits_per_ofdm_sc))
            total_bits_sc = n_ofdm_sc_needed * bits_per_ofdm_sc
            pad_bits_sc = total_bits_sc - n_bits
            bits_in_sc = np.pad(bits_tx, (0, pad_bits_sc)) if pad_bits_sc else bits_tx.copy()

            symbols_tx_sc = ofdm_tx.qam_mod(bits_in_sc, M)
            tx_signal_sc, papr_list_sc, n_ofdm_sc = ofdm_tx.scfdma_tx_block(
                symbols_tx_sc, Nfft, cp_len, sc_map, pilot_value, M_dft
            )

            # Canal SC-FDMA (mismo canal)
            rx_signal_sc, _ = ofdm_channel.apply_channel(
                tx_signal_sc, h_impulse, snr_sim, velocity, fs=fs
            )

            # RX SC-FDMA
            Y_frames_sc = ofdm_rx.ofdm_rx_block(rx_signal_sc, Nfft, cp_len)
            Xhat_sc, _ = ofdm_rx.equalize_with_pilots(
                Y_frames_sc, sc_map_sc, pilot_value, Nfft
            )
            Xhat_sc = ofdm_rx.scfdma_despread(Xhat_sc, M_dft)
            bits_rx_sc = ofdm_rx.qam_demod(Xhat_sc, M)

            if len(bits_rx_sc) < total_bits_sc:
                bits_rx_sc = np.pad(bits_rx_sc, (0, total_bits_sc - len(bits_rx_sc)))
            else:
                bits_rx_sc = bits_rx_sc[:total_bits_sc]
            ber_img_sc = np.mean(bits_rx_sc[:n_bits] != bits_tx)

            # Monte Carlo SC-FDMA
            self._status("Monte Carlo SC-FDMA...\n(puede tardar)")
            ber_data_sc, ccdf_data_sc = ofdm_utils.run_analysis_scfdma(
                bits_tx, Nfft, cp_len, sc_map, pilot_value,
                profile, taps_L, self.snr_list, n_mc, velocity, M_dft,
            )

            # -- Diversidad RX --
            NR = max(1, int(self.nr_ant_var.get()))
            self._status("Monte Carlo Diversidad RX...\n(SISO vs MRC vs SC vs MMSE)")
            div_results = ofdm_utils.run_analysis_diversity(
                bits_tx, Nfft, cp_len, sc_map, pilot_value,
                profile, taps_L, self.snr_list, n_mc, velocity, NR,
                use_scfdma=True, M_dft=M_dft,
            )

            self._status("Generando datos de potencia diversidad...")
            H_single, H_combined = ofdm_utils.generate_diversity_power_data(
                Nfft, cp_len, sc_map, pilot_value, profile, taps_L,
                snr_sim, velocity, NR, M_mod=M,
                use_scfdma=True, M_dft=M_dft,
            )

            # -- Diversidad TX (MISO-SFBC) --
            self._status("Monte Carlo Diversidad TX...\n(SISO vs SIMO-MRC vs MISO-SFBC)")
            txdiv_results = ofdm_utils.run_analysis_diversity_tx(
                bits_tx, Nfft, cp_len, sc_map, pilot_value,
                profile, taps_L, self.snr_list, n_mc, velocity,
            )

            self._status("Generando constelaciones MRC y potencia PAPR...")
            const_mrc_before, const_mrc_after = ofdm_utils.generate_mrc_constellation_data(
                Nfft, cp_len, sc_map, pilot_value, profile, taps_L,
                snr_sim, velocity, M_mod=M,
            )
            papr_inst_sfbc, papr_inst_sc, papr_avg_sfbc, papr_avg_sc = (
                ofdm_utils.generate_papr_instant_data(
                    Nfft, cp_len, sc_map, pilot_value, M, M_dft,
                )
            )

            # -- Transmisión de imagen en paralelo: SISO vs MISO-SFBC --
            self._status("Transmitiendo imagen en paralelo...\n(Hebra A: SISO  |  Hebra B: SFBC)")
            img_sfbc_cmp = ofdm_utils.transmit_image_siso_vs_sfbc(
                bits_tx, img_arr, Nfft, cp_len, sc_map, pilot_value,
                profile, taps_L, snr_sim, velocity, M,
            )

            # -- Reporte --
            h_px, w_px = img_arr.shape
            report = (
                f"--- SISTEMA ---\n"
                f"Nfft: {Nfft}  |  CP: {cp_len}\n"
                f"Δf: {delta_f/1e3:.0f} kHz  |  fs: {fs/1e6:.2f} MHz\n"
                f"N_used: {N_used} ({N_used*100//Nfft}% de {Nfft})\n"
                f"Datos OFDM: {sc_map['n_data']}\n"
                f"Pilotos: {sc_map['n_pilots']}\n"
                f"Guarda+DC: {sc_map['n_guard']}\n"
                f"Bits/símbolo OFDM: {bits_per_ofdm}\n"
                f"Ejecuciones IFFT: {n_ofdm}\n\n"
                f"--- SC-FDMA ---\n"
                f"DFT size: {M_dft} (auto)\n"
                f"Datos SC-FDMA: {M_dft}\n"
                f"PAPR máx OFDM: {np.max(papr_list):.2f} dB\n"
                f"PAPR máx SC-FDMA: {np.max(papr_list_sc):.2f} dB\n"
                f"BER OFDM: {ber_img:.4e}\n"
                f"BER SC-FDMA: {ber_img_sc:.4e}\n\n"
                f"--- IMAGEN ({mod_name}) ---\n"
                f"Tamaño: {w_px}×{h_px} px\n"
                f"Bits: {n_bits}  |  PSNR: {psnr:.2f} dB\n\n"
                f"--- CANAL ---\n"
                f"{chan_class['freq_type']}\n"
                f"{chan_class['time_type']}\n\n"
                f"--- DIVERSIDAD RX ---\n"
                f"Antenas RX (NR): {NR}\n"
                f"Técnicas: SISO, MRC, SC, MMSE\n\n"
                f"--- DIVERSIDAD TX ---\n"
                f"MISO-SFBC: 2 antenas TX (Alamouti)\n"
                f"Comparativa: SISO, SIMO-MRC, MISO-SFBC\n"
            )
            self._status(report)

            # -- Almacenar datos --
            self.sim_data = {
                "img_tx": img_arr,
                "img_rx": img_rx,
                "psnr": psnr,
                "ber_img": ber_img,
                "const_tx": const_tx,
                "const_rx": const_rx,
                "tx_signal": tx_signal,
                "h_used": h_used,
                "Nfft": Nfft, "cp_len": cp_len, "N_used": N_used,
                "fs": fs, "delta_f": delta_f,
                "sc_map": sc_map,
                "n_ofdm": n_ofdm,
                "n_symbols_qam": len(symbols_tx),
                "pad_bits": pad_bits,
                "papr_list": papr_list,
                "ber_data": ber_data,
                "ber_no_eq": ber_no_eq,
                "ccdf_data": ccdf_data,
                "papr_means": papr_means,
                "papr_ci": papr_ci,
                "snr_list": self.snr_list.copy(),
                "profile": profile,
                "velocity": velocity,
                "cp_mode": cp_mode,
                "snr_sim": snr_sim,
                "total_bits": n_bits,
                "mod_selected": mod_name,
                "chan_class": chan_class,
                "M_dft": M_dft,
                "sc_map_sc": sc_map_sc,
                "tx_signal_sc": tx_signal_sc,
                "papr_list_sc": papr_list_sc,
                "ber_data_sc": ber_data_sc,
                "ccdf_data_sc": ccdf_data_sc,
                "ber_img_sc": ber_img_sc,
                "n_ofdm_sc": n_ofdm_sc,
                "NR": NR,
                "div_results": div_results,
                "H_single": H_single,
                "H_combined": H_combined,
                "txdiv_results": txdiv_results,
                "const_mrc_before": const_mrc_before,
                "const_mrc_after": const_mrc_after,
                "papr_inst_sfbc": papr_inst_sfbc,
                "papr_inst_sc": papr_inst_sc,
                "papr_avg_sfbc": papr_avg_sfbc,
                "papr_avg_sc": papr_avg_sc,
                "img_sfbc_cmp": img_sfbc_cmp,
            }
            self.render_plots()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise

    # ==============================================================
    # Gráficas
    # ==============================================================

    def render_plots(self):
        d = self.sim_data
        if not d:
            return
        self._render_tab1(d)
        self._render_tab2(d)
        self._render_tab3(d)
        self._render_tab4(d)
        self._render_tab5(d)
        self._render_tab6(d)
        self._render_tab7(d)
        self._render_tab8(d)
        self._render_tab9(d)

    # --- Tab 1: Imagen TX/RX y constelaciones (modulación seleccionada) ---

    def _render_tab1(self, d):
        self._clear_tab(self.tab1)
        mod = d["mod_selected"]
        h_px, w_px = d["img_tx"].shape

        info = (
            f"  {mod}  |  {w_px}×{h_px} px  |  Bits: {d['total_bits']}  |  "
            f"Símbolos: {d['n_symbols_qam']}  |  OFDM: {d['n_ofdm']}  |  "
            f"BER: {d['ber_img']:.4e}  |  PSNR: {d['psnr']:.2f} dB  "
        )
        tk.Label(
            self.tab1, text=info, font=("Consolas", 10, "bold"),
            bg="#0b314d", fg="white", relief="groove", padx=8, pady=4,
        ).pack(fill="x", padx=6, pady=(5, 0))

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(
            f"{mod}  —  SNR: {d['snr_sim']} dB  |  Canal: {d['profile']}  |  "
            f"Vel: {d['velocity']} km/h",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93], pad=2.5)

        axes[0, 0].imshow(d["img_tx"], cmap="gray")
        axes[0, 0].set_title("TX: Imagen original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(d["img_rx"], cmap="gray")
        axes[0, 1].set_title(f"RX: PSNR {d['psnr']:.2f} dB")
        axes[0, 1].axis("off")

        npts = min(2000, len(d["const_tx"]))
        axes[1, 0].scatter(
            np.real(d["const_tx"][:npts]), np.imag(d["const_tx"][:npts]),
            s=4, alpha=0.5, c="#2ecc71",
        )
        axes[1, 0].set_title(f"Constelación TX ({mod})")
        axes[1, 0].set_xlabel("I")
        axes[1, 0].set_ylabel("Q")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_aspect("equal")

        npts_rx = min(2000, len(d["const_rx"]))
        axes[1, 1].scatter(
            np.real(d["const_rx"][:npts_rx]), np.imag(d["const_rx"][:npts_rx]),
            s=4, alpha=0.5, c="#e74c3c",
        )
        axes[1, 1].set_title("Constelación RX (ecualizada por pilotos)")
        axes[1, 1].set_xlabel("I")
        axes[1, 1].set_ylabel("Q")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect("equal")

        self._embed(fig, self.tab1)

    # --- Tab 2: Subportadoras ---

    def _render_tab2(self, d):
        self._clear_tab(self.tab2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.tight_layout(pad=3.0)

        # [0,0] Mapa de color  |  [0,1] Barras resumen
        ofdm_utils.plot_subcarrier_map(
            axes[0, 0], axes[0, 1], d["sc_map"], d["Nfft"]
        )

        # [1,0] Ortogonalidad sinc
        ofdm_utils.plot_subcarrier_spacing(axes[1, 0], d["delta_f"])

        # [1,1] PSD de la señal OFDM
        from scipy import signal as sig

        nperseg = min(256, d["Nfft"])
        f_psd, Pxx = sig.welch(d["tx_signal"], d["fs"], nperseg=nperseg)
        axes[1, 1].semilogy(f_psd / 1e6, Pxx)
        axes[1, 1].set_title("Densidad Espectral de Potencia (OFDM)")
        axes[1, 1].set_xlabel("Frecuencia (MHz)")
        axes[1, 1].set_ylabel("PSD")
        axes[1, 1].grid(True, alpha=0.3)

        self._embed(fig, self.tab2)

    # --- Tab 3: Canal ---

    def _render_tab3(self, d):
        self._clear_tab(self.tab3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout(pad=3.0)
        ofdm_utils.plot_channel_response(
            ax1, ax2, d["h_used"], d["fs"], d["Nfft"],
            d["profile"], d.get("chan_class"),
        )
        self._embed(fig, self.tab3)

    # --- Tab 4: Prefijo cíclico ---

    def _render_tab4(self, d):
        self._clear_tab(self.tab4)
        fig, ax = plt.subplots(figsize=(8, 5))
        cp_n = d["Nfft"] // 8
        cp_e = d["Nfft"] // 4
        ofdm_utils.plot_cp_analysis(ax, cp_n, cp_e, d["Nfft"], d["fs"])
        self._embed(fig, self.tab4)

    # --- Tab 5: BER y PAPR ---

    def _render_tab5(self, d):
        self._clear_tab(self.tab5)
        fig, axes = plt.subplots(3, 2, figsize=(14, 20))
        fig.subplots_adjust(
            left=0.08, right=0.96, top=0.96, bottom=0.04,
            hspace=0.40, wspace=0.28,
        )

        snr_arr = np.array(d["snr_list"])

        # [0,0] BER vs SNR OFDM con pilotos + IC 95%
        any_plotted = False
        for mod_name, ber_info in d["ber_data"].items():
            means = np.array(ber_info["mean"])
            cis = np.array(ber_info["ci"])
            mask = means > 0
            if np.any(mask):
                s = snr_arr[mask]
                m = means[mask]
                c = cis[mask]
                axes[0, 0].semilogy(
                    s, m, marker="o", label=mod_name, color=MOD_COLORS[mod_name],
                )
                upper = m + c
                lower = np.maximum(m - c, 1e-10)
                axes[0, 0].fill_between(
                    s, lower, upper, alpha=0.2, color=MOD_COLORS[mod_name],
                )
                any_plotted = True
        if not any_plotted:
            axes[0, 0].text(
                0.5, 0.5, "BER = 0 en todos los puntos",
                ha="center", va="center", transform=axes[0, 0].transAxes,
            )
        axes[0, 0].set_title("OFDM — BER vs SNR (pilotos + IC 95%)")
        axes[0, 0].set_xlabel("SNR (dB)")
        axes[0, 0].set_ylabel("BER")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, which="both", alpha=0.3)

        # [0,1] BER vs SNR SC-FDMA con pilotos + IC 95%
        any_plotted_sc = False
        for mod_name, ber_info in d["ber_data_sc"].items():
            means = np.array(ber_info["mean"])
            cis = np.array(ber_info["ci"])
            mask = means > 0
            if np.any(mask):
                s = snr_arr[mask]
                m = means[mask]
                c = cis[mask]
                axes[0, 1].semilogy(
                    s, m, marker="s", label=mod_name, color=MOD_COLORS[mod_name],
                )
                upper = m + c
                lower = np.maximum(m - c, 1e-10)
                axes[0, 1].fill_between(
                    s, lower, upper, alpha=0.2, color=MOD_COLORS[mod_name],
                )
                any_plotted_sc = True
        if not any_plotted_sc:
            axes[0, 1].text(
                0.5, 0.5, "BER = 0 en todos los puntos",
                ha="center", va="center", transform=axes[0, 1].transAxes,
            )
        axes[0, 1].set_title(f"SC-FDMA (DFT={d['M_dft']}) — BER vs SNR (pilotos + IC 95%)")
        axes[0, 1].set_xlabel("SNR (dB)")
        axes[0, 1].set_ylabel("BER")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, which="both", alpha=0.3)

        # [1,0] Comparación con pilotos vs sin ecualización (OFDM)
        for mod_name in d["ber_data"]:
            m_eq = np.array(d["ber_data"][mod_name]["mean"])
            m_noeq = np.array(d["ber_no_eq"][mod_name]["mean"])
            color = MOD_COLORS[mod_name]

            mask_eq = m_eq > 0
            if np.any(mask_eq):
                axes[1, 0].semilogy(
                    snr_arr[mask_eq], m_eq[mask_eq],
                    marker="o", color=color, label=f"{mod_name} (pilotos)",
                )
            mask_noeq = m_noeq > 0
            if np.any(mask_noeq):
                axes[1, 0].semilogy(
                    snr_arr[mask_noeq], m_noeq[mask_noeq],
                    marker="x", linestyle="--", color=color,
                    label=f"{mod_name} (sin ecual.)", alpha=0.7,
                )
        axes[1, 0].set_title("Efecto de Ecualización por Pilotos")
        axes[1, 0].set_xlabel("SNR (dB)")
        axes[1, 0].set_ylabel("BER")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, which="both", alpha=0.3)

        # [1,1] CCDF del PAPR — OFDM (sólido) vs SC-FDMA (punteado)
        for mod_name, (papr_sorted, ccdf) in d["ccdf_data"].items():
            color = MOD_COLORS[mod_name]
            axes[1, 1].semilogy(
                papr_sorted, ccdf, color=color, label=f"{mod_name} OFDM",
            )
        for mod_name, (papr_sorted, ccdf) in d["ccdf_data_sc"].items():
            color = MOD_COLORS[mod_name]
            axes[1, 1].semilogy(
                papr_sorted, ccdf, color=color, linestyle="--",
                label=f"{mod_name} SC-FDMA", alpha=0.8,
            )
        axes[1, 1].set_title("CCDF PAPR — OFDM vs SC-FDMA")
        axes[1, 1].set_xlabel("PAPR (dB)")
        axes[1, 1].set_ylabel("Prob{PAPR > x}")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(True, which="both", alpha=0.3)

        # [2,0] Potencia instantánea OFDM
        ofdm_utils.plot_papr_time_domain(
            axes[2, 0], d["tx_signal"], d["Nfft"], d["cp_len"]
        )

        # [2,1] Potencia instantánea SC-FDMA
        ofdm_utils.plot_papr_time_domain(
            axes[2, 1], d["tx_signal_sc"], d["Nfft"], d["cp_len"]
        )
        axes[2, 1].set_title("Potencia de un Símbolo SC-FDMA")

        self._embed(fig, self.tab5)


    # --- Tab 6: OFDM vs SC-FDMA ---

    def _render_tab6(self, d):
        self._clear_tab(self.tab6)
        M_dft = d["M_dft"]

        info = (
            f"  OFDM (IFFT={d['Nfft']})  vs  SC-FDMA (FFT={M_dft} → IFFT={d['Nfft']})  |  "
            f"Datos OFDM: {d['sc_map']['n_data']}  |  Datos SC-FDMA: {M_dft}  |  "
            f"Canal: {d['profile']}  |  Vel: {d['velocity']} km/h"
        )
        tk.Label(
            self.tab6, text=info, font=("Consolas", 9, "bold"),
            bg="#4a235a", fg="white", relief="groove", padx=8, pady=4,
        ).pack(fill="x", padx=6, pady=(5, 0))

        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        fig.suptitle(
            f"Comparación OFDM vs SC-FDMA  —  DFT={M_dft} (auto), IFFT={d['Nfft']}",
            fontsize=12, fontweight="bold",
        )
        fig.subplots_adjust(
            left=0.08, right=0.96, top=0.94, bottom=0.04,
            hspace=0.45, wspace=0.3,
        )

        # [0,0] y [0,1] Mapas de subportadoras
        ofdm_utils.plot_comparative_subcarrier_maps(
            axes[0, 0], axes[0, 1],
            d["sc_map"], d["sc_map_sc"], d["Nfft"], M_dft,
        )

        # [1,0] Conteo de subportadoras
        ofdm_utils.plot_comparative_subcarrier_count(
            axes[1, 0], d["sc_map"], d["sc_map_sc"], d["Nfft"], M_dft,
        )

        # [1,1] Ancho de banda
        ofdm_utils.plot_comparative_bandwidth(
            axes[1, 1], d["sc_map"], d["sc_map_sc"], d["delta_f"], M_dft,
        )

        # [2,0] BER comparativo
        ofdm_utils.plot_comparative_ber(
            axes[2, 0], d["snr_list"],
            d["ber_data"], d["ber_data_sc"], MOD_COLORS,
        )

        # [2,1] Potencia de símbolo OFDM vs SC-FDMA
        ofdm_utils.plot_comparative_symbol_power(
            axes[2, 1], d["tx_signal"], d["tx_signal_sc"],
            d["Nfft"], d["cp_len"],
        )

        self._embed(fig, self.tab6)

        # Segunda figura: CCDF PAPR comparativa (más grande)
        fig2, ax_ccdf = plt.subplots(figsize=(10, 5))
        ofdm_utils.plot_comparative_ccdf(
            ax_ccdf, d["ccdf_data"], d["ccdf_data_sc"], MOD_COLORS,
        )
        fig2.tight_layout()
        self._embed(fig2, self.tab6)


    # --- Tab 7: Diversidad RX ---

    def _render_tab7(self, d):
        self._clear_tab(self.tab7)
        NR = d["NR"]

        info = (
            f"  Diversidad RX  |  NR={NR} antenas  |  "
            f"Técnicas: SISO vs MRC vs SC vs MMSE  |  "
            f"Canal: {d['profile']}  |  Vel: {d['velocity']} km/h  |  "
            f"SC-FDMA (DFT={d['M_dft']})"
        )
        tk.Label(
            self.tab7, text=info, font=("Consolas", 9, "bold"),
            bg="#1a5276", fg="white", relief="groove", padx=8, pady=4,
        ).pack(fill="x", padx=6, pady=(5, 0))

        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        fig.suptitle(
            f"Diversidad en Recepción — NR={NR} antenas, SC-FDMA",
            fontsize=12, fontweight="bold",
        )
        fig.subplots_adjust(
            left=0.08, right=0.96, top=0.93, bottom=0.06,
            hspace=0.35, wspace=0.28,
        )

        # [0,0] BER QPSK
        ofdm_utils.plot_diversity_ber(
            axes[0, 0], d["snr_list"], d["div_results"], "QPSK", NR
        )

        # [0,1] BER 16QAM
        ofdm_utils.plot_diversity_ber(
            axes[0, 1], d["snr_list"], d["div_results"], "16QAM", NR
        )

        # [1,0] BER 64QAM
        ofdm_utils.plot_diversity_ber(
            axes[1, 0], d["snr_list"], d["div_results"], "64QAM", NR
        )

        # [1,1] Ganancia de diversidad vs fading
        ofdm_utils.plot_diversity_power_stability(
            axes[1, 1], d["H_single"], d["H_combined"], d["Nfft"], NR
        )

        self._embed(fig, self.tab7)

    # --- Tab 8: Diversidad TX (MISO-SFBC) ---

    def _render_tab8(self, d):
        self._clear_tab(self.tab8)

        info = (
            f"  Diversidad TX  |  MISO-SFBC (2 TX, Alamouti)  |  "
            f"Comparativa: SISO vs SIMO-MRC (2 RX) vs MISO-SFBC (2 TX)  |  "
            f"Canal: {d['profile']}  |  Vel: {d['velocity']} km/h  |  "
            f"SC-FDMA (DFT={d['M_dft']})"
        )
        tk.Label(
            self.tab8, text=info, font=("Consolas", 9, "bold"),
            bg="#7e3517", fg="white", relief="groove", padx=8, pady=4,
        ).pack(fill="x", padx=6, pady=(5, 0))

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(
            "Diversidad en Transmisión — MISO con Codificación SFBC (Alamouti)",
            fontsize=13, fontweight="bold",
        )
        fig.subplots_adjust(
            left=0.06, right=0.97, top=0.92, bottom=0.07,
            hspace=0.30, wspace=0.30,
        )

        # Fila superior: BER comparativo SISO vs SIMO-MRC vs MISO-SFBC
        ofdm_utils.plot_diversity_tx_ber(axes[0, 0], d["snr_list"], d["txdiv_results"], "QPSK")
        ofdm_utils.plot_diversity_tx_ber(axes[0, 1], d["snr_list"], d["txdiv_results"], "16QAM")
        ofdm_utils.plot_diversity_tx_ber(axes[0, 2], d["snr_list"], d["txdiv_results"], "64QAM")

        # Fila inferior: constelación antes/después MRC + potencia instantánea
        ofdm_utils.plot_mrc_constellation(
            axes[1, 0], d["const_mrc_before"],
            f"Constelación antes de MRC (SISO) — SNR={d['snr_sim']} dB",
            M_mod=MOD_MAP[d["mod_selected"]],
        )
        ofdm_utils.plot_mrc_constellation(
            axes[1, 1], d["const_mrc_after"],
            f"Constelación después de MRC (2 RX) — SNR={d['snr_sim']} dB",
            M_mod=MOD_MAP[d["mod_selected"]],
        )
        ofdm_utils.plot_papr_instant(
            axes[1, 2], d["papr_inst_sfbc"], d["papr_inst_sc"],
            d["papr_avg_sfbc"], d["papr_avg_sc"],
        )

        self._embed(fig, self.tab8)

    # --- Tab 9: Imagen SISO vs SFBC (transmisión paralela) ---

    def _render_tab9(self, d):
        self._clear_tab(self.tab9)
        cmp = d["img_sfbc_cmp"]

        mejora = cmp["psnr_sfbc"] - cmp["psnr_siso"]
        info = (
            f"  Transmisión paralela de imagen  |  "
            f"Hebra A: SISO (1 TX)  vs  Hebra B: MISO-SFBC (2 TX, Alamouti)  |  "
            f"Canal: {d['profile']}  |  SNR: {cmp['snr_db']} dB  |  "
            f"Mejora PSNR con SFBC: {mejora:+.2f} dB"
        )
        tk.Label(
            self.tab9, text=info, font=("Consolas", 9, "bold"),
            bg="#145a32", fg="white", relief="groove", padx=8, pady=4,
        ).pack(fill="x", padx=6, pady=(5, 0))

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0])
        fig.suptitle(
            "Transmisión de Imagen: SISO vs MISO-SFBC sobre el mismo canal",
            fontsize=13, fontweight="bold",
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        axc = fig.add_subplot(gs[1, :])

        ofdm_utils.plot_image_panel(
            ax0, cmp["img_orig"], "Original (Referencia)",
        )
        ofdm_utils.plot_image_panel(
            ax1, cmp["img_siso"], "SISO (1 antena)",
            subtitle=(f"PSNR = {cmp['psnr_siso']:.2f} dB   |   "
                      f"MSE = {cmp['mse_siso']:.1f}   |   "
                      f"BER = {cmp['ber_siso']:.2e}"),
        )
        ofdm_utils.plot_image_panel(
            ax2, cmp["img_sfbc"], "MISO-SFBC (2 antenas)",
            subtitle=(f"PSNR = {cmp['psnr_sfbc']:.2f} dB   |   "
                      f"MSE = {cmp['mse_sfbc']:.1f}   |   "
                      f"BER = {cmp['ber_sfbc']:.2e}"),
        )

        ofdm_utils.plot_sfbc_channel_redundancy(
            axc, cmp["H1"], cmp["H2"], cmp["used_indices"], cmp["Nfft"],
        )

        fig.subplots_adjust(
            left=0.05, right=0.97, top=0.90, bottom=0.08, hspace=0.30, wspace=0.10,
        )
        self._embed(fig, self.tab9)


if __name__ == "__main__":
    root = tk.Tk()
    app = OFDM_Simulator(root)
    root.mainloop()
