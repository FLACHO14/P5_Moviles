import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import ofdm_tx, ofdm_channel, ofdm_rx, ofdm_utils, ofdm_params

MOD_MAP = {"QPSK":4, "16QAM":16, "64QAM":64}
MOD_COLORS = {"QPSK":"#3498db", "16QAM":"#2ecc71", "64QAM":"#e74c3c"}
VEL_MAP = {"Estático (0)":0, "Pedestre (3 km/h)":3, "Urbano (50 km/h)":50, "Autopista (120 km/h)":120}
CHAN_MODELS = ["Ideal", "Rayleigh (NLoS)", "Rician (LoS)", "Suburbano (EPA)", "Urbano (EVA)", "Rural (ETU)"]

class OFDM_Simulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador OFDM 4G LTE - Ingeniería")
        self.root.geometry("1600x1000")
        
        self.img_path = tk.StringVar()
        self.mod_var = tk.StringVar(value="16QAM")
        self.chan_profile = tk.StringVar(value="Rayleigh (NLoS)")
        self.vel_var = tk.StringVar(value="Urbano (50 km/h)")
        self.cp_var = tk.StringVar(value="Normal")
        self.bw_var = tk.DoubleVar(value=50.0)
        self.snr_entry_var = tk.StringVar(value="20")
        self.mc_var = tk.StringVar(value="10")
        self.taps_var = tk.StringVar(value="8")
        self.max_side_var = tk.StringVar(value="128")
        
        self.snr_list = [0, 5, 10, 15, 20]
        self.sim_data = {}
        self.setup_ui()
        self._refresh_snr_listbox()
    
    def setup_ui(self):
        ctrl_frame = ttk.LabelFrame(self.root, text=" Configuración de Parámetros ")
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        tk.Button(ctrl_frame, text="CARGAR IMAGEN", command=self.load_image,
                  bg="#34495e", fg="white", font=("Helvetica",10,"bold")).pack(pady=10, fill="x")
        
        ttk.Label(ctrl_frame, text="Lado máx. imagen (px):").pack()
        ttk.Entry(ctrl_frame, textvariable=self.max_side_var, width=8).pack(pady=3)
        
        ttk.Label(ctrl_frame, text="Ancho de Banda (MHz):").pack()
        ttk.Entry(ctrl_frame, textvariable=self.bw_var).pack(pady=3, fill="x", padx=5)
        
        ttk.Label(ctrl_frame, text="Modulación (referencia):").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.mod_var, values=list(MOD_MAP.keys())).pack(pady=3, fill="x", padx=5)
        
        ttk.Label(ctrl_frame, text="Modelo de Canal:").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.chan_profile, values=CHAN_MODELS).pack(pady=3, fill="x", padx=5)
        
        ttk.Label(ctrl_frame, text="Velocidad (Doppler):").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.vel_var, values=list(VEL_MAP.keys())).pack(pady=3, fill="x", padx=5)
        
        ttk.Label(ctrl_frame, text="Prefijo Cíclico:").pack()
        ttk.Combobox(ctrl_frame, textvariable=self.cp_var, values=["Normal","Extendido"]).pack(pady=3, fill="x", padx=5)
        
        ttk.Label(ctrl_frame, text="Taps del Canal (L):").pack()
        ttk.Entry(ctrl_frame, textvariable=self.taps_var, width=8).pack(pady=3)
        
        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", padx=5, pady=6)
        
        ttk.Label(ctrl_frame, text="Valores SNR (dB):", font=("Helvetica",9,"bold")).pack()
        snr_add_frame = ttk.Frame(ctrl_frame)
        snr_add_frame.pack(fill="x", padx=5, pady=2)
        ttk.Entry(snr_add_frame, textvariable=self.snr_entry_var, width=8).pack(side="left", padx=(0,4))
        tk.Button(snr_add_frame, text="Añadir", command=self.add_snr, bg="#2980b9", fg="white").pack(side="left")
        self.snr_listbox = tk.Listbox(ctrl_frame, height=5, width=30, font=("Consolas",9))
        self.snr_listbox.pack(pady=2, padx=5, fill="x")
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", padx=5)
        tk.Button(btn_frame, text="Eliminar", command=self.remove_snr, bg="#c0392b", fg="white").pack(side="left", padx=(0,3))
        tk.Button(btn_frame, text="Limpiar", command=self.clear_snr, bg="#7f8c8d", fg="white").pack(side="left")
        
        ttk.Label(ctrl_frame, text="Iteraciones Monte Carlo:").pack()
        ttk.Entry(ctrl_frame, textvariable=self.mc_var, width=8).pack(pady=3)
        
        tk.Button(ctrl_frame, text="EJECUTAR TRANSMISIÓN", command=self.run_simulation,
                  bg="#27ae60", fg="white", font=("Helvetica",12,"bold"), height=2).pack(pady=10, fill="x")
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook); self.notebook.add(self.tab1, text="1. Imagen y Constelaciones")
        self.tab2 = ttk.Frame(self.notebook); self.notebook.add(self.tab2, text="2. Análisis de Subportadoras")
        self.tab3 = ttk.Frame(self.notebook); self.notebook.add(self.tab3, text="3. Respuesta del Canal")
        self.tab4 = ttk.Frame(self.notebook); self.notebook.add(self.tab4, text="4. Prefijo Cíclico")
        self.tab5 = ttk.Frame(self.notebook); self.notebook.add(self.tab5, text="5. Curvas BER y PAPR")
    
    def add_snr(self):
        try:
            val = float(self.snr_entry_var.get())
            if val not in self.snr_list:
                self.snr_list.append(val)
                self.snr_list.sort()
                self._refresh_snr_listbox()
        except: pass
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
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.img_path.set(path)
    def _psnr(self, orig, dec):
        mse = np.mean((orig.astype(float)-dec.astype(float))**2)
        if mse==0: return 100
        return 20*np.log10(255.0/np.sqrt(mse))
    def _clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()
    def _status(self, msg):
        print(msg)
    
    def run_simulation(self):
        if not self.img_path.get():
            messagebox.showwarning("Imagen", "Cargue una imagen primero.")
            return
        if not self.snr_list:
            messagebox.showwarning("SNR", "Añada al menos un valor de SNR.")
            return
        
        try:
            bw = self.bw_var.get()
            cp_mode = self.cp_var.get()
            Nfft, cp_len, N_used = ofdm_utils.get_nfft_cp(bw, cp_mode)
            fs = Nfft * ofdm_params.DELTA_F
            velocity = VEL_MAP[self.vel_var.get()]
            profile = self.chan_profile.get()
            taps_L = max(1, int(self.taps_var.get()))
            n_mc = max(1, int(self.mc_var.get()))
            snr_sim = max(self.snr_list)
            
            max_side = max(8, int(self.max_side_var.get()))
            img = Image.open(self.img_path.get()).convert("L")
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            img_arr = np.array(img)
            bits_tx = np.unpackbits(img_arr.flatten())
            n_bits = len(bits_tx)
            img_shape = img_arr.shape
            
            h_impulse = ofdm_channel.get_channel_profile(profile, taps_L)
            
            results = {}
            mods = {"QPSK":4, "16QAM":16, "64QAM":64}
            
            for mod_name, M in mods.items():
                k = int(np.log2(M))
                bits_per_block = Nfft * k
                n_blocks = int(np.ceil(n_bits / bits_per_block))
                total_bits_to_tx = n_blocks * bits_per_block
                pad_bits = total_bits_to_tx - n_bits
                bits_in = np.pad(bits_tx, (0, pad_bits), constant_values=0) if pad_bits else bits_tx.copy()
                
                symbols_qam = ofdm_tx.qam_mod(bits_in, M)
                tx_signal, _ = ofdm_tx.ofdm_tx_block(symbols_qam, Nfft, cp_len)
                rx_signal, _ = ofdm_channel.apply_channel(tx_signal, h_impulse, snr_sim, velocity, fs=fs)
                Y = ofdm_rx.ofdm_rx_block(rx_signal, Nfft, cp_len)
                expected_len = n_blocks * Nfft
                if len(Y) < expected_len:
                    Y = np.pad(Y, (0, expected_len - len(Y)), constant_values=0)
                else:
                    Y = Y[:expected_len]
                H_freq = np.fft.fft(h_impulse, n=Nfft)
                n_frames = len(Y)//Nfft
                Xhat = ofdm_rx.equalize(Y, np.tile(H_freq, n_frames))
                bits_rx_full = ofdm_rx.qam_demod(Xhat, M)
                if len(bits_rx_full) < total_bits_to_tx:
                    bits_rx_full = np.pad(bits_rx_full, (0, total_bits_to_tx - len(bits_rx_full)), constant_values=0)
                else:
                    bits_rx_full = bits_rx_full[:total_bits_to_tx]
                bits_rx = bits_rx_full[:n_bits]
                if len(bits_rx) < n_bits:
                    bits_rx = np.pad(bits_rx, (0, n_bits - len(bits_rx)), constant_values=0)
                img_bytes = np.packbits(bits_rx)
                if len(img_bytes) != img_arr.size:
                    if len(img_bytes) > img_arr.size:
                        img_bytes = img_bytes[:img_arr.size]
                    else:
                        img_bytes = np.pad(img_bytes, (0, img_arr.size - len(img_bytes)), constant_values=0)
                img_rx = img_bytes.reshape(img_shape)
                psnr = self._psnr(img_arr, img_rx)
                
                results[mod_name] = {
                    "img_rx": img_rx,
                    "psnr": psnr,
                    "n_ofdm_blocks": n_blocks,
                    "const_rx": Xhat[:2000]
                }
            
            ber_data, ccdf_data = ofdm_utils.run_analysis(bits_tx, Nfft, cp_len, profile, taps_L, self.snr_list, n_mc)
            papr_means, papr_ci = {}, {}
            for mod, M in mods.items():
                means, cis = ofdm_utils.compute_papr_vs_snr(Nfft, cp_len, M, self.snr_list, n_mc=10)
                papr_means[mod] = means
                papr_ci[mod] = cis
            
            self.sim_data = {
                "img_tx": img_arr,
                "results": results,
                "h_used": h_impulse,
                "Nfft": Nfft, "cp_len": cp_len, "fs": fs,
                "ber_data": ber_data, "ccdf_data": ccdf_data,
                "snr_list": self.snr_list.copy(),
                "papr_means": papr_means, "papr_ci": papr_ci,
                "profile": profile, "velocity": velocity, "cp_mode": cp_mode,
                "snr_sim": snr_sim,
                "total_bits": n_bits,
                "total_ofdm_blocks": sum(results[m]["n_ofdm_blocks"] for m in mods)
            }
            self.render_plots()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise
    
    def render_plots(self):
        d = self.sim_data
        if not d: return
        self._render_tab1(d)
        self._render_tab2(d)
        self._render_tab3(d)
        self._render_tab4(d)
        self._render_tab5(d)
    
    def _embed(self, fig, tab):
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
    
    def _render_tab1(self, d):
        self._clear_tab(self.tab1)
        mods = list(d["results"].keys())
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(
            f"SNR transmisión: {d['snr_sim']} dB | Perfil: {d['profile']} | Velocidad: {d['velocity']} km/h\n"
            f"Bits de imagen: {d['total_bits']} | Símbolos OFDM totales: {d['total_ofdm_blocks']}",
            fontsize=12, fontweight='bold'
        )
        # Imagen TX (fila superior, columnas 1-3)
        ax_tx = plt.subplot(3, 3, (1, 3))
        ax_tx.imshow(d["img_tx"], cmap="gray")
        ax_tx.set_title("Imagen original (TX)")
        ax_tx.axis("off")
        # Para cada modulación: imagen RX y constelación RX
        for col, mod in enumerate(mods):
            ax_rx = plt.subplot(3, 3, 3 + col + 1)   # segunda fila
            ax_rx.imshow(d["results"][mod]["img_rx"], cmap="gray")
            ax_rx.set_title(f"{mod} - RX\nPSNR: {d['results'][mod]['psnr']:.2f} dB")
            ax_rx.axis("off")
            ax_c = plt.subplot(3, 3, 6 + col + 1)    # tercera fila
            const_rx = d["results"][mod]["const_rx"]
            ax_c.scatter(np.real(const_rx), np.imag(const_rx), s=5, alpha=0.5, c='r', label='RX')
            ax_c.set_title(f"Constelación recibida ({mod})")
            ax_c.set_xlabel("I"); ax_c.set_ylabel("Q")
            ax_c.grid(True)
            ax_c.set_aspect('equal')
            ax_c.legend()
        plt.tight_layout()
        self._embed(fig, self.tab1)
    
    def _render_tab2(self, d):
        self._clear_tab(self.tab2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ofdm_utils.plot_subcarrier_analysis(axes, d["Nfft"], d["fs"])
        self._embed(fig, self.tab2)
    
    def _render_tab3(self, d):
        self._clear_tab(self.tab3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ofdm_utils.plot_channel_response(ax1, ax2, d["h_used"], d["fs"], d["Nfft"], d["profile"])
        self._embed(fig, self.tab3)
    
    def _render_tab4(self, d):
        self._clear_tab(self.tab4)
        fig, ax = plt.subplots(figsize=(8, 5))
        cp_normal = d["Nfft"] // 8
        cp_extended = d["Nfft"] // 4
        ofdm_utils.plot_cp_analysis(ax, cp_normal, cp_extended, d["Nfft"], d["fs"])
        self._embed(fig, self.tab4)
    
    def _render_tab5(self, d):
        self._clear_tab(self.tab5)
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        for mod_name, (papr_sorted, ccdf) in d["ccdf_data"].items():
            axes[0].semilogy(papr_sorted, ccdf, label=mod_name, color=MOD_COLORS[mod_name])
        axes[0].set_title("CCDF del PAPR")
        axes[0].set_xlabel("PAPR (dB)"); axes[0].set_ylabel("Prob(PAPR > x)")
        axes[0].legend(); axes[0].grid(True)
        snr_arr = d["snr_list"]
        for mod_name, ber_vals in d["ber_data"].items():
            axes[1].semilogy(snr_arr, ber_vals, marker='o', label=mod_name, color=MOD_COLORS[mod_name])
        axes[1].set_title("BER vs SNR")
        axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("BER")
        axes[1].legend(); axes[1].grid(True)
        self._embed(fig, self.tab5)

if __name__ == "__main__":
    root = tk.Tk()
    app = OFDM_Simulator(root)
    root.mainloop()