import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import bits
import modulacion # El nuevo integrante

class AppTelecom:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Modulación de Imagen")
        self.ventana.geometry("1200x700")
        self.ventana.configure(bg="#212121")

        self.matriz_bits = None
        self.ruta_img = None
        self.setup_ui()

    def setup_ui(self):
        # Frame Lateral de Controles
        side_panel = tk.Frame(self.ventana, bg="#2c3e50", width=250)
        side_panel.pack(side="left", fill="y", padx=10)

        tk.Label(side_panel, text="OPCIONES", fg="white", bg="#2c3e50", font=("Arial", 12, "bold")).pack(pady=20)

        tk.Button(side_panel, text="1. Cargar Foto", command=self.cargar, bg="#3498db", fg="white").pack(fill="x", padx=10, pady=5)
        tk.Button(side_panel, text="2. Ver Bits", command=self.procesar_bits, bg="#e67e22", fg="white").pack(fill="x", padx=10, pady=5)

        tk.Label(side_panel, text="Modulación:", fg="white", bg="#2c3e50").pack(pady=(20, 0))
        self.combo_mod = ttk.Combobox(side_panel, values=["QPSK", "16QAM", "64QAM"], state="readonly")
        self.combo_mod.current(0)
        self.combo_mod.pack(fill="x", padx=10, pady=5)

        tk.Button(side_panel, text="3. Modular y Ver Constelación", command=self.ejecutar_modulacion, bg="#9b59b6", fg="white").pack(fill="x", padx=10, pady=5)

        # Área de visualización (Derecha)
        self.main_frame = tk.Frame(self.ventana, bg="#212121")
        self.main_frame.pack(side="right", expand=True, fill="both")

        self.lbl_img = tk.Label(self.main_frame, bg="#333", text="Imagen")
        self.lbl_img.grid(row=0, column=0, padx=10, pady=10)

        # Espacio para el gráfico de Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.fig.patch.set_facecolor('#212121')
        self.ax.set_facecolor('black')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_plot.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

    def cargar(self):
        self.ruta_img = filedialog.askopenfilename()
        if self.ruta_img:
            img = Image.open(self.ruta_img).convert('RGB')
            img.thumbnail((300, 300))
            foto = ImageTk.PhotoImage(img)
            self.lbl_img.config(image=foto)
            self.lbl_img.image = foto

    def procesar_bits(self):
        if self.ruta_img:
            self.matriz_bits, _ = bits.transformar_a_bits(self.ruta_img)
            messagebox.showinfo("Bits", "Imagen convertida a bits con éxito")

    def ejecutar_modulacion(self):
        if self.matriz_bits is None:
            return messagebox.showwarning("Error", "Primero procesa los bits")
        
        tipo = self.combo_mod.get()
        simbolos = modulacion.modular_bits(self.matriz_bits, tipo)
        
        # Actualizar Diagrama de Constelación
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.scatter(simbolos.real, simbolos.imag, s=1, color="#00ff00", alpha=0.5)
        self.ax.set_title(f"Constelación {tipo}", color="white")
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
        self.ax.tick_params(colors='white')
        
        self.canvas_plot.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppTelecom(root)
    root.mainloop()