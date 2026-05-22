import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import bits
import modulacion 

class AppTelecomLight:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Sistema de Modulación de Imagen - Modo Claro")
        self.ventana.geometry("1200x750")
        
        # Colores del tema claro
        self.color_fondo = "#f0f2f5"  # Gris casi blanco
        self.color_lateral = "#ffffff" # Blanco puro
        self.color_texto = "#1c1e21"   # Negro/Gris oscuro
        
        self.ventana.configure(bg=self.color_fondo)

        self.matriz_bits = None
        self.ruta_img = None
        self.setup_ui()

    def setup_ui(self):
        # --- Panel Lateral de Controles (Blanco con sombra ligera) ---
        side_panel = tk.Frame(self.ventana, bg=self.color_lateral, width=280, relief="flat")
        side_panel.pack(side="left", fill="y", padx=0)
        
        tk.Label(side_panel, text="SISTEMA DE CONTROL", fg="#0561ff", bg=self.color_lateral, 
                 font=("Segoe UI", 14, "bold")).pack(pady=30)

        # Estilo de botones
        btn_config = {"font": ("Segoe UI", 10), "bd": 0, "cursor": "hand2", "pady": 8}

        tk.Button(side_panel, text="1. Cargar Imagen", command=self.cargar, 
                  bg="#e7f3ff", fg="#1877f2", **btn_config).pack(fill="x", padx=20, pady=10)
        
        tk.Button(side_panel, text="2. Generar Bits", command=self.procesar_bits, 
                  bg="#f2f2f2", fg="#4b4b4b", **btn_config).pack(fill="x", padx=20, pady=10)

        tk.Label(side_panel, text="Tipo de Modulación:", fg=self.color_texto, bg=self.color_lateral, 
                 font=("Segoe UI", 10)).pack(pady=(20, 0))
        
        self.combo_mod = ttk.Combobox(side_panel, values=["QPSK", "16QAM", "64QAM"], state="readonly")
        self.combo_mod.current(0)
        self.combo_mod.pack(fill="x", padx=20, pady=10)

        tk.Button(side_panel, text="3. Ver Constelación", command=self.ejecutar_modulacion, 
                  bg="#1877f2", fg="white", **btn_config).pack(fill="x", padx=20, pady=10)

        # --- Área Principal (Derecha) ---
        self.main_frame = tk.Frame(self.ventana, bg=self.color_fondo)
        self.main_frame.pack(side="right", expand=True, fill="both", padx=20)

        # Contenedor para la imagen original
        frame_img = tk.Frame(self.main_frame, bg="white", bd=1, relief="solid")
        frame_img.grid(row=0, column=0, padx=20, pady=20)
        
        self.lbl_img = tk.Label(frame_img, bg="white", text="Esperando imagen...", width=40, height=20)
        self.lbl_img.pack(padx=5, pady=5)

        # Espacio para el gráfico de Matplotlib (Constelación)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.patch.set_facecolor('white') # Fondo del gráfico blanco
        self.ax.set_facecolor('#f8f9fa')
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_plot.get_tk_widget().grid(row=0, column=1, padx=20, pady=20)

    def cargar(self):
        self.ruta_img = filedialog.askopenfilename()
        if self.ruta_img:
            img = Image.open(self.ruta_img).convert('RGB')
            img.thumbnail((350, 350))
            foto = ImageTk.PhotoImage(img)
            self.lbl_img.config(image=foto, text="")
            self.lbl_img.image = foto

    def procesar_bits(self):
        if self.ruta_img:
            self.matriz_bits, _ = bits.transformar_a_bits(self.ruta_img)
            messagebox.showinfo("Éxito", "Imagen descompuesta en bits correctamente.")

    def ejecutar_modulacion(self):
        if self.matriz_bits is None:
            return messagebox.showwarning("Atención", "Primero genera los bits de la imagen.")
        
        tipo = self.combo_mod.get()
        simbolos = modulacion.modular_bits(self.matriz_bits, tipo)
        
        # Actualizar Diagrama de Constelación con colores claros
        self.ax.clear()
        self.ax.set_facecolor('#f8f9fa')
        # Dibujamos los puntos en color azul para que resalten en blanco
        self.ax.scatter(simbolos.real, simbolos.imag, s=10, color="#1877f2", alpha=0.6)
        
        self.ax.set_title(f"Diagrama de Constelación {tipo}", color="#1c1e21", fontsize=12, pad=15)
        self.ax.grid(color='white', linestyle='-', linewidth=2)
        self.ax.tick_params(colors='#4b4b4b')
        
        # Dibujar ejes centrales
        self.ax.axhline(0, color='gray', linewidth=1)
        self.ax.axvline(0, color='gray', linewidth=1)
        
        self.canvas_plot.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppTelecomLight(root)
    root.mainloop()