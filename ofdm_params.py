# ofdm_params.py

# Parámetros de Radiofrecuencia y OFDM
DELTA_F = 15000  # Espaciado de subportadoras 15 kHz (Estándar 4G)
NFFT_DEFAULT = 1024 # Tamaño de FFT por defecto
FC = 2.1e9       # Frecuencia portadora (2.1 GHz)

# Parámetros de Prefijo Cíclico (en muestras para NFFT=1024)
# En LTE, el CP normal es ~4.7us y el extendido ~16.6us
CP_NORMAL = 72
CP_EXTENDED = 256

# Configuración de Pilotos
PILOT_SPACING = 8 # Un piloto cada 8 subportadoras (tipo peine)
PILOT_AMPLITUDE = 1.0 + 1j

# Parámetros de Canal
SPEEDS = {
    "Estático": 0,
    "Pedestre": 3,
    "Urbano": 50,
    "Autopista": 120
}

# Colores para la interfaz (Estilo 4G)
COLORS = {
    'tx': '#2ecc71', # Verde
    'rx': '#e74c3c', # Rojo
    'pilot': '#3498db', # Azul
    'channel': '#f1c40f' # Amarillo
}