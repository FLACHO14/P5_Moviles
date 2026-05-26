# ofdm_params.py
import numpy as np

DELTA_F = 15000          # 15 kHz (estándar LTE)
FC = 2.1e9               # Frecuencia portadora 2.1 GHz

PILOT_SPACING_DEFAULT = 8
PILOT_AMPLITUDE = 1.0 + 1.0j

PROFILES = {
    "Ideal": None,
    "Rayleigh (NLoS)": "rayleigh",
    "Rician (LoS)": "rician",
    "Suburbano (EPA)": "EPA",
    "Urbano (EVA)": "EVA",
    "Rural (ETU)": "ETU"
}

COLORS = {
    'tx': '#2ecc71',
    'rx': '#e74c3c',
    'pilot': '#3498db',
    'channel': '#f1c40f',
    'qpsk': '#3498db',
    '16qam': '#2ecc71',
    '64qam': '#e74c3c'
}