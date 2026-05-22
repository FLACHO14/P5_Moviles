import numpy as np
from PIL import Image

def transformar_a_bits(ruta_imagen):
    """Convierte una imagen en una matriz de strings binarios (8 bits)."""
    # Abrimos y aseguramos RGB
    img = Image.open(ruta_imagen).convert('RGB')
    arr = np.array(img)
    
    # Vectorizamos la función format para que sea rápida en NumPy
    # '08b' convierte el número a binario con 8 dígitos (ej: 2 -> 00000010)
    v_binario = np.vectorize(lambda x: format(x, '08b'))
    matriz_bits = v_binario(arr)
    
    return matriz_bits, arr.shape

def transformar_a_color(matriz_bits):
    """Convierte una matriz de bits de vuelta a una imagen PIL a color."""
    # Convertimos cada string binario de vuelta a un entero decimal
    v_decimal = np.vectorize(lambda x: int(x, 2))
    arr_regreso = v_decimal(matriz_bits).astype(np.uint8)
    
    # Reconstruimos la imagen
    return Image.fromarray(arr_regreso)