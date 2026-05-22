import numpy as np
import matplotlib.pyplot as plt

def modular_bits(matriz_bits, tipo):
    # 1. Convertir matriz de bits a una sola cadena plana de ceros y unos
    bitstream = "".join(matriz_bits.flatten())
    
    # Definir bits por símbolo
    n_bits = {"QPSK": 2, "16QAM": 4, "64QAM": 6}[tipo]
    
    # Ajustar el bitstream para que sea múltiplo de n_bits (relleno con ceros)
    sobrante = len(bitstream) % n_bits
    if sobrante != 0:
        bitstream += "0" * (n_bits - sobrante)
    
    # 2. Agrupar bits y convertir a símbolos complejos
    simbolos = []
    for i in range(0, len(bitstream), n_bits):
        grupo = bitstream[i:i+n_bits]
        
        # Mapeo simplificado para demostración (Sistema I/Q)
        if tipo == "QPSK":
            # 2 bits: 00, 01, 10, 11
            re = 1 if grupo[0] == '0' else -1
            im = 1 if grupo[1] == '0' else -1
            simbolos.append(complex(re, im))
            
        elif tipo == "16QAM":
            # 4 bits: niveles -3, -1, 1, 3
            mapping = {'00':-3, '01':-1, '11':1, '10':3}
            re = mapping[grupo[0:2]]
            im = mapping[grupo[2:4]]
            simbolos.append(complex(re, im))
            
        elif tipo == "64QAM":
            # 6 bits: niveles -7 a 7
            mapping = {'000':-7, '001':-5, '011':-3, '010':-1, 
                       '110':1, '111':3, '101':5, '100':7}
            re = mapping[grupo[0:3]]
            im = mapping[grupo[3:6]]
            simbolos.append(complex(re, im))
            
    return np.array(simbolos)