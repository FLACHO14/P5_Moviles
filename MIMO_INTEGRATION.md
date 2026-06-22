# Integración MIMO - Documentación Técnica

## Resumen de Cambios

Se ha extendido el simulador OFDM/SC-FDMA existente con capacidades MIMO completas:

### 1. **Nuevas Funciones de Transmisión (ofdm_tx.py)**

#### SFBC (Space-Frequency Block Code) para 2 antenas TX
```python
tx1, tx2, papr_list, n_ofdm, pilots_ant1, pilots_ant2 = sfbc_tx_2ant(
    data_symbols, Nfft, cp_len, sc_map, pilot_value
)
```
- Implementa esquema Alamouti en frecuencia
- Mapeo ortogonal de pilotos por antena
- Retorna señales de ambas antenas y parámetros PAPR

#### SFBC para 3 antenas TX
```python
tx1, tx2, tx3, papr_list, n_ofdm, pilots_1, pilots_2, pilots_3 = sfbc_tx_3ant(...)
```
- Esquema extendido usando bloques de 4 subportadoras
- Estructura: Ant1=[s0,s1,-s2*,-s3*], Ant2=[s1,-s0*,s3*,-s2], Ant3=[s2,s3,s0*,s1*]

#### SFBC para 4 antenas TX
```python
tx1, tx2, tx3, tx4, papr_list, n_ofdm, pilot_indices = sfbc_tx_4ant(...)
```
- Dos bloques Alamouti independientes en paralelo
- Mejor manejo de 4 antenas con baja correlación de código

---

### 2. **Canales MIMO (ofdm_channel.py)**

#### Generación de Canales MISO (Multiple Input Single Output)
```python
channels = generate_miso_channels(NT=2, profile_name='Rayleigh', taps_L=8)
rx_signal, channels = apply_channel_miso(
    tx_signals=[tx1, tx2],
    channels=channels,
    snr_db=15
)
```
- Genera NT canales independientes para NT antenas TX
- Receptor combina linealmente todas las contribuciones
- Modelo realista de 1 RX recibiendo desde múltiples TX

---

### 3. **Recepción MIMO con Decodificación (ofdm_rx.py)**

#### MRC en Dominio de Frecuencia (crítico para SC-FDMA)
```python
X_combined = combine_mrc_frequency(Y_frames_list, H_est_all, sc_map)
```
- Realiza combinación MRC **antes** de ecualizador
- En SC-FDMA: se aplica antes del de-spreading IDFT
- Fórmula: `X[k] = sum_r(H_r*[k]·Y_r[k]) / sum_r(|H_r[k]|²)`

#### Decodificación SFBC 2 Antenas
```python
decoded_symbols = sfbc_decode_2ant(Y_frames_rx, H_est_tx1, H_est_tx2, sc_map, data_idx)
```
- Decodifica estructura Alamouti
- Recupera símbolos originales desde Y recibido

#### Decodificación SFBC 3 y 4 Antenas
```python
decoded_3 = sfbc_decode_3ant(Y_frames_rx, H_est_list, sc_map, data_idx)
decoded_4 = sfbc_decode_4ant(Y_frames_rx, H_est_list, sc_map, data_idx)
```
- Esquemas extendidos con manejo de múltiples bloques

---

### 4. **Nuevas Visualizaciones (ofdm_mimo_utils.py)**

#### BER Comparativo SISO vs SIMO vs MISO
```python
fig = plot_ber_comparison_mimo(results, modulation='16QAM', channel_profile='Rayleigh')
```
- Compara 3 técnicas en la misma figura
- Incluye intervalos de confianza 95%
- Gráficas separadas para QPSK, 16QAM, 64QAM

#### Constelaciones Antes/Después de MRC
```python
fig = plot_constellation_comparison_mrc(data_siso, data_mrc, M_qam=16)
```
- Muestra reducción visual del ruido por diversidad
- Mejora en el espacio de decisión

#### PAPR Comparativo
```python
fig = plot_papr_scfdma_vs_ofdm_mimo(papr_ofdm, papr_scfdma, papr_sfbc)
```
- Demuestra que SC-FDMA mantiene PAPR bajo incluso en MIMO
- Diagramas de caja con promedios

#### Potencia del Canal
```python
fig = plot_channel_power_stability_mimo(H_single, H_combined)
```
- Visualiza suavizado de desvanecimientos profundos
- Muestra ganancia instantánea en dB

---

## Arquitectura de Integración

### Flujo SISO Tradicional
```
Datos → QAM → OFDM TX (IFFT) → Canal → OFDM RX (FFT) → Ecualización → Demod
```

### Flujo SIMO-MRC (Nuevo)
```
Datos → QAM → OFDM TX (IFFT) → Canal×N_RX → OFDM RX (FFT)×N_RX
                                                  ↓
                                        Combinación MRC (Frecuencia)
                                                  ↓
                                           Ecualización ZF → Demod
```

### Flujo MISO-SFBC (Nuevo)
```
Datos → QAM → SFBC Encoding → TX_ANT1, TX_ANT2 (IFFT) → Canal×2
                                                            ↓
                                                    OFDM RX (FFT)
                                                            ↓
                                                    SFBC Decoding
                                                            ↓
                                              Ecualización ZF → Demod
```

### Flujo SC-FDMA + MRC (Óptimo)
```
Datos → QAM → DFT → OFDM TX (IFFT) → Canal×N_RX → OFDM RX (FFT)×N_RX
                                                           ↓
                                                   MRC (Frecuencia)
                                                           ↓
                                                   Ecualización ZF
                                                           ↓
                                                    De-spreading IDFT → Demod
```

**Nota Crítica**: En SC-FDMA, MRC debe aplicarse **en frecuencia ANTES** de la IDFT de de-spreading.

---

## Ejemplos de Uso

### Ejemplo 1: SFBC con 2 Antenas TX
```python
from ofdm_tx import sfbc_tx_2ant
from ofdm_channel import apply_channel_miso, generate_miso_channels

# Transmisión
symbols = qam_mod(bits, M=16)
tx1, tx2, papr_list, n_ofdm, p1, p2 = sfbc_tx_2ant(
    symbols, Nfft, cp_len, sc_map, PILOT_AMPLITUDE
)

# Canal MISO
channels = generate_miso_channels(2, 'Rayleigh', taps_L=8)
rx_signal, _ = apply_channel_miso([tx1, tx2], channels, snr_db=15)

# Recepción
Y_frames = ofdm_rx_block(rx_signal, Nfft, cp_len)
data_eq, _, _ = equalize_with_pilots(Y_frames, sc_map, PILOT_AMPLITUDE, Nfft)

# Decodificación SFBC
from ofdm_rx import sfbc_decode_2ant
data_decoded = sfbc_decode_2ant(Y_frames, H_est_1, H_est_2, sc_map, data_idx)
```

### Ejemplo 2: SIMO-MRC con 3 Antenas RX
```python
from ofdm_channel import generate_mimo_channels, apply_channel_mimo
from ofdm_rx import combine_mrc_frequency

# Transmisión OFDM estándar
tx_signal, _, _ = ofdm_tx_block(symbols, Nfft, cp_len, sc_map, PILOT_AMPLITUDE)

# Canal SIMO (1 TX, 3 RX independientes)
h_channels = generate_mimo_channels(3, 'Rayleigh', taps_L=8)
rx_signals = apply_channel_mimo(tx_signal, h_channels, snr_db=15)

# Recepción en cada antena
Y_frames_list = [ofdm_rx_block(rx, Nfft, cp_len) for rx in rx_signals]

# Estimación de canal por antena
H_est_list = []
for Y in Y_frames_list:
    _, H_est, _ = equalize_with_pilots(Y, sc_map, PILOT_AMPLITUDE, Nfft)
    H_est_list.append(np.array(H_est))

# Combinación MRC en frecuencia
data_combined = combine_mrc_frequency(Y_frames_list, H_est_list, sc_map)

# Puede aplicarse de-spreading SC-FDMA aquí si es necesario
# data_despread = scfdma_despread(data_combined, M_dft)
```

### Ejemplo 3: SC-FDMA + MRC (Óptimo para Uplink)
```python
from ofdm_tx import scfdma_tx_block, get_best_dft_size

# Transmisión SC-FDMA
M_dft = get_best_dft_size(sc_map['n_data'], Nfft)
tx_signal, papr_list, _ = scfdma_tx_block(
    symbols, Nfft, cp_len, sc_map, PILOT_AMPLITUDE, M_dft
)

# Canal SIMO
h_channels = generate_mimo_channels(2, 'Rayleigh', taps_L=8)
rx_signals = apply_channel_mimo(tx_signal, h_channels, snr_db=15)

# Recepción y estimación de canal
Y_frames_list = [ofdm_rx_block(rx, Nfft, cp_len) for rx in rx_signals]
H_est_list = [np.array(equalize_with_pilots(Y, sc_map, PILOT_AMPLITUDE, Nfft)[1])
              for Y in Y_frames_list]

# MRC en frecuencia (antes de de-spreading)
data_combined = combine_mrc_frequency(Y_frames_list, H_est_list, sc_map)

# De-spreading IDFT
from ofdm_rx import scfdma_despread
data_despread = scfdma_despread(data_combined, M_dft)

# Demodulación
bits_recovered = qam_demod(data_despread, M=16)
```

---

## Parámetros de Configuración Recomendados

### Para Simulaciones LTE Standard
```python
BW = 10e6  # 10 MHz (50 RB)
Nfft = 1024
cp_len = 128
n_data = 600
n_pilots = 100

# SC-FDMA
M_dft = 256  # Automático: larger power-of-2 < n_data

# SIMO
n_rx_antennas = [2, 3, 4]  # Comparación

# MISO
n_tx_antennas = 2  # SFBC Alamouti estándar
```

### Para Canales Realistas
```python
channel_profiles = ['Rayleigh', 'Rician', 'EPA', 'EVA', 'ETU']
velocities = [0, 50, 120]  # kmh
snr_range = np.arange(0, 21, 5)  # dB
```

---

## Resultados Esperados

### Ganancia de Diversidad
- **SISO → SIMO (2 RX)**: ~3-4 dB en SNR para mismo BER
- **SISO → SIMO (3 RX)**: ~5-6 dB en SNR
- **SISO → SIMO (4 RX)**: ~8-10 dB en SNR

### PAPR Reduction
- **OFDM**: 8-10 dB (típico)
- **SC-FDMA**: 3-4 dB (incluso con SFBC)
- **SFBC**: Similar a SC-FDMA single-antenna

### Eficiencia Espectral
- **OFDM**: N_data subportadoras de datos
- **SC-FDMA**: M_dft ≤ N_data (menor por guard bands)
- **MISO-SFBC**: Misma como TX single con código espacial

---

## Debugging y Validación

### Verificar Mapeo SFBC
```python
# Las antenas deben tener pilotos ortogonales
assert len(np.intersect1d(pilot_indices_ant1, pilot_indices_ant2)) == 0
```

### Verificar MRC Gain
```python
# Ganancia debe ser proporcional a N_RX
gain_expected = N_RX
gain_observed = np.mean(|H_combined|**2) / np.mean(|H_single|**2)
assert np.abs(gain_observed - gain_expected) < 0.5  # Tolerancia
```

### Validar De-spreading en SC-FDMA
```python
# Datos desprendidos deben ser similares a datos originales
# hasta el ruido y ecualización
correlation = np.abs(np.corrcoef(data_original.real, data_despread.real)[0, 1])
assert correlation > 0.7
```

---

## Archivos Generados/Modificados

| Archivo | Cambios |
|---------|---------|
| `ofdm_tx.py` | +SFBC (2, 3, 4 ant) |
| `ofdm_channel.py` | +MISO channels, +apply_channel_miso |
| `ofdm_rx.py` | +SFBC decoders, +combine_mrc_frequency |
| `ofdm_mimo_utils.py` | **NUEVO**: visualizaciones MIMO |
| `example_mimo_analysis.py` | **NUEVO**: ejemplos de uso |
| `MIMO_INTEGRATION.md` | **NUEVO**: esta documentación |

---

## Próximas Extensiones

1. **Beamforming adaptativo**: Ajustar pesos según estadísticas del canal
2. **Precoding MIMO**: Matriz de precoding antes de TX
3. **Transmisión SFBC-MRC combinada**: TX y RX diversidad simultánea
4. **Estimación de canal adaptativa**: Tracking de canal Doppler rápido
5. **Codificación de canal convolucional**: Agregará ganancia de codificación a la ganancia de diversidad

---

## Referencias

- 3GPP TS 36.211: Physical Channels and Modulation (LTE)
- Alamouti, S.M., "A simple transmit diversity technique for wireless communications" (1998)
- Proakis & Salehi, "Digital Communications" (5th ed., 2007)
- Winters, J.H., "On the capacity of radio communication systems with diversity" (1987)

---

**Última actualización**: 2026-06-21  
**Versión**: 2.0 (MIMO Integrada)
