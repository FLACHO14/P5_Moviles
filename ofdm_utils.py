import numpy as np
from PIL import Image

def gray_encode(n):
    return n ^ (n >> 1)

def gray_decode(g):
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def image_to_bitstream(image_path, max_side=256):
    img = Image.open(image_path).convert('L')
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    img_arr = np.array(img)
    bits = np.unpackbits(img_arr.flatten())
    return bits, img_arr.shape, img_arr  # ahora devuelve también el array de la imagen

def bitstream_to_image(bits, shape):
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - len(bits) % 8), constant_values=0)
    img_bytes = np.packbits(bits)
    if img_bytes.size > shape[0] * shape[1]:
        img_bytes = img_bytes[:shape[0]*shape[1]]
    elif img_bytes.size < shape[0] * shape[1]:
        img_bytes = np.pad(img_bytes, (0, shape[0]*shape[1] - img_bytes.size), constant_values=0)
    img_arr = img_bytes.reshape(shape)
    return Image.fromarray(img_arr, mode='L')

def calculate_ber(bits_tx, bits_rx):
    min_len = min(len(bits_tx), len(bits_rx))
    errors = np.sum(bits_tx[:min_len] != bits_rx[:min_len])
    return errors / min_len if min_len > 0 else 1.0

def calculate_psnr(img_orig, img_rec):
    # Asegurar que ambas imágenes tengan las mismas dimensiones
    if img_orig.shape != img_rec.shape:
        # Redimensionar la imagen original al tamaño de la reconstruida
        from PIL import Image as PILImage
        img_orig_pil = PILImage.fromarray(img_orig.astype('uint8'))
        img_orig_pil = img_orig_pil.resize((img_rec.shape[1], img_rec.shape[0]), PILImage.Resampling.LANCZOS)
        img_orig = np.array(img_orig_pil)
    mse = np.mean((img_orig.astype(float) - img_rec.astype(float))**2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def qam_constellation(M):
    m = int(np.sqrt(M))
    levels = np.arange(-(m-1), m, 2)
    Es = (2 * (m**2 - 1)) / 3
    return levels / np.sqrt(Es)

def bits_to_symbols(bits, M):
    k = int(np.log2(float(M)))
    bits = np.array(bits, dtype=np.uint8)
    if len(bits) % k != 0:
        bits = np.pad(bits, (0, k - len(bits) % k), constant_values=0)
    const = qam_constellation(M)
    kb = k // 2
    symbols = []
    for i in range(0, len(bits), k):
        chunk = bits[i:i+k]
        I_val = int(''.join(map(str, chunk[:kb])), 2)
        Q_val = int(''.join(map(str, chunk[kb:])), 2)
        I_gray = gray_encode(I_val)
        Q_gray = gray_encode(Q_val)
        symbols.append(const[I_gray] + 1j * const[Q_gray])
    return np.array(symbols)

def symbols_to_bits(symbols, M):
    k = int(np.log2(float(M)))
    const = qam_constellation(M)
    kb = k // 2
    bits = []
    for sym in symbols:
        I = np.real(sym)
        Q = np.imag(sym)
        idxI = np.argmin(np.abs(I - const))
        idxQ = np.argmin(np.abs(Q - const))
        I_gray = gray_decode(idxI)
        Q_gray = gray_decode(idxQ)
        I_bits = [(I_gray >> (kb-1-b)) & 1 for b in range(kb)]
        Q_bits = [(Q_gray >> (kb-1-b)) & 1 for b in range(kb)]
        bits.extend(I_bits + Q_bits)
    return np.array(bits, dtype=np.uint8)