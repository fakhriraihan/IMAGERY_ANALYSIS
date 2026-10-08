"""
=============================================================================
CROP IMAGERY GDAL - HYBRID APPROACH (PIXEL & OBJECT-BASED) & OVR PYRAMIDS
=============================================================================
Script ini menggunakan library GDAL murni untuk memproses citra satelit/ortofoto:
1. Mendukung format input GeoTIFF (.tif) maupun ERDAS ECW (.ecw).
2. Menerapkan Hybrid Approach untuk deteksi background:
   - Jika ada Alpha channel: Pixel-based thresholding (Otsu algorithm).
   - Jika RGB murni (tanpa Alpha): Object-based connected components
     agar awan putih / atap bangunan di daratan tidak ikut terhapus.
3. Mengekspor GeoTIFF 4-Band RGBA dengan:
   - BIGTIFF=YES (mendukung ukuran > 4 GB)
   - Tiling 512x512 + Kompresi Deflate + Predictor 2
   - Multithreading NUM_THREADS=ALL_CPUS
4. Membangun piramida sejati (.ovr eksternal) dengan level powers-of-2 lengkap
   sehingga saat dibuka di ArcGIS Pro / QGIS langsung terbuka tanpa peringatan.
=============================================================================
"""

import os
import sys
import glob
import argparse
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm
from osgeo import gdal, osr

# Aktifkan penanganan exception GDAL
gdal.UseExceptions()


def otsu_threshold(arr):
    """Menghitung ambang batas optimal pemisah bimodal (Otsu algorithm)."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b, w_b, current_max, threshold = 0, 0, 0, 128
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        v = w_b * w_f * ((m_b - m_f) ** 2)
        if v > current_max:
            current_max, threshold = v, i
    return threshold


def auto_detect_hybrid_profile(ds, sample_factor=64):
    """
    Menganalisis dataset GDAL secara cepat pada skala thumbnail:
    - Memeriksa ketersediaan kanal Alpha
    - Menghitung nilai median tepi luar (perimeter)
    - Menentukan strategi terbaik (Pixel-based vs Object-based)
    """
    width = ds.RasterXSize
    height = ds.RasterYSize
    count = ds.RasterCount
    
    thumb_w = max(1, width // sample_factor)
    thumb_h = max(1, height // sample_factor)
    
    # Baca thumbnail ke NumPy array: shape (bands, thumb_h, thumb_w)
    thumb = ds.ReadAsArray(0, 0, width, height, buf_xsize=thumb_w, buf_ysize=thumb_h)
    if count == 1:
        thumb = np.expand_dims(thumb, axis=0)
    
    # Ambil piksel keliling tepi (perimeter 5 piksel terluar)
    top = thumb[:, :5, :]
    bot = thumb[:, -5:, :]
    left = thumb[:, :, :5]
    right = thumb[:, :, -5:]
    borders = np.concatenate([
        top.reshape(count, -1),
        bot.reshape(count, -1),
        left.reshape(count, -1),
        right.reshape(count, -1)
    ], axis=1)
    
    border_vals = np.round(np.median(borders, axis=1)).astype(int)
    
    has_alpha = False
    alpha_thresh = None
    
    # Evaluasi Band 4 sebagai kanal Alpha
    if count >= 4:
        if border_vals[3] <= 25 and np.percentile(thumb[3], 85) > 150:
            has_alpha = True
            alpha_thresh = otsu_threshold(thumb[3])
    
    profile = {
        'width': width,
        'height': height,
        'count': count,
        'border_values': border_vals.tolist(),
        'has_alpha_channel': has_alpha,
        'alpha_threshold': alpha_thresh
    }
    
    if has_alpha:
        # STRATEGI 1: PIXEL-BASED PADA ALPHA CHANNEL (OTSU)
        profile['approach'] = 'Pixel-Based (Alpha Otsu Thresholding)'
        profile['rule'] = f'Band 4 (Alpha) >= {alpha_thresh}'
        profile['mask_fn'] = lambda arr: (arr[3] >= alpha_thresh)
    else:
        # STRATEGI 2: OBJECT-BASED PADA RGB (BORDER CONNECTIVITY)
        # Menjaga awan putih / atap bangunan di daratan agar tidak terhapus
        profile['approach'] = 'Object-Based (Border-Connected Components)'
        is_white = bool(np.all(border_vals[:3] >= 240))
        if is_white:
            profile['rule'] = 'Object-Based: Outer White Border Only (Awan Daratan Terlindungi)'
            def get_mask_white(arr):
                white_cand = (arr[0] >= 245) & (arr[1] >= 245) & (arr[2] >= 245)
                labeled, _ = ndi.label(white_cand)
                border_labels = set(labeled[0, :]).union(set(labeled[-1, :])).union(set(labeled[:, 0])).union(set(labeled[:, -1]))
                border_labels.discard(0)
                is_outer_bg = np.isin(labeled, list(border_labels))
                return ~is_outer_bg
            profile['mask_fn'] = get_mask_white
        else:
            profile['rule'] = 'Object-Based: Outer Dark Border Only'
            def get_mask_dark(arr):
                dark_cand = (arr[0] <= 10) & (arr[1] <= 10) & (arr[2] <= 10)
                labeled, _ = ndi.label(dark_cand)
                border_labels = set(labeled[0, :]).union(set(labeled[-1, :])).union(set(labeled[:, 0])).union(set(labeled[:, -1]))
                border_labels.discard(0)
                is_outer_bg = np.isin(labeled, list(border_labels))
                return ~is_outer_bg
            profile['mask_fn'] = get_mask_dark
            
    return profile


def process_raster_gdal(input_path, output_dir, chunk_size=4096):
    """
    Memproses citra menggunakan GDAL murni:
    1. Buka dataset (mendukung .tif maupun .ecw).
    2. Deteksi otomatis nilai background via Hybrid Approach.
    3. Tulis GeoTIFF RGBA 4-Band per blok chunk dengan BigTIFF.
    4. Bangun piramida .ovr eksternal lengkap untuk ArcGIS dan QGIS.
    """
    file_name = os.path.basename(input_path)
    base_name = os.path.splitext(file_name)[0]
    out_file = os.path.join(output_dir, f"{base_name}.tif")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 75)
    print(f"MEMPROSES CITRA GDAL: {file_name}")
    print("=" * 75)
    
    # 1. Buka citra dengan GDAL
    in_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if not in_ds:
        raise RuntimeError(f"Gagal membuka file citra: {input_path}")
        
    width = in_ds.RasterXSize
    height = in_ds.RasterYSize
    bands = in_ds.RasterCount
    driver_name = in_ds.GetDriver().ShortName
    
    print(f"Driver Input        : {driver_name}")
    print(f"Dimensi             : {width:,} x {height:,} piksel (~{(width*height)/1e9:.2f} Miliar piksel)")
    print(f"Jumlah Band         : {bands}")
    
    # 2. Analisis Hybrid Approach
    profile = auto_detect_hybrid_profile(in_ds)
    print(f"Nilai Border Tepi   : {profile['border_values']}")
    print(f"Pendekatan Terpilih : {profile['approach']}")
    print(f"Aturan Masking      : {profile['rule']}")
    
    # 3. Siapkan driver dan opsi pembuatan GeoTIFF
    gtiff_driver = gdal.GetDriverByName("GTiff")
    creation_options = [
        "BIGTIFF=YES",            # Mendukung ukuran file raksasa (> 4 GB)
        "TILED=YES",              # Format tiling
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "COMPRESS=DEFLATE",       # Kompresi Deflate lossless
        "PREDICTOR=2",            # Optimal untuk citra visual
        "PHOTOMETRIC=RGB",
        "NUM_THREADS=ALL_CPUS"    # Multithreading CPU penuh
    ]
    
    # Target output selalu 4 band (RGBA)
    out_bands = 4
    out_ds = gtiff_driver.Create(out_file, width, height, out_bands, gdal.GDT_Byte, creation_options)
    
    # Salin Geotransform & Proyeksi CRS dari file sumber
    geo_transform = in_ds.GetGeoTransform()
    if geo_transform:
        out_ds.SetGeoTransform(geo_transform)
        
    projection = in_ds.GetProjection()
    if projection:
        out_ds.SetProjection(projection)
        
    # Tetapkan interpretasi warna resmi RGBA
    out_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    out_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    out_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    
    # 4. Pemrosesan per blok/chunk agar hemat memori RAM
    n_chunks_x = int(np.ceil(width / chunk_size))
    n_chunks_y = int(np.ceil(height / chunk_size))
    total_chunks = n_chunks_x * n_chunks_y
    
    print(f"\nMenulis ke GeoTIFF  : {out_file}")
    print(f"Total Chunk         : {total_chunks} ({chunk_size}x{chunk_size} piksel per chunk)")
    
    mask_fn = profile['mask_fn']
    
    with tqdm(total=total_chunks, desc="Menulis Chunk RGBA") as pbar:
        for y_offset in range(0, height, chunk_size):
            h = min(chunk_size, height - y_offset)
            for x_offset in range(0, width, chunk_size):
                w = min(chunk_size, width - x_offset)
                
                # Baca blok dari input dataset: shape (bands, h, w)
                block_data = in_ds.ReadAsArray(x_offset, y_offset, w, h)
                if bands == 1:
                    block_data = np.expand_dims(block_data, axis=0)
                
                # Jika input 3 band (RGB), tambahkan band ke-4 (Alpha)
                if block_data.shape[0] == 3:
                    alpha_init = np.full((1, h, w), 255, dtype=np.uint8)
                    block_data = np.concatenate([block_data, alpha_init], axis=0)
                
                # Dapatkan mask valid data
                is_valid = mask_fn(block_data)
                
                # Terapkan transparansi pada Band 4 (255 = Citra, 0 = Transparan)
                block_data[3] = np.where(is_valid, 255, 0).astype(np.uint8)
                
                # Nolkan nilai RGB di background agar bersih tanpa sisa
                block_data[0, ~is_valid] = 0
                block_data[1, ~is_valid] = 0
                block_data[2, ~is_valid] = 0
                
                # Tulis tiap band ke output dataset
                for b in range(4):
                    out_ds.GetRasterBand(b + 1).WriteArray(block_data[b], x_offset, y_offset)
                    
                pbar.update(1)
                
    out_ds.FlushCache()
    print("Penulisan data citra selesai!")
    
    # 5. Membangun Piramida Sejati (.ovr) untuk ArcGIS & QGIS
    print("\nMembangun piramida eksternal (.ovr) powers-of-2 lengkap untuk ArcGIS & QGIS...")
    levels = []
    factor = 2
    min_dim = min(width, height)
    while min_dim // factor >= 128:
        levels.append(factor)
        factor *= 2
        
    print(f"Level piramida yang dibangun : {levels}")
    
    gdal.SetConfigOption("TIFF_USE_OVR", "YES")
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
    
    out_ds.BuildOverviews("AVERAGE", levels)
    out_ds.FlushCache()
    
    # Tutup dataset untuk finalisasi file di disk
    out_ds = None
    in_ds = None
    
    out_size_gb = os.path.getsize(out_file) / (1024 ** 3)
    ovr_file = out_file + ".ovr"
    
    print("\n" + "=" * 75)
    print("PEMROSESAN BERHASIL SELESAI!")
    print(f"File Output GeoTIFF : {out_file} ({out_size_gb:.2f} GB)")
    if os.path.exists(ovr_file):
        ovr_size_mb = os.path.getsize(ovr_file) / (1024 ** 2)
        print(f"File Piramida (.ovr): {ovr_file} ({ovr_size_mb:.1f} MB)")
        print("Status: Piramida eksternal siap. ArcGIS & QGIS tidak akan meminta build pyramids lagi.")
    print("=" * 75)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="GDAL Hybrid Imagery Background Removal & Pyramid Builder")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path ke file imagery (.tif atau .ecw). Jika kosong, mencari otomatis di folder default.")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Folder tujuan output. Default: D:\\JOB\\TASK\\DATA\\IMAGERY\\Kalbar April 2026\\Nodata")
    parser.add_argument("--chunk_size", "-c", type=int, default=4096,
                        help="Ukuran chunk pemrosesan blok (default: 4096)")
    args = parser.parse_args()
    
    # Path default jika tidak ditentukan
    default_dir = r"D:\JOB\TASK\DATA\IMAGERY\Kalbar April 2026"
    default_out = os.path.join(default_dir, "Nodata")
    
    output_dir = args.output_dir if args.output_dir else default_out
    
    if args.input:
        input_file = args.input
    else:
        # Cari file .ecw atau .tif yang tersedia di direktori data
        candidates = sorted(glob.glob(os.path.join(default_dir, "*.ecw*")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(default_dir, "tif", "*.tif*")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(default_dir, "*.tif*")))
            
        if not candidates:
            print(f"Tidak ditemukan file .ecw atau .tif di {default_dir}")
            sys.exit(1)
            
        input_file = candidates[0]
        
    process_raster_gdal(input_file, output_dir, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()

