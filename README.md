# AIR HouseGuard

Sistem pemantau udara rumah yang membandingkan kondisi **halaman** dan **dalam rumah**, mengenali asal polusi dan sumber kelembapan, lalu memberi saran tindakan (buka/tutup/tunda jendela, memasak, menjemur, dll.) **sesuai peralatan yang dimiliki rumah**. Setiap 7 hari sistem mengevaluasi apakah saran yang dijalankan benar-benar menurunkan polusi dan jam lembap.

Dokumen ini adalah panduan penerapan untuk tiga kode yang sudah ada, ditambah bagian yang **masih harus dibangun** agar sesuai dengan dokumen rancangan proyek (AIRIMPACT / AIR HouseGuard).

| Berkas | Fungsi | Dijalankan |
|---|---|---|
| `deteksi-api-asap.ipynb` | Melatih/memuat model YOLO **asap & api** → `houseguard_model_best.pt` | Sekali (Kaggle GPU) |
| `deteksi_kendaraan.py` | Menghitung kendaraan per jenis dari kamera halaman + memantau asap/api → CSV | Terus-menerus |
| `olahdata.ipynb` | Membersihkan & merekap data kendaraan (harian, jam sibuk, komposisi) | Berkala / kalibrasi |

> **Status jujur: apa yang sudah ada dan apa yang belum.**
> Ketiga kode hanya mencakup **sisi kamera** (kendaraan + asap/api) dan analisis datanya. Bagian lain dari rancangan — sensor PM/suhu/kelembapan, kalibrasi dua alat, data cuaca, onboarding inventaris rumah, deteksi kejadian, keputusan jendela, notifikasi, evaluasi 7 hari, dan halaman web — **belum ada kodenya**. Bagian B di bawah memberi spesifikasi dan contoh kode awal untuk membangunnya. Firmware sensor tidak dibahas; hanya kontrak datanya (Langkah 4).
> Semua ambang pada [Langkah 7](#langkah-7--ambang-batas) memakai nilai acuan resmi bila ada, dan **ditandai [Asumsi]** bila merupakan nilai awal yang harus dikalibrasi dengan data rumah Anda.

---

## 1. Peta Rancangan → Komponen

Alur 8 langkah dari dokumen rancangan dan padanannya di repositori ini:

| # | Alur sistem (dokumen) | Komponen | Status |
|---|---|---|---|
| 1 | Pemasangan & kalibrasi dua alat | `kalibrasi.py` (Langkah 4) | Perlu dibangun |
| 2 | Onboarding: inventaris, kebiasaan, penghuni rentan | `inventaris.json` + form di aplikasi (Langkah 6) | Perlu dibangun |
| 3 | Pengumpulan data (sensor tiap menit → rata-rata 15 menit & jam; cuaca per jam) | Endpoint ingest + `cuaca.py` (Langkah 4–5) | Perlu dibangun |
| 3a | Kendaraan per jenis per jam + deteksi asap di halaman | `deteksi_kendaraan.py` | **Ada** |
| 3b | Model asap/api | `deteksi-api-asap.ipynb` | **Ada** |
| 4 | Deteksi kejadian (jalan, dalam rumah, lembap, asap, hujan) | `keputusan.py` (Langkah 8) | Perlu dibangun |
| 5 | Keputusan ventilasi (buka/tutup/tunda) | `keputusan.py` (Langkah 8) | Perlu dibangun |
| 6 | Notifikasi saran (maks. 3/hari, kecuali bahaya) | `keputusan.py` + aplikasi (Langkah 9) | Perlu dibangun |
| 7 | Konfirmasi tindakan (sudah/belum) | Tabel `log_tindakan` (Langkah 9) | Perlu dibangun |
| 8 | Evaluasi 7 hari | `keputusan.py` (Langkah 9) | Perlu dibangun |
| – | Rekap & kalibrasi ambang kendaraan | `olahdata.ipynb` + `kalibrasi.py` | **Ada** (sebagian) |

### Arsitektur

```
 Alat HALAMAN                                   Alat DALAM RUMAH
 ┌──────────────────────────────┐               ┌──────────────────────┐
 │ Kamera → deteksi_kendaraan.py│               │ PM2.5/PM10, suhu, RH │
 │  • kendaraan per jenis       │               │ (opsional CO2)       │
 │  • asap/api (model .pt)      │               └──────────┬───────────┘
 │ PM2.5/PM10, suhu, RH         │                          │ POST /api/ingest/dalam
 └───────┬──────────────┬───────┘                          │
         │ CSV          │ POST /api/ingest/halaman         │
         ▼              ▼                                  ▼
   hasil_deteksi.csv   sensor_halaman.csv            sensor_dalam.csv
   kejadian_asap.csv          │                           │
         └────────────────────┴───────────┬───────────────┘
                                          ▼
   Open-Meteo (cuaca, PM wilayah) ──►  app.py + keputusan.py  ◄── inventaris.json (onboarding)
                                          │
                       ┌──────────────────┼─────────────────────┐
                       ▼                  ▼                     ▼
               status jendela      saran sesuai alat      laporan 7 hari
                       └──────────────► halaman web / notifikasi ◄──── konfirmasi sudah/belum
```

---

## 2. Prasyarat & Struktur Folder

- Python 3.10+, disarankan virtual environment.
- **GPU NVIDIA + CUDA** sangat disarankan untuk `deteksi_kendaraan.py`; di CPU FPS rendah (script memberi peringatan bila FPS < 8).
- Akun Kaggle untuk training model asap/api.

```bash
pip install ultralytics==8.3.143 opencv-python numpy torch pandas flask requests
pip install kagglehub pyyaml matplotlib jupyter   # notebook asap/api & olahdata
```

> Versi `ultralytics` disamakan dengan notebook Kaggle (8.3.143) agar model hasil training pasti bisa dibuka di laptop.

```
airhouseguard/
├── deteksi_kendaraan.py
├── keputusan.py            # ambang, kejadian, keputusan ventilasi, saran, evaluasi (Langkah 7–9)
├── kalibrasi.py            # kalibrasi alat & ambang kendaraan (Langkah 4, 7)
├── cuaca.py                # Open-Meteo (Langkah 5)
├── app.py                  # backend web (Langkah 10)
├── static/index.html
├── inventaris.json         # jawaban onboarding (Langkah 6)
├── ambang_lalin.json       # dihasilkan kalibrasi.py
├── models/
│   ├── houseguard_model_best.pt
│   └── yolo11s.pt          # diunduh otomatis sekali
├── data/data_bersih.csv    # data historis untuk olahdata.ipynb
├── hasil_deteksi.csv       # dibuat deteksi_kendaraan.py
├── kejadian_asap.csv       # dibuat deteksi_kendaraan.py
├── sensor_halaman.csv      # dibuat endpoint ingest
└── sensor_dalam.csv        # dibuat endpoint ingest
```

---

# Bagian A — Yang Sudah Ada

## Langkah 1 — Model asap & api (`deteksi-api-asap.ipynb`)

Melatih **YOLOv8n** dengan dataset Kaggle `sayedgamal99/smoke-fire-detection-yolo` (kelas `smoke`, `fire`; 50 epoch; `imgsz=640`) dan menyimpan `models/houseguard_model_best.pt`.

1. Unggah notebook ke Kaggle, aktifkan **Settings → Accelerator → GPU T4**.
2. Jalankan semua sel. Bila model sudah ada, training **dilewati**; paksa ulang dengan `LATIH_ULANG = True`.
3. Unduh `houseguard_model_best.pt` (link di notebook) ke folder `models/` laptop. Opsional: isi `KAGGLE_USERNAME` di sel "Pengaturan folder" agar model diambil otomatis dari Kaggle Models.
4. Training di CPU ± 2,5 jam **per epoch**, sehingga dinonaktifkan bawaan (`IZINKAN_TRAINING_CPU = False`).
5. Ubah path `MODELS_DIR` di notebook dan `DEFAULT_MODELS_DIR` di `deteksi_kendaraan.py` dari `C:\Users\NADIA\...` ke folder Anda.

Notebook juga berisi evaluasi: grafik training, skor test set, contoh prediksi, uji ketahanan (blur, gelap, gelap+noise, dan gelap yang dicerahkan dengan mode malam), serta tes pada satu frame CCTV (sering gagal di Kaggle karena jaringan — jalankan di laptop). Tanpa kebakaran, hasil yang benar adalah **tidak ada deteksi**.

## Langkah 2 — Deteksi kendaraan + asap/api (`deteksi_kendaraan.py`)

```bash
python deteksi_kendaraan.py \
  --url "https://alamat-stream-anda/index.m3u8" \
  --camera-id cam1 \
  --models-dir models \
  --smoke-model models/houseguard_model_best.pt \
  --no-show
```

Uji dahulu dengan file video: `--url "data/contoh.mp4"`. `yolo11s.pt` diunduh otomatis sekali ke `--models-dir`.

| Keluaran | Isi |
|---|---|
| `hasil_deteksi.csv` | Satu baris per kendaraan: `camera_id, object, track_id, detection_timestamp, timestamp` (waktu **UTC**) |
| `kejadian_asap.csv` | `camera_id, status, jenis, conf_maks, waktu_mulai, waktu_selesai, durasi_detik, foto`; status `mulai` → `selesai`/`dihentikan` |
| `foto_asap/` | Foto bukti saat kejadian dimulai (lihat [catatan privasi](#12-privasi)) |

Jenis di kolom `object`: `mobil`, `motor`, `bus`, `truk`, `sepeda`.

| Opsi penting | Keterangan |
|---|---|
| `--mode masuk` (bawaan) / `--mode garis --set-line` | Hitung saat terlacak stabil & bergerak / saat melewati garis (klik 2 titik di frame pertama) |
| `--imgsz 1280` | Jauh lebih baik untuk motor kecil daripada 640 |
| `--device auto`, `--skip N` | GPU bila ada; proses 1 dari tiap N frame bila lambat |
| `--min-move 0.5`, `--heavy-conf 0.5` | Abaikan objek diam (gerobak, mobil parkir); truk/bus butuh confidence tinggi |
| `--night auto`, `--night-thresh`, `--night-target` | Pencerahan otomatis (gamma + CLAHE) saat gelap |
| `--smoke-every 1.0`, `--smoke-window 5`, `--smoke-min-hits 3` | Cek asap tiap 1 detik; kejadian nyata bila ≥ 3 dari 5 pengecekan terakhir positif; selesai bila 5 pengecekan terakhir bersih |
| `--smoke-keep-vehicle` | Bawaan: asap yang menempel di kotak kendaraan (kemungkinan knalpot) **diabaikan** |
| `--no-smoke` | Matikan deteksi asap/api |

File CSV bersifat **append**, aman dijalankan ulang. Hentikan dengan `Ctrl+C`.

## Langkah 3 — Rekap & analisis data (`olahdata.ipynb`)

Bekerja pada **data historis** `data/data_bersih.csv` (181.420 baris, dua kamera, 25–31 Maret 2026): memperbaiki kolom waktu, konversi UTC → WIB, rekap harian, hari & jam tersibuk, komposisi jenis, pemeriksaan kualitas, dan ekspor `data_cctv_clean_v1.csv`.

Kegunaannya di sistem ini: **pola jam sibuk** dan sebaran jumlah kendaraan per jam dipakai untuk menetapkan ambang "sepi/ramai" (Langkah 7.3).

| Hal | `data_bersih.csv` (historis) | `hasil_deteksi.csv` (script) |
|---|---|---|
| Kategori berat | `bus_truk` (digabung) | `bus` dan `truk` terpisah |
| Kolom `location` | Ada | Tidak ada |
| Zona waktu | Dikonversi ke WIB di notebook | **UTC** — konversi sendiri |
| `track_id` | — | Mulai dari 1 lagi tiap script dijalankan ulang |

Script sudah menulis **satu baris per kendaraan**, jadi rekap cukup menghitung baris. Jangan memakai `nunique(track_id)` lintas beberapa kali jalan: ID yang sama bisa muncul lagi di sesi lain dan hanya terhitung sekali.

---

# Bagian B — Yang Perlu Dibangun

## Langkah 4 — Sensor, kontrak data, dan kalibrasi awal

### 4a. Kontrak data (firmware cukup mengirim ini)

| Alat | Kolom | Frekuensi | Dikirim ke |
|---|---|---|---|
| Halaman | `pm25, pm10, suhu, rh` | tiap menit | `POST /api/ingest/halaman` |
| Dalam rumah | `pm25, pm10, suhu, rh, co2` (co2 opsional) | tiap menit | `POST /api/ingest/dalam` |

Dirata-ratakan per **15 menit** dan per **jam** di sisi server (Langkah 8). Penempatan alat dalam rumah: ruang yang paling lama ditempati (ruang keluarga/kamar tidur), tinggi sekitar 1–1,5 m, jauh dari kompor, kamar mandi, dan jendela. Alat halaman memproses video **di dalam alat**; yang disimpan hanya angka.

### 4b. Kalibrasi berdampingan (24–48 jam)

Letakkan kedua alat berdampingan, lalu cari koreksi linear agar alat dalam rumah "sejajar" dengan alat halaman. Pembacaan PM juga harus dikoreksi terhadap kelembapan karena sensor PM murah cenderung membaca lebih tinggi saat udara lembap — **gunakan model koreksi yang sesuai tipe sensor Anda** (dari datasheet/literatur untuk tipe itu); koefisien tipe sensor lain tidak boleh dipakai.

```python
# kalibrasi.py
import numpy as np
import pandas as pd


def kalibrasi_berdampingan(halaman: pd.Series, dalam: pd.Series):
    """Kedua alat berdampingan. Hasil: dalam_terkoreksi = a * dalam + b  ≈ halaman."""
    d = pd.concat([halaman, dalam], axis=1, keys=["ref", "alat"]).dropna()
    a, b = np.polyfit(d["alat"], d["ref"], 1)
    sisa = d["ref"] - (a * d["alat"] + b)
    r2 = np.corrcoef(d["alat"], d["ref"])[0, 1] ** 2
    return {"a": float(a), "b": float(b), "r2": float(r2), "mae": float(sisa.abs().mean())}

# Jalankan terpisah untuk pm25, pm10, suhu, rh. Simpan (a, b) per variabel,
# lalu pakai di endpoint ingest (KOREKSI pada app.py).
```

**Kriteria lolos** (**[Asumsi]**, sesuaikan spesifikasi sensor): PM → `r2 ≥ 0.9` **atau** `mae ≤ 3 µg/m³` (bila rentang PM selama kalibrasi sempit, r² bisa rendah walau alat baik); suhu → `mae ≤ 0.5 °C`; RH → `mae ≤ 2 poin RH`. Bila tidak lolos, perpanjang kalibrasi atau ganti alat.

## Langkah 5 — Cuaca & kualitas udara wilayah (`cuaca.py`)

Sumber per jam: **Open-Meteo** (cuaca, prakiraan hujan, PM wilayah). Dokumen rancangan juga menyebut BMKG dan OpenAQ; keduanya dapat ditambahkan mengikuti dokumentasi data terbukanya (tidak dicontohkan di sini).

```python
# cuaca.py
import os
import requests

LAT = float(os.environ["RUMAH_LAT"])   # koordinat rumah
LON = float(os.environ["RUMAH_LON"])


def _ambil(url, hourly):
    r = requests.get(url, params={"latitude": LAT, "longitude": LON, "hourly": hourly,
                                  "timezone": "Asia/Jakarta", "forecast_days": 2}, timeout=15)
    r.raise_for_status()
    return r.json()["hourly"]


def cuaca_sekarang(jam_ke_depan=1):
    """Kembalikan ramalan jam ini + beberapa jam berikutnya (untuk 'jam aman terdekat')."""
    c = _ambil("https://api.open-meteo.com/v1/forecast",
               "temperature_2m,relative_humidity_2m,precipitation,precipitation_probability,"
               "wind_speed_10m,wind_direction_10m")
    q = _ambil("https://air-quality-api.open-meteo.com/v1/air-quality", "pm2_5,pm10")
    from datetime import datetime
    kunci = datetime.now().strftime("%Y-%m-%dT%H:00")
    i = c["time"].index(kunci)
    return {
        "hujan_sekarang": (c["precipitation"][i] or 0) > 0,
        "prob_hujan_1jam": c["precipitation_probability"][i + jam_ke_depan] or 0,
        "pm25_wilayah": q["pm2_5"][i],
        "pm10_wilayah": q["pm10"][i],
        "ramalan": [(c["time"][j][11:16], c["precipitation_probability"][j] or 0)
                    for j in range(i, min(i + 12, len(c["time"])))],
    }
```

> PM dari Open-Meteo adalah **model regional** (resolusi ± 11 km), bukan pengukuran di tepi jalan. Gunakan hanya sebagai pembanding latar; keputusan PM halaman memakai **sensor halaman**.

## Langkah 6 — Onboarding: inventaris rumah (`inventaris.json`)

Aplikasi menanyakan peralatan dan kondisi rumah **sebelum** saran apa pun. Jawaban disimpan sebagai berikut (nilai sesuai pilihan jawaban di dokumen rancangan):

```json
{
  "exhaust_dapur": "berfungsi",
  "exhaust_kamar_mandi": "tidak ada",
  "jendela_sisi_non_jalan": true,
  "ac": "dry",
  "dehumidifier": false,
  "hepa": false,
  "kipas": true,
  "rokok_dalam": false,
  "obat_nyamuk_bakar": true,
  "noda_jamur": [{"ruang": "kamar belakang", "luas_m2": 0.5}],
  "penghuni_rentan": ["balita"],
  "kebiasaan": {"masak": ["06:00-07:00", "17:00-18:00"], "tidur": "21:30-05:30", "jemur_dalam": true}
}
```

Pilihan: `exhaust_*` ∈ {`berfungsi`, `rusak`, `tidak ada`}; `ac` ∈ {`dry`, `tanpa_dry`, `tidak`}; `penghuni_rentan` ⊆ {`balita`, `lansia`, `ibu_hamil`, `asma_paru`}. Kosong = tidak ada penghuni rentan. Jam masak/tidur dipakai untuk mencocokkan lonjakan PM dan kelembapan dengan aktivitas (Langkah 8).

## Langkah 7 — Ambang batas

Prinsip: **nilai acuan resmi dipakai langsung**; nilai yang tidak punya acuan ditandai **[Asumsi]** dan harus dikalibrasi. Dua profil: **standar** dan **ketat**. Profil **ketat** dipakai bila ada penghuni rentan (balita, lansia, ibu hamil, asma/penyakit paru) atau noda jamur.

### 7.1 Kelembapan

| Parameter | Standar | Ketat | Dasar |
|---|---|---|---|
| Kelembapan relatif dalam rumah dianggap **tinggi** | > 60 % | > 55 % **[Asumsi]** | Anjuran US EPA di dokumen: di bawah 60 % (ideal 30–50 %) |
| Metrik utama laporan | jam/hari dengan RH > 60 % | sama | Dokumen rancangan |
| Perbandingan dalam vs luar | **Kelembapan absolut (g/m³)**, bukan RH % | sama | Dokumen rancangan |
| Buka jendela bila | `AH_luar < AH_dalam − 1,0 g/m³` | sama | Margin 1,0 g/m³ **[Asumsi]**: sensor RH murah ± 2–3 % RH ≈ ± 0,8 g/m³ pada 28 °C; margin mencegah buka-tutup bolak-balik |
| Hujan → tutup | `hujan sekarang` atau `peluang hujan 1 jam ke depan ≥ 50 %` | sama | Dokumen: "sedang atau diprakirakan hujan dalam 1 jam"; 50 % **[Asumsi]** |

Rumus kelembapan absolut (Magnus), untuk suhu `T` °C dan RH %:

```python
import math

def kelembapan_absolut(suhu_c, rh):
    """Gram uap air per m³ udara."""
    es = 6.112 * math.exp(17.67 * suhu_c / (suhu_c + 243.5))   # tekanan uap jenuh, hPa
    return 216.7 * (rh / 100.0 * es) / (273.15 + suhu_c)
```

### 7.2 Partikulat (PM2.5 & PM10)

Acuan yang dipakai:

| Acuan | PM2.5 | PM10 | Catatan |
|---|---|---|---|
| WHO 2021, 24 jam — **AQG** | 15 | 45 | µg/m³; pembanding utama (dokumen rancangan) |
| WHO 2021, 24 jam — interim target 4 | 25 | 50 | |
| WHO 2021, 24 jam — interim target 3 | 37,5 | 75 | |
| Baku mutu nasional (PP 22/2021), 24 jam | 55 | 75 | |
| ISPU (Permen LHK P.14/2020; BMKG 8/2022) | Baik 0–15,5 · Sedang 15,6–55,4 · Tidak sehat 55,5–150,4 · Sangat tidak sehat 150,5–250,4 · Berbahaya > 250,4 | Baik 0–50 · Sedang 51–150 · Tidak sehat 151–350 · Sangat tidak sehat 351–420 · Berbahaya > 420 | Dipakai untuk **komunikasi** ke pengguna |

Aturan operasional (rata-rata **1 jam** dari sensor **halaman**):

| Parameter | Standar | Ketat | Dasar |
|---|---|---|---|
| PM2.5 halaman **boleh buka jendela** bila ≤ | 25 µg/m³ | 15 µg/m³ | WHO interim target 4 / AQG. Nilai WHO adalah rata-rata 24 jam; di sini dipakai sebagai pemicu per jam yang konservatif **[Asumsi]** |
| PM10 halaman **boleh buka jendela** bila ≤ | 50 µg/m³ | 45 µg/m³ | WHO interim target 4 / AQG (24 jam) **[Asumsi]** |
| Status **berbahaya** (di luar kuota notifikasi) | PM2.5 halaman atau dalam > 55,4 µg/m³ (1 jam) | sama | Batas atas kategori "Sedang" ISPU; ≈ baku mutu 55 |
| Laporan 7 hari: jam PM2.5 dalam rumah **di atas** | 15 µg/m³ | sama | Dokumen rancangan (WHO 2021) |
| Lonjakan PM | lihat 7.4 | sama | — |

> Wilayah dengan lalu lintas padat sering melewati 25 µg/m³. Bila demikian, status jendela akan lebih sering **tunda** — itu perilaku yang diharapkan, bukan galat. Ambang buka dapat dilonggarkan satu tingkat (mis. 15 → 25) hanya bila penghuni tidak rentan.

### 7.3 Kendaraan (kalibrasi dari data sendiri)

Jumlah kendaraan per jam bergantung pada **sudut kamera, mode hitung, dan jalan**, sehingga **tidak ada angka universal**. Ambang ditetapkan per `camera_id` dari data rumah itu sendiri:

| Ambang | Cara menentukan | Dipakai untuk |
|---|---|---|
| `sepi` | Persentil 25 kendaraan/jam | Info "jalan sepi" |
| `ramai` | Persentil 75 kendaraan/jam | Memicu **tunda** jendela & klasifikasi "polusi dari jalan" |
| `puncak` | Persentil 90 kendaraan/jam | Penandaan jam sibuk di laporan |
| `berat_tinggi` | Persentil 90 (bus + truk)/jam | Penanda tambahan emisi berat |

Syarat dan validasi:

1. **Data minimal 7 hari** kamera berjalan **kontinu** (jam tanpa rekaman jangan dihitung nol). Sebelum terpenuhi, kendaraan hanya ditampilkan sebagai informasi; keputusan jendela bertumpu pada **PM halaman** (`lalin_ramai = False`).
2. **Uji korelasi:** hitung korelasi Spearman antara kendaraan per 15 menit dan PM2.5 halaman dengan jeda 0–60 menit. **Pakai kendaraan sebagai pemicu hanya bila ρ ≥ 0,3** (**[Asumsi]**); bila lebih lemah, jalan bukan sumber utama PM di titik itu dan kendaraan cukup jadi informasi.
3. **Penajaman empiris (opsional):** kelompokkan kendaraan/jam ke dalam desil; ambang `ramai` ≈ batas bawah desil terendah yang median PM2.5 halamannya melewati ambang buka (7.2).
4. Ulangi tiap bulan atau bila kamera dipindah/diarahkan ulang.

```python
# kalibrasi.py (lanjutan)
import json
import pandas as pd


def ambang_kendaraan(csv="hasil_deteksi.csv", camera_id="cam1", min_hari=7, simpan="ambang_lalin.json"):
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    df = df[df["camera_id"] == camera_id].copy()
    df["waktu"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Jakarta")

    per_jam = df.groupby(df["waktu"].dt.floor("h")).size().asfreq("h", fill_value=0)
    berat = (df[df["object"].isin(["bus", "truk"])]
             .groupby(df["waktu"].dt.floor("h")).size().reindex(per_jam.index, fill_value=0))
    if len(per_jam) < 24 * min_hari:
        return None   # data belum cukup: jangan pakai kendaraan sebagai pemicu

    sepi, ramai, puncak = per_jam.quantile([0.25, 0.75, 0.90])
    hasil = {"camera_id": camera_id, "sepi": float(sepi), "ramai": float(ramai),
             "puncak": float(puncak), "berat_tinggi": float(berat.quantile(0.90)),
             "pola_jam": per_jam.groupby(per_jam.index.hour).mean().round(1).to_dict()}
    with open(simpan, "w") as f:
        json.dump(hasil, f, indent=2)
    return hasil


def korelasi_lalin_pm(lalin_15m: pd.Series, pm25_halaman_15m: pd.Series, maks_lag=4):
    """Spearman antara kendaraan (t) dan PM2.5 halaman (t + lag). lag dalam bin 15 menit."""
    return pd.Series({lag * 15: lalin_15m.corr(pm25_halaman_15m.shift(-lag), method="spearman")
                      for lag in range(maks_lag + 1)}, name="spearman")
```

> **Catatan:** `asfreq(..., fill_value=0)` mengisi jam kosong dengan 0. Itu benar hanya bila kamera memang menyala selama jam itu; bila kamera sempat mati, potong periode tersebut lebih dulu.

### 7.4 Kejadian (deteksi lonjakan & sumber)

| Parameter | Nilai awal | Dasar |
|---|---|---|
| **Lonjakan PM2.5** (rata-rata 15 menit) | ≥ median 3 jam sebelumnya **+ 10 µg/m³** dan ≥ **1,5 ×** median | 10 µg/m³ ≈ galat sensor PM murah **[Asumsi]**; 1,5× **[Asumsi]** |
| **Polusi dari jalan** | Lonjakan PM halaman; PM dalam ikut naik **0–45 menit** kemudian; kendaraan 1 jam ≥ `ramai` | Dokumen rancangan; angka jeda **[Asumsi]** |
| **Sumber dari dalam rumah** | Lonjakan PM dalam, PM halaman **tidak** melonjak; cocokkan dengan jam masak/tidur/obat nyamuk di onboarding (± 30 menit); di luar jadwal = sumber tak terjadwal | Dokumen rancangan |
| **Uap dari aktivitas** | RH dalam naik ≥ **8 poin dalam 30 menit**, bertepatan jam masak/mandi | **[Asumsi]** |
| **Sumber lembap tetap** | AH dalam − AH luar **≥ 2 g/m³** pada median jam 00–05 selama ≥ 3 malam, tanpa lonjakan aktivitas; atau RH > 60 % ≥ 20 jam/hari selama ≥ 3 hari | Dokumen: "tetap tinggi sepanjang hari tanpa pola aktivitas"; angka **[Asumsi]** |
| **Asap pembakaran sekitar** | Kejadian asap dari kamera aktif; **atau** PM2.5 halaman 1 jam > 55,4 sementara kendaraan 1 jam < `ramai` (lalu lintas tidak menjelaskan lonjakan) | Kamera: `kejadian_asap.csv`; aturan kedua **[Asumsi]** |
| **Ventilasi kurang** (opsional CO₂) | CO₂ dalam ≥ **1000 ppm** | Acuan praktis umum, bukan baku mutu **[Asumsi]** |
| **Jamur terlihat** | ≤ ~1 m²: bersihkan & keringkan sendiri; > ~1 m²: jasa profesional | Panduan US EPA di dokumen |

### 7.5 Kualitas alat

Peringatkan pengguna untuk membersihkan sensor / mengulang kalibrasi bila: nilai **konstan ≥ 6 jam**, **data hilang > 20 %** dalam sehari, atau perbandingan dalam–luar tampak tidak wajar (mis. RH terus-menerus ≥ 95 % ≥ 48 jam). Jadwal pembersihan mengikuti panduan produsen sensor. (Semua angka **[Asumsi]**.)

### 7.6 Peninjauan ambang

Setelah beberapa minggu, rumah dapat dipindah ke profil **ketat** bila penghuni dengan asma/penyakit paru melaporkan gejala **≥ 3 kali dalam 4 minggu** pada hari dengan jam RH > 60 % ≥ 12 jam **atau** rata-rata PM2.5 dalam > 15 µg/m³ (**[Asumsi]**, sesuai dokumen: "memperketat ambang untuk rumah tertentu"). Sistem tidak mendiagnosis; keluhan kesehatan diarahkan ke tenaga kesehatan.

## Langkah 8 — Deteksi kejadian & keputusan ventilasi (`keputusan.py`)

```python
# keputusan.py
import math
import numpy as np
import pandas as pd

AMBANG = {
    "rh_lembap":         {"standar": 60, "ketat": 55},    # % RH dalam rumah
    "rh_aktivitas_naik": 8,                                # poin RH / 30 menit
    "margin_ah":         1.0,                              # g/m³
    "selisih_ah_tetap":  2.0,                              # g/m³ (sumber lembap tetap)
    "pm25_buka":         {"standar": 25, "ketat": 15},     # µg/m³, rata-rata 1 jam halaman
    "pm10_buka":         {"standar": 50, "ketat": 45},
    "pm25_bahaya":       55.4,                             # batas atas ISPU "Sedang"
    "lonjakan_min":      10,                               # µg/m³
    "lonjakan_rasio":    1.5,
    "co2_ventilasi":     1000,                             # ppm
    "hujan_prob":        50,                               # %
    "kuota_saran_harian": 3,
}


def kelembapan_absolut(suhu_c, rh):
    es = 6.112 * math.exp(17.67 * suhu_c / (suhu_c + 243.5))
    return 216.7 * (rh / 100.0 * es) / (273.15 + suhu_c)


# ---------- 8a. Deteksi kejadian (data 15 menit) ----------
def lonjakan(seri_15m: pd.Series) -> pd.Series:
    dasar = seri_15m.rolling(12, min_periods=6).median().shift(1)      # 12 × 15 menit = 3 jam
    return ((seri_15m - dasar) >= AMBANG["lonjakan_min"]) & (seri_15m >= dasar * AMBANG["lonjakan_rasio"])


def klasifikasi_kejadian(d: pd.DataFrame, ambang_ramai: float | None) -> pd.Series:
    """d: indeks waktu (WIB, per 15 menit); kolom pm25_luar, pm25_dalam, rh_dalam, lalin (kendaraan/15 mnt),
    asap (bool), hujan (bool). ambang_ramai: kendaraan/jam (None bila data belum cukup)."""
    d = d.copy()
    lonj_luar, lonj_dalam = lonjakan(d["pm25_luar"]), lonjakan(d["pm25_dalam"])
    luar_baru = lonj_luar.astype(float).rolling(4, min_periods=1).max() > 0     # ≤ 45 menit terakhir
    lalin_1j = d["lalin"].rolling(4, min_periods=1).sum()
    ramai = (lalin_1j >= ambang_ramai) if ambang_ramai else pd.Series(False, index=d.index)
    naik_rh = d["rh_dalam"].diff(2) >= AMBANG["rh_aktivitas_naik"]               # 30 menit

    kondisi = [d["asap"], lonj_dalam & luar_baru & ramai, lonj_dalam & ~luar_baru, naik_rh, d["hujan"]]
    label = ["asap_pembakaran", "polusi_dari_jalan", "sumber_dalam_rumah", "kelembapan_aktivitas", "hujan"]
    return pd.Series(np.select(kondisi, label, default=""), index=d.index, name="kejadian")
    # Untuk "sumber_dalam_rumah": cocokkan jam dengan inventaris["kebiasaan"] (masak/tidur/obat nyamuk).


# ---------- 8b. Keputusan ventilasi ----------
def status_jendela(s: dict, rentan: bool = False):
    """s: rh_dalam, suhu_dalam, rh_luar, suhu_luar, pm25_luar, pm10_luar,
          hujan_sekarang, prob_hujan_1jam, lalin_ramai, asap.   -> (status, alasan)"""
    p = "ketat" if rentan else "standar"
    A = AMBANG
    ah_d = kelembapan_absolut(s["suhu_dalam"], s["rh_dalam"])
    ah_l = kelembapan_absolut(s["suhu_luar"], s["rh_luar"])
    lembap = s["rh_dalam"] > A["rh_lembap"][p]

    if s["asap"]:
        return "tutup", "Asap terdeteksi di halaman. Laporkan sumber asap ke pengurus RT."
    if s["hujan_sekarang"] or s["prob_hujan_1jam"] >= A["hujan_prob"]:
        return "tutup", "Sedang atau akan hujan." + (
            " Pakai AC mode dry, dehumidifier, atau penyerap lembap bila ada." if lembap else "")
    # Perbandingan AH hanya relevan bila rumah memang lembap (perluasan dari dokumen).
    if lembap and ah_l >= ah_d - A["margin_ah"]:
        return "tutup", (f"Udara luar tidak lebih kering (luar {ah_l:.1f} vs dalam {ah_d:.1f} g/m³). "
                         "Pakai alat pengering yang ada, atau penyerap lembap.")
    pm_ok = s["pm25_luar"] <= A["pm25_buka"][p] and s["pm10_luar"] <= A["pm10_buka"][p]
    if not pm_ok or s["lalin_ramai"]:
        return "tunda", "PM halaman tinggi atau jam sibuk kendaraan. Tunggu jam aman terdekat."
    return "buka", ("Udara luar lebih kering dan PM rendah. " if lembap else "PM halaman rendah dan kendaraan sepi. ") \
        + "Ventilasi silang selama kondisi ini bertahan."


def jam_aman_terdekat(ramalan, pola_jam, ambang_ramai, prob_hujan_maks=50):
    """ramalan: [("14:00", prob_hujan), ...]; pola_jam: {jam(int): rata-rata kendaraan/jam}."""
    for hhmm, prob in ramalan:
        jam = int(hhmm[:2])
        if prob < prob_hujan_maks and (ambang_ramai is None or pola_jam.get(jam, 0) < ambang_ramai):
            return hhmm
    return None
```

Agar status tidak berkedip, ubah status hanya bila kondisi baru bertahan **≥ 2 pembacaan 15 menit** berturut-turut (kecuali `asap` dan `hujan`, yang langsung berlaku).

### Saran tambahan untuk dapur (PM dalam naik saat memasak, PM halaman rendah)

Status jendela dapur → **buka**, saran: nyalakan exhaust fan atau kipas angin yang diarahkan ke luar (sesuai inventaris, Langkah 9).

### Penyiraman halaman (opsional)

Saran penyiraman dari rancangan awal (bukan bagian dokumen AIR HouseGuard): aktifkan hanya bila PM10 halaman 1 jam > ambang buka, tidak hujan/berpeluang hujan, dan tidak ada asap. Efektivitasnya menekan debu jalan **belum diverifikasi di sini**, dan air menambah kelembapan di sekitar rumah; alternatif yang ada di dokumen adalah **pagar semak rapat di sisi jalan**. Bila tidak ingin mempertahankannya, hapus saja.

## Langkah 9 — Saran sesuai peralatan, notifikasi, konfirmasi, evaluasi 7 hari

### 9a. Tingkatan saran (dari dokumen)

Untuk setiap kebutuhan, sistem memeriksa inventaris: alat ada → cara memakainya; bila tidak ada → alternatif **tanpa biaya → biaya rendah → investasi**. Sistem **tidak** menyarankan tanaman pot dalam rumah untuk menurunkan polusi/kelembapan (efeknya kecil, dan tanah menambah uap serta bisa menumbuhkan jamur); pagar semak di halaman sisi jalan boleh.

```python
URUTAN = ["alat_ada", "tanpa_biaya", "biaya_rendah", "investasi"]

KATALOG = {
    "asap_dapur": {
        "alat_ada": "Nyalakan exhaust fan atau cooker hood selama memasak dan beberapa menit sesudahnya.",
        "tanpa_biaya": "Buka jendela dapur saat memasak bila udara luar baik; tutup panci saat merebus.",
        "biaya_rendah": "Arahkan kipas angin ke luar jendela dapur.",
        "investasi": "Pasang exhaust fan dinding atau cooker hood."},
    "lembap_umum": {      # RH > 60 %
        "alat_ada": "Nyalakan AC mode dry atau dehumidifier pada jam yang disarankan.",
        "tanpa_biaya": "Ventilasi silang saat udara luar lebih kering dan PM rendah; jemur pakaian di luar; perbaiki kebocoran.",
        "biaya_rendah": "Pakai penyerap lembap untuk lemari dan ruang kecil.",
        "investasi": "Beli dehumidifier."},
    "polusi_jalan": {
        "alat_ada": "Nyalakan penjernih udara HEPA di kamar tidur pada jam sibuk kendaraan.",
        "tanpa_biaya": "Tutup jendela sisi jalan pada jam sibuk; tutup celah bawah pintu.",
        "biaya_rendah": "Pasang filter udara di bagian belakang kipas angin.",
        "investasi": "Beli penjernih udara HEPA."},
    "sumber_dalam": {     # tidak bergantung alat
        "tanpa_biaya": "Tidak merokok di dalam rumah; ganti obat nyamuk bakar dengan kelambu.",
        "biaya_rendah": "Gunakan obat nyamuk elektrik."},
}


def alat_tersedia(kebutuhan, inv):
    return {"asap_dapur": inv["exhaust_dapur"] == "berfungsi",
            "lembap_umum": inv["ac"] == "dry" or inv["dehumidifier"],
            "polusi_jalan": inv["hepa"],
            "sumber_dalam": False}[kebutuhan]


def pilih_saran(jam_bermasalah, inv, tidak_dijalankan, n=3):
    """jam_bermasalah: {"asap_dapur": 9, "lembap_umum": 14, ...} (jam bermasalah minggu ini).
    tidak_dijalankan: {teks_saran: berapa kali berturut-turut belum dijalankan}.
    Sumber dengan jam terbanyak diprioritaskan; saran yang 2× tidak dijalankan diganti, bukan diulang."""
    hasil = []
    for k, _ in sorted(jam_bermasalah.items(), key=lambda kv: -kv[1]):
        awal = 0 if alat_tersedia(k, inv) else 1
        for tingkat in URUTAN[awal:]:
            teks = KATALOG[k].get(tingkat)
            if teks and tidak_dijalankan.get(teks, 0) < 2:
                hasil.append((k, tingkat, teks))
                break
        if len(hasil) == n:
            break
    return hasil
```

Saran yang **terbukti berhasil** (jam bermasalah sumber itu turun) dipertahankan sebagai kebiasaan, dan laporan berikutnya beralih ke sumber masalah berikutnya.

### 9b. Notifikasi & konfirmasi

```python
def boleh_kirim(level, terkirim_hari_ini):
    """Maks. 3 saran/hari; kondisi berbahaya (api, asap, PM2.5 > 55,4) selalu boleh."""
    return level == "bahaya" or terkirim_hari_ini < AMBANG["kuota_saran_harian"]
```

Setiap saran yang dikirim dicatat di `log_tindakan` (`waktu, saran, dijalankan: null/true/false`). Tombol **sudah / belum** di aplikasi mengisi kolom `dijalankan`; data ini dipakai untuk "dua kali tidak dijalankan" dan untuk persentase saran yang dijalankan.

### 9c. Evaluasi 7 hari

```python
def ringkasan_minggu(jam: pd.DataFrame, kejadian_15m: pd.Series) -> dict:
    """jam: indeks waktu WIB per jam; kolom pm25_dalam, rh_dalam.
    kejadian_15m: label kejadian per 15 menit (indeks WIB), dari klasifikasi_kejadian()."""
    lembap = (jam["rh_dalam"] > 60).groupby(jam.index.date).sum().mean()
    k = kejadian_15m[kejadian_15m.isin(["polusi_dari_jalan", "sumber_dalam_rumah"])]
    return {
        "jam_lembap_per_hari": round(float(lembap), 1),
        "pm25_rata_dalam": round(float(jam["pm25_dalam"].mean()), 1),
        "jam_pm25_dalam_di_atas_15": int((jam["pm25_dalam"] > 15).sum()),
        "porsi_lonjakan_pm_persen": (k.value_counts(normalize=True) * 100).round().to_dict(),
    }
# Bandingkan minggu ini dengan minggu lalu (panggil dua kali, kurangkan).
```

**Isi laporan** (sesuai dokumen): ringkasan angka (jam lembap/hari, PM2.5 rata-rata dalam, jam > 15 µg/m³, jam puncak); asal polusi (porsi luar/dalam beserta jam); 2–3 saran prioritas (`pilih_saran`); hasil saran minggu lalu (sebelum–sesudah); status alat (Langkah 7.5).

**Indikator keberhasilan:** turunnya jam/hari RH > 60 %, turunnya PM2.5 rata-rata dalam dan jam > 15 µg/m³, turunnya lonjakan PM dari dalam rumah, persentase saran dijalankan, dan perubahan pengetahuan penghuni (survei singkat sebelum–sesudah). Untuk uji coba satu RT, gabungkan angka **tanpa identitas rumah** (mis. porsi polusi pagi dari jalan).

## Langkah 10 — Halaman web

### 10a. Live kamera — pilih salah satu

**Opsi 1 (paling sederhana):** stream asli dengan `hls.js` (tanpa kotak deteksi). Server kamera bisa membatasi pemutaran dari domain lain; jika diblokir, pakai Opsi 2.

**Opsi 2:** hasil beranotasi lewat endpoint MJPEG tambahan di `deteksi_kendaraan.py`:

```python
# --- bagian atas deteksi_kendaraan.py ---
import threading
from flask import Flask, Response

frame_terbaru = None
app_video = Flask("video")

@app_video.route("/video")
def video():
    def alur():
        while True:
            if frame_terbaru is not None:
                ok, buf = cv2.imencode(".jpg", frame_terbaru, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.1)
    return Response(alur(), mimetype="multipart/x-mixed-replace; boundary=f")

# --- di main(), sebelum loop ---
threading.Thread(target=lambda: app_video.run(port=5001, threaded=True), daemon=True).start()

# --- di dalam loop, setelah semua gambar selesai ---
frame_terbaru = frame
```

Tampilan live bersifat **sementara** (tidak disimpan) dan hanya boleh dibuka keluarga pemilik rumah (beri autentikasi sebelum dipublikasikan; lihat [Privasi](#12-privasi)).

### 10b. Backend (`app.py`)

```python
import json
import os
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from cuaca import cuaca_sekarang
from keputusan import status_jendela, jam_aman_terdekat

app = Flask(__name__, static_folder="static")
CSV_KENDARAAN, CSV_ASAP = "hasil_deteksi.csv", "kejadian_asap.csv"
FILE = {"halaman": "sensor_halaman.csv", "dalam": "sensor_dalam.csv"}
KOLOM = {"halaman": ["pm25", "pm10", "suhu", "rh"], "dalam": ["pm25", "pm10", "suhu", "rh", "co2"]}
KOREKSI = {"dalam": {"pm25": (1.0, 0.0), "pm10": (1.0, 0.0), "suhu": (1.0, 0.0), "rh": (1.0, 0.0)}}  # (a, b) dari kalibrasi
JENIS = ["mobil", "motor", "bus", "truk", "sepeda"]


def muat_inventaris():
    with open("inventaris.json", encoding="utf-8") as f:
        return json.load(f)


def muat_ambang_lalin():
    try:
        with open("ambang_lalin.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None          # data belum cukup -> kendaraan hanya informasi


@app.post("/api/ingest/<lokasi>")
def ingest(lokasi):
    if lokasi not in FILE:
        abort(404)
    if request.headers.get("X-Token") != os.environ["INGEST_TOKEN"]:
        abort(401)
    b = request.get_json(force=True)
    nilai = []
    for k in KOLOM[lokasi]:
        v = b.get(k)
        a, c = KOREKSI.get(lokasi, {}).get(k, (1.0, 0.0))
        nilai.append("" if v is None else a * float(v) + c)
    baru = not os.path.exists(FILE[lokasi])
    with open(FILE[lokasi], "a", encoding="utf-8") as f:
        if baru:
            f.write("waktu," + ",".join(KOLOM[lokasi]) + "\n")
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + "," + ",".join(map(str, nilai)) + "\n")
    return jsonify(ok=True)


def rata_terakhir(lokasi, menit=60):
    if not os.path.exists(FILE[lokasi]):
        return None
    d = pd.read_csv(FILE[lokasi], parse_dates=["waktu"])
    if d.empty:
        return None
    d["waktu"] = d["waktu"].dt.tz_localize("UTC")
    d = d[d["waktu"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=menit)]
    return None if d.empty else d.drop(columns="waktu").mean()


def kendaraan_df():
    if not os.path.exists(CSV_KENDARAAN):
        return None
    df = pd.read_csv(CSV_KENDARAAN, parse_dates=["timestamp"])
    if df.empty:
        return None
    df["waktu"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Jakarta")
    return df


def asap_aktif():
    if not os.path.exists(CSV_ASAP):
        return False
    d = pd.read_csv(CSV_ASAP)
    return (not d.empty) and d.iloc[-1]["status"] == "mulai"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/jam")
def api_jam():
    df = kendaraan_df()
    if df is None:
        return jsonify([])
    df = df[df["waktu"] >= pd.Timestamp.now(tz="Asia/Jakarta") - pd.Timedelta(hours=24)]
    t = (df.groupby([df["waktu"].dt.floor("h"), "object"]).size().unstack(fill_value=0)
           .reindex(columns=JENIS, fill_value=0))
    t["TOTAL"] = t.sum(axis=1)
    t = t.reset_index()
    t["waktu"] = t["waktu"].dt.strftime("%Y-%m-%d %H:00")
    return jsonify(t.to_dict("records"))


@app.route("/api/status")
def api_status():
    luar, dalam = rata_terakhir("halaman"), rata_terakhir("dalam")
    if luar is None or dalam is None:
        return jsonify(status="tidak ada data", alasan="Sensor belum mengirim data 60 menit terakhir.")
    inv, ambang = muat_inventaris(), muat_ambang_lalin()
    cw = cuaca_sekarang()

    ramai = False
    df = kendaraan_df()
    if df is not None and ambang:
        ramai = int((df["waktu"] >= pd.Timestamp.now(tz="Asia/Jakarta") - pd.Timedelta(minutes=60)).sum()) >= ambang["ramai"]

    s = {"rh_dalam": dalam["rh"], "suhu_dalam": dalam["suhu"], "rh_luar": luar["rh"], "suhu_luar": luar["suhu"],
         "pm25_luar": luar["pm25"], "pm10_luar": luar["pm10"], "hujan_sekarang": cw["hujan_sekarang"],
         "prob_hujan_1jam": cw["prob_hujan_1jam"], "lalin_ramai": ramai, "asap": asap_aktif()}
    status, alasan = status_jendela(s, rentan=bool(inv.get("penghuni_rentan")))
    jam_aman = None
    if status == "tunda":
        pola = {int(k): v for k, v in (ambang or {}).get("pola_jam", {}).items()}
        jam_aman = jam_aman_terdekat(cw["ramalan"], pola, ambang["ramai"] if ambang else None)
    return jsonify(status=status, alasan=alasan, jam_aman=jam_aman,
                   pm25_halaman=round(float(luar["pm25"]), 1), pm25_dalam=round(float(dalam["pm25"]), 1),
                   rh_dalam=round(float(dalam["rh"]), 1))


if __name__ == "__main__":
    app.run(port=5000)
```

Jalankan dengan `INGEST_TOKEN`, `RUMAH_LAT`, `RUMAH_LON` sebagai variabel lingkungan. File CSV cukup untuk prototipe; untuk produksi pakai basis data (mis. SQLite/PostgreSQL).

### 10c. Halaman (`static/index.html`)

```html
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIR HouseGuard</title>
  <script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 1rem auto; padding: 0 1rem; }
    video, img { width: 100%; background: #000; border-radius: 8px; }
    .kartu { padding: 1rem; border-radius: 8px; margin: 1rem 0; background: #eef; }
    .buka { background: #dfd; } .tunda { background: #ffd; } .tutup { background: #fdd; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: .4rem; text-align: center; }
  </style>
</head>
<body>
  <h1>AIR HouseGuard</h1>

  <video id="live" controls autoplay muted playsinline></video>
  <!-- Opsi 2: ganti dengan <img src="http://localhost:5001/video"> -->

  <div id="status" class="kartu">Memuat status jendela…</div>

  <h2>Kendaraan per jam (24 jam terakhir, WIB)</h2>
  <table>
    <thead><tr><th>Jam</th><th>Mobil</th><th>Motor</th><th>Bus</th><th>Truk</th><th>Sepeda</th><th>Total</th></tr></thead>
    <tbody id="tabel"></tbody>
  </table>

<script>
  const URL_STREAM = "https://alamat-stream-anda/index.m3u8";
  const v = document.getElementById("live");
  if (v.canPlayType("application/vnd.apple.mpegurl")) v.src = URL_STREAM;
  else if (Hls.isSupported()) { const h = new Hls(); h.loadSource(URL_STREAM); h.attachMedia(v); }

  async function perbarui() {
    const jam = await (await fetch("/api/jam")).json();
    document.getElementById("tabel").innerHTML = jam.reverse().map(r =>
      `<tr><td>${r.waktu.slice(11)}</td><td>${r.mobil}</td><td>${r.motor}</td>` +
      `<td>${r.bus}</td><td>${r.truk}</td><td>${r.sepeda}</td><td><b>${r.TOTAL}</b></td></tr>`).join("");

    const s = await (await fetch("/api/status")).json();
    const el = document.getElementById("status");
    el.className = "kartu " + (["buka", "tunda", "tutup"].includes(s.status) ? s.status : "");
    el.innerHTML = `<b>Jendela: ${s.status.toUpperCase()}</b><br>${s.alasan}` +
      (s.jam_aman ? `<br>Jam aman terdekat: ${s.jam_aman}` : "") +
      (s.pm25_halaman !== undefined ? `<br><small>PM2.5 halaman ${s.pm25_halaman} · dalam ${s.pm25_dalam} µg/m³ · RH dalam ${s.rh_dalam}%</small>` : "");
  }
  perbarui(); setInterval(perbarui, 30000);
</script>
</body>
</html>
```

---

## 11. Menjalankan Semuanya

1. **Sekali:** latih/unduh model asap/api (Langkah 1); letakkan di `models/`.
2. **Sekali per rumah:** kalibrasi dua alat berdampingan 24–48 jam (Langkah 4b), isi `KOREKSI` di `app.py`, pasang alat, isi `inventaris.json` (Langkah 6).
3. **Terminal 1 (terus-menerus):**
   ```bash
   python deteksi_kendaraan.py --url "https://alamat-stream-anda/index.m3u8" --camera-id cam1 --no-show
   ```
4. **Terminal 2:**
   ```bash
   set INGEST_TOKEN=rahasia-anda && set RUMAH_LAT=... && set RUMAH_LON=... && python app.py   # Windows; gunakan export di Linux/macOS
   ```
5. Alat sensor mengirim data tiap menit ke `/api/ingest/halaman` dan `/api/ingest/dalam` (header `X-Token`).
6. Buka `http://localhost:5000`.
7. **Setelah ≥ 7 hari data kamera:** jalankan `ambang_kendaraan()` dan `korelasi_lalin_pm()` (Langkah 7.3) → `ambang_lalin.json`; ulangi tiap bulan.
8. **Tiap 7 hari:** jalankan `ringkasan_minggu()` + `pilih_saran()` dan kirim laporan.

Untuk akses dari perangkat lain di jaringan yang sama jalankan Flask dengan `host="0.0.0.0"`; untuk akses publik pakai server produksi (mis. `waitress`/`gunicorn`) di balik reverse proxy dengan HTTPS dan autentikasi.

---

## 12. Privasi

Dokumen rancangan menetapkan: **video diproses di dalam alat; yang disimpan hanya angka** (jumlah kendaraan, ada/tidaknya asap), bukan wajah atau pelat nomor; data tiap rumah hanya bisa dilihat keluarga itu; laporan tingkat RT hanya angka gabungan tanpa identitas rumah.

**Ada dua hal pada kode saat ini yang bertentangan dengan prinsip itu dan perlu disesuaikan sebelum dipakai di rumah:**

1. `deteksi_kendaraan.py` **menyimpan foto bukti** kejadian asap/api di `foto_asap/` (frame penuh, bisa memuat wajah/pelat). Tambahkan opsi untuk mematikannya:
   ```python
   # tambahkan argumen
   p.add_argument("--no-snapshot", action="store_true", help="jangan simpan foto bukti (privasi)")

   # di PemantauAsap._mulai_kejadian, ganti blok penyimpanan foto:
   if self.args.no_snapshot:
       self.foto = ""
   else:
       bukti = frame_asli.copy()
       self.gambar(bukti, banner=False)
       self.foto = os.path.join(self.args.smoke_snapshot_dir,
                                f"{self.args.camera_id}_{self.mulai:%Y%m%d_%H%M%S}.jpg")
       cv2.imwrite(self.foto, bukti)
   ```
   Pada pemasangan nyata, jalankan dengan `--no-snapshot`.
2. `--save-video` menyimpan video beranotasi. **Jangan dipakai** di pemasangan nyata.

Tampilan live (Langkah 10a) hanya menayangkan frame secara sementara; pastikan akses dibatasi ke pemilik rumah. Pada contoh di repositori ini kamera yang dipakai adalah stream CCTV publik untuk **pengujian**; pastikan Anda berhak memakainya.

---

## 13. Batasan & Pemecahan Masalah

**Batasan**

- Sensor PM murah tidak bisa membedakan partikel jamur, asap, dan debu. Kelembapan dipakai sebagai **penanda risiko** jamur, bukan pengukur jamur.
- Sistem **tidak memberi diagnosis atau saran pengobatan**; keluhan kesehatan diarahkan ke tenaga kesehatan.
- Hitungan kendaraan adalah estimasi dan bergantung posisi kamera; korelasi kendaraan–PM harus diuji per rumah (Langkah 7.3).
- Ambang bertanda **[Asumsi]** belum divalidasi; nilai acuan WHO 2021 berlaku untuk rata-rata 24 jam sedangkan di sini dipakai sebagai pemicu per jam.
- PM wilayah dari Open-Meteo adalah model regional, bukan data tepi jalan.
- Model asap/api dilatih pada dataset umum; uji pada kamera Anda dan periksa deteksi palsu (kabut, uap, debu, lampu).

**Pemecahan masalah**

| Masalah | Penyebab / solusi |
|---|---|
| `Gagal membuka stream` | Buka URL di VLC; pastikan stream aktif dan Anda punya izin. |
| FPS rendah (< 8) | Pakai GPU, turunkan `--imgsz`, naikkan `--skip`, atau `--model yolo11n.pt`. |
| Motor kecil tidak terhitung | Naikkan `--imgsz` (1280), turunkan `--conf`. |
| Gerobak/mobil parkir terhitung | Naikkan `--min-move`; truk/bus palsu → naikkan `--heavy-conf`. |
| Kendaraan terhitung ganda | Naikkan `--merge-gap`/`--merge-dist` (mode `masuk`) atau pakai `--mode garis`. |
| Malam terlalu gelap | Naikkan `--night-thresh` dan/atau `--night-target`. |
| Asap knalpot dihitung sebagai kejadian | Jangan pakai `--smoke-keep-vehicle`; naikkan `--smoke-conf`. |
| Model asap tidak ditemukan | Cek `--smoke-model`; script melewatkan deteksi asap tanpa menghentikan penghitungan. |
| Path `C:\Users\NADIA\...` tidak ada | Path bawaan masih milik komputer pengembang — beri argumen `--url`, `--models-dir`, `--smoke-model`. |
| Jam di tabel bergeser 7 jam | CSV bersifat UTC; konversi ke `Asia/Jakarta`. |
| Status jendela hampir selalu "tunda" | PM halaman memang sering di atas ambang buka; lihat catatan di 7.2 sebelum melonggarkannya. |
| `ambang_kendaraan()` mengembalikan `None` | Data kamera belum 7 hari; sementara keputusan hanya memakai PM halaman. |

**Sumber acuan ambang:** WHO Global Air Quality Guidelines 2021 (PM2.5/PM10, 24 jam & interim target); Permen LHK P.14/MENLHK/SETJEN/KUM.1/7/2020 dan Peraturan BMKG No. 8/2022 (kategori ISPU); PP 22/2021 (baku mutu nasional PM2.5 55 dan PM10 75 µg/m³, 24 jam); anjuran kelembapan dan panduan jamur US EPA seperti tercantum di dokumen rancangan AIR HouseGuard.
