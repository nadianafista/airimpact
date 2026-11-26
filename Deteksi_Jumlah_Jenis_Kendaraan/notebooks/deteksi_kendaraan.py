import argparse
import csv
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_URL = r"C:\Users\NADIA\OneDrive\ドキュメント\airimpact\Deteksi_Kendaraan\Deteksi_Jumlah_Jenis_Kendaraan\data\WhatsApp Video 2026-11-23 at 13.57.42.mp4"

# Folder tempat model YOLO disimpan. Model hanya di-download SEKALI ke sini,
# run berikutnya langsung dipakai dari folder ini tanpa download lagi.
DEFAULT_MODELS_DIR = r"C:\Users\NADIA\OneDrive\ドキュメント\airimpact\Deteksi_Kendaraan\Deteksi_Jumlah_Jenis_Kendaraan\models"

# Model deteksi asap & api hasil training HouseGuard
DEFAULT_SMOKE_MODEL = os.path.join(DEFAULT_MODELS_DIR, "houseguard_model_best.pt")
NAMA_ASAP = {"smoke": "asap", "fire": "api"}
WARNA_ASAP = {"asap": (170, 170, 170), "api": (0, 0, 255)}

# ID kelas COCO -> nama jenis kendaraan (sesuai isi kolom 'object' di data lo)
KELAS = {1: "sepeda", 2: "mobil", 3: "motor", 5: "bus", 7: "truk"}

# Kelas yang jarang muncul dan sering salah tebak (gerobak, lapak, kios dikira truk/bus).
# Deteksi kelas ini baru dipercaya kalau confidence-nya tinggi.
KELAS_BERAT = {"truk", "bus"}

# Warna tetap per jenis (format BGR)
WARNA = {
    "mobil": (255, 140, 0),    # biru
    "motor": (0, 165, 255),    # oranye
    "bus": (200, 0, 200),      # ungu
    "truk": (0, 0, 230),       # merah
    "sepeda": (200, 200, 0),   # toska
}


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def memotong(a, b, c, d):
    """True kalau segmen a-b memotong segmen c-d."""
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


# ======================= PENINGKATAN GAMBAR MALAM =======================

def ukur_kecerahan(frame):
    """Kecerahan frame (0-255). Pakai median supaya tidak tertipu lampu kendaraan yang sangat terang."""
    kecil = cv2.resize(frame, (160, int(160 * frame.shape[0] / frame.shape[1])))
    abu = cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY)
    return float(np.median(abu))


@lru_cache(maxsize=64)
def lut_gamma(gamma):
    """Tabel konversi gamma (dicache supaya tidak dihitung ulang tiap frame)."""
    tabel = ((np.arange(256) / 255.0) ** gamma) * 255
    return np.clip(tabel, 0, 255).astype("uint8")


def tingkatkan_malam(frame, kecerahan, clahe, target=0.45):
    """
    1) Gamma correction: mencerahkan area gelap tanpa membuat area terang (lampu) makin silau.
       Gamma dihitung otomatis supaya kecerahan median naik ke sekitar `target`.
    2) CLAHE di kanal L (LAB): menaikkan kontras lokal supaya bentuk kendaraan lebih jelas.
    """
    m = max(kecerahan / 255.0, 0.02)
    gamma = math.log(target) / math.log(m)
    gamma = round(min(max(gamma, 0.35), 1.0), 2)   # batasi supaya tidak berlebihan

    hasil = cv2.LUT(frame, lut_gamma(gamma))
    lab = cv2.cvtColor(hasil, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), gamma

# ========================================================================


def pilih_garis(frame):
    """Klik 2 titik di frame untuk menentukan garis hitung. Enter = simpan, r = ulang."""
    titik = []
    jendela = "Klik 2 titik garis hitung (Enter = OK, r = ulang)"

    def klik(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(titik) < 2:
            titik.append((x, y))

    cv2.namedWindow(jendela, cv2.WINDOW_NORMAL)          # jendela bisa di-resize
    h, w = frame.shape[:2]
    cv2.resizeWindow(jendela, 1280, int(1280 * h / w))   # seluruh frame muat di layar
    cv2.setMouseCallback(jendela, klik)
    while True:
        tampil = frame.copy()
        for t in titik:
            cv2.circle(tampil, t, 6, (0, 255, 255), -1)
        if len(titik) == 2:
            cv2.line(tampil, titik[0], titik[1], (0, 255, 255), 3)
        cv2.imshow(jendela, tampil)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            titik.clear()
        elif k in (13, 10) and len(titik) == 2:
            break
        elif k == 27:
            raise SystemExit("Dibatalkan.")
    cv2.destroyWindow(jendela)
    return titik


def buat_tracker_yaml(ambang, buffer):
    """Config ByteTrack yang lebih 'sabar' daripada default, supaya motor yang
    kecil / tertutup motor lain tetap dapat track dan tidak gampang hilang."""
    isi = (
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {ambang}\n"   # deteksi >= ini dipakai untuk pencocokan utama
        "track_low_thresh: 0.1\n"          # deteksi lemah tetap dipakai untuk menyambung track lama
        f"new_track_thresh: {ambang}\n"    # ambang untuk membuat track (kendaraan) baru
        f"track_buffer: {buffer}\n"        # berapa frame track ditahan saat kendaraan tertutup
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )
    path = os.path.join(tempfile.gettempdir(), "bytetrack_kendaraan.yaml")
    with open(path, "w") as f:
        f.write(isi)
    return path


def siapkan_model(nama, folder):
    """Muat model dari folder lokal; download hanya kalau belum pernah ada di sana."""
    # Kalau --model berupa path ke file .pt yang sudah ada, langsung pakai
    if os.path.isfile(nama):
        print(f"Pakai model: {os.path.abspath(nama)}")
        return YOLO(nama)

    os.makedirs(folder, exist_ok=True)
    nama_file = os.path.basename(nama)
    path_lokal = os.path.join(folder, nama_file)

    if os.path.isfile(path_lokal):
        print(f"Pakai model lokal (tanpa download): {path_lokal}")
        return YOLO(path_lokal)

    print(f"Model {nama_file} belum ada, download sekali ke: {folder}")
    cwd = os.getcwd()
    try:
        os.chdir(folder)              # ultralytics men-download ke folder kerja saat ini
        model = YOLO(nama_file)
    finally:
        os.chdir(cwd)

    # Jaga-jaga kalau ultralytics menyimpannya di tempat lain: salin ke folder models
    ckpt = getattr(model, "ckpt_path", None)
    if not os.path.isfile(path_lokal) and ckpt and os.path.isfile(ckpt):
        shutil.copy(ckpt, path_lokal)
    if os.path.isfile(path_lokal):
        print(f"Model tersimpan di: {path_lokal}")
    return model


# ======================= DETEKSI ASAP & API =======================

def porsi_di_dalam(a, b):
    """Porsi luas kotak a yang berada di dalam kotak b (0-1)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    irisan = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return irisan / max(1, (a[2] - a[0]) * (a[3] - a[1]))


class PemantauAsap:
    """
    Menjalankan model asap/api tiap `smoke_every` detik (asap bergerak lambat, tidak perlu tiap frame).
    Kejadian baru dianggap NYATA kalau terdeteksi minimal `smoke_min_hits` kali dari `smoke_window`
    pengecekan terakhir, dan dianggap SELESAI kalau seluruh `smoke_window` pengecekan terakhir bersih.
    Setiap kejadian dicatat ke CSV (mulai & selesai) dan disimpan foto buktinya.
    """

    def __init__(self, model, args):
        self.model, self.args = model, args
        self.riwayat = deque(maxlen=args.smoke_window)
        self.kotak = []                  # deteksi terakhir: [(jenis, conf, (x1, y1, x2, y2))]
        self.aktif = False
        self.t_cek = 0.0
        self.mulai = None
        self.jenis = set()
        self.conf_maks = 0.0
        self.foto = ""
        self.jumlah_kejadian = 0
        os.makedirs(args.smoke_snapshot_dir, exist_ok=True)
        baru = not os.path.exists(args.smoke_output)
        self.f = open(args.smoke_output, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.f)
        if baru:
            self.writer.writerow(["camera_id", "status", "jenis", "conf_maks", "waktu_mulai",
                                  "waktu_selesai", "durasi_detik", "foto"])

    def perlu_cek(self):
        return time.time() - self.t_cek >= self.args.smoke_every

    def cek(self, frame_asli, kotak_kendaraan):
        self.t_cek = time.time()
        r = self.model.predict(frame_asli, conf=self.args.smoke_conf, imgsz=self.args.smoke_imgsz,
                               device=self.args.device, verbose=False)[0]
        kotak = []
        for c, s, xyxy in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.int().tolist()):
            nama = self.model.names[c]
            jenis = NAMA_ASAP.get(nama, nama)
            # Asap yang menempel di kotak kendaraan kemungkinan besar asap knalpot -> abaikan
            if (jenis == "asap" and not self.args.smoke_keep_vehicle
                    and any(porsi_di_dalam(xyxy, k) >= 0.6 for k in kotak_kendaraan)):
                continue
            kotak.append((jenis, s, tuple(xyxy)))
        self.kotak = kotak
        self.riwayat.append(len(kotak) > 0)
        hit = sum(self.riwayat)

        if not self.aktif and hit >= self.args.smoke_min_hits:
            self._mulai_kejadian(frame_asli)
        elif self.aktif:
            if kotak:
                self.jenis |= {k[0] for k in kotak}
                self.conf_maks = max(self.conf_maks, max(k[1] for k in kotak))
            if hit == 0 and len(self.riwayat) == self.riwayat.maxlen:
                self.selesai()

    def _mulai_kejadian(self, frame_asli):
        self.aktif = True
        self.jumlah_kejadian += 1
        self.mulai = datetime.now(timezone.utc)
        self.jenis = {k[0] for k in self.kotak}
        self.conf_maks = max((k[1] for k in self.kotak), default=0.0)
        bukti = frame_asli.copy()
        self.gambar(bukti, banner=False)
        self.foto = os.path.join(self.args.smoke_snapshot_dir,
                                 f"{self.args.camera_id}_{self.mulai:%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(self.foto, bukti)
        self._tulis("mulai")
        print(f"[{datetime.now():%H:%M:%S}] ⚠️ KEJADIAN {'+'.join(sorted(self.jenis)).upper()} "
              f"(conf {self.conf_maks:.2f}) -> foto: {self.foto}")

    def selesai(self, status="selesai"):
        if not self.aktif:
            return
        self._tulis(status, datetime.now(timezone.utc))
        print(f"[{datetime.now():%H:%M:%S}] Kejadian asap/api {status}.")
        self.aktif = False

    def _tulis(self, status, akhir=None):
        self.writer.writerow([
            self.args.camera_id, status, "+".join(sorted(self.jenis)), f"{self.conf_maks:.2f}",
            self.mulai.strftime("%Y-%m-%d %H:%M:%S"),
            akhir.strftime("%Y-%m-%d %H:%M:%S") if akhir else "",
            f"{(akhir - self.mulai).total_seconds():.0f}" if akhir else "",
            self.foto,
        ])
        self.f.flush()

    def gambar(self, frame, banner=True):
        for jenis, s, (x1, y1, x2, y2) in self.kotak:
            w = WARNA_ASAP.get(jenis, (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), w, 2)
            cv2.putText(frame, f"{jenis.upper()} {s:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, w, 2)
        if banner and self.aktif:
            teks = f"PERINGATAN: {'+'.join(sorted(self.jenis)).upper()} TERDETEKSI"
            (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            x = (frame.shape[1] - tw) // 2
            cv2.rectangle(frame, (x - 12, 8), (x + tw + 12, 20 + th), (0, 0, 200), -1)
            cv2.putText(frame, teks, (x, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def tutup(self):
        self.selesai("dihentikan")
        self.f.close()

# ===================================================================


def buka_stream(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL, help="URL stream .m3u8 atau path file video")
    p.add_argument("--camera-id", default="cam1")
    p.add_argument("--output", default="hasil_deteksi.csv")
    p.add_argument("--model", default="yolo11s.pt",
                   help="yolo11n.pt (paling cepat) / yolo11s.pt (default) / yolo11m.pt (paling akurat, berat)")
    p.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="folder penyimpanan model YOLO")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="resolusi input YOLO. 1280 jauh lebih bagus untuk motor kecil daripada 640")
    p.add_argument("--minutes", type=float, default=0, help="lama rekam dalam menit, 0 = sampai tekan q")
    p.add_argument("--conf", type=float, default=0.25,
                   help="ambang confidence untuk kendaraan baru (turunkan kalau masih banyak yang lolos)")
    p.add_argument("--track-buffer", type=int, default=60,
                   help="berapa frame track ditahan saat kendaraan tertutup kendaraan lain")
    p.add_argument("--mode", choices=["masuk", "garis"], default="masuk",
                   help="masuk = hitung begitu kendaraan muncul (tanpa garis); garis = hitung saat lewat garis")
    p.add_argument("--merge-gap", type=int, default=50,
                   help="[mode masuk] ID baru yang muncul <= sekian frame setelah ID lama hilang di dekatnya "
                        "dianggap kendaraan yang sama")
    p.add_argument("--merge-dist", type=float, default=1.0,
                   help="[mode masuk] jarak maksimum (x ukuran kotak kendaraan) untuk menyambung ID lama dan baru")
    p.add_argument("--line", default="0.0,0.9,1.0,0.9",
                   help="garis hitung x1,y1,x2,y2 dalam pecahan lebar/tinggi frame (0-1)")
    p.add_argument("--set-line", action="store_true", help="tentukan garis hitung dengan klik di frame pertama")
    p.add_argument("--min-frames", type=int, default=None,
                   help="kendaraan harus ter-track minimal sekian frame sebelum dihitung (anti deteksi palsu). "
                        "Default: 5 untuk mode masuk, 3 untuk mode garis")
    p.add_argument("--min-move", type=float, default=0.5,
                   help="kendaraan baru dihitung setelah bergerak minimal sekian x ukuran kotaknya. "
                        "Objek diam (gerobak, lapak, mobil parkir) tidak ikut terhitung")
    p.add_argument("--heavy-conf", type=float, default=0.5,
                   help="confidence minimum agar deteksi truk/bus dipercaya (cegah gerobak dikira truk)")
    p.add_argument("--device", default="auto",
                   help="auto = pakai GPU (CUDA) kalau ada; 'cpu' = paksa CPU; '0' = GPU pertama")
    p.add_argument("--skip", type=int, default=1, help="proses 1 dari tiap N frame (naikkan kalau laptop lemot)")
    p.add_argument("--no-show", action="store_true", help="jangan tampilkan jendela video")
    p.add_argument("--save-video", default="", help="simpan video beranotasi, mis. hasil.mp4")
    # --- deteksi asap & api ---
    p.add_argument("--smoke-model", default=DEFAULT_SMOKE_MODEL, help="path model asap/api (.pt)")
    p.add_argument("--no-smoke", action="store_true", help="matikan deteksi asap/api")
    p.add_argument("--smoke-conf", type=float, default=0.4, help="confidence minimum deteksi asap/api")
    p.add_argument("--smoke-every", type=float, default=1.0, help="cek asap tiap sekian detik")
    p.add_argument("--smoke-window", type=int, default=5, help="jumlah pengecekan terakhir yang diperhitungkan")
    p.add_argument("--smoke-min-hits", type=int, default=3,
                   help="kejadian dianggap nyata kalau terdeteksi minimal sekian kali dari --smoke-window")
    p.add_argument("--smoke-imgsz", type=int, default=640, help="resolusi input model asap (sama dengan training)")
    p.add_argument("--smoke-output", default="kejadian_asap.csv", help="CSV catatan kejadian asap/api")
    p.add_argument("--smoke-snapshot-dir", default="foto_asap", help="folder foto bukti kejadian")
    p.add_argument("--smoke-keep-vehicle", action="store_true",
                   help="jangan abaikan asap yang menempel di kotak kendaraan (default: diabaikan, kemungkinan knalpot)")
    # --- mode malam ---
    p.add_argument("--night", choices=["auto", "on", "off"], default="auto",
                   help="auto = pencerahan otomatis saat gelap; on = selalu; off = tidak pernah")
    p.add_argument("--night-thresh", type=float, default=70,
                   help="kecerahan (0-255) di bawah ini dianggap malam. Naikkan kalau mode malam telat aktif")
    p.add_argument("--night-target", type=float, default=0.45,
                   help="target kecerahan setelah dicerahkan (0-1). Naikkan kalau masih kurang terang")
    p.add_argument("--clahe-clip", type=float, default=2.0,
                   help="kekuatan penambah kontras CLAHE (1.5-4). Terlalu tinggi = noise makin kelihatan")
    args = p.parse_args()
    if args.min_frames is None:
        args.min_frames = 5 if args.mode == "masuk" else 3

    if args.device == "auto":
        args.device = 0 if torch.cuda.is_available() else "cpu"
    nama_perangkat = (f"GPU {torch.cuda.get_device_name(0)}" if args.device != "cpu" and torch.cuda.is_available()
                      else "CPU")
    print(f"Perangkat: {nama_perangkat} | PyTorch {torch.__version__}")
    if nama_perangkat == "CPU":
        print("⚠️ Berjalan di CPU. Kendaraan dekat yang bergerak cepat bisa gagal ter-track kalau FPS rendah.")

    model = siapkan_model(args.model, args.models_dir)

    asap = None
    if args.no_smoke:
        print("Deteksi asap/api: MATI (--no-smoke)")
    elif os.path.isfile(args.smoke_model):
        model_asap = YOLO(args.smoke_model)
        asap = PemantauAsap(model_asap, args)
        print(f"Deteksi asap/api: AKTIF ({os.path.basename(args.smoke_model)}, kelas {list(model_asap.names.values())}, "
              f"cek tiap {args.smoke_every:g} detik)")
    else:
        print(f"⚠️ Model asap tidak ditemukan: {args.smoke_model}. Deteksi asap/api dilewati.")

    tracker_yaml = buat_tracker_yaml(args.conf, args.track_buffer)
    cap = buka_stream(args.url)
    if not cap.isOpened():
        raise SystemExit(f"Gagal membuka stream: {args.url}\nCoba buka URL-nya dulu di VLC untuk memastikan stream aktif.")

    file_baru = not os.path.exists(args.output)
    f = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if file_baru:
        writer.writerow(["camera_id", "object", "track_id", "detection_timestamp", "timestamp"])

    # --- tentukan garis hitung ---
    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Tidak bisa membaca frame pertama dari stream.")
    H, W = frame0.shape[:2]
    if args.mode == "garis" and args.set_line:
        (ax, ay), (bx, by) = pilih_garis(frame0)
        garis_frac = (ax / W, ay / H, bx / W, by / H)
        print("Garis dipilih. Pakai lagi lain kali dengan:\n  --line " +
              ",".join(f"{v:.3f}" for v in garis_frac))
    else:
        garis_frac = tuple(float(v) for v in args.line.split(","))
    garis_a = (int(garis_frac[0] * W), int(garis_frac[1] * H))
    garis_b = (int(garis_frac[2] * W), int(garis_frac[3] * H))
    print(f"Resolusi frame: {W}x{H} | garis hitung: {garis_a} -> {garis_b}")

    JUDUL = "Deteksi Kendaraan"
    if not args.no_show:
        cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL)        # jendela bisa di-resize
        cv2.resizeWindow(JUDUL, 1280, int(1280 * H / W))  # muat seluruh frame di layar

    # --- status mode malam ---
    clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(8, 8))
    mode_malam = args.night == "on"
    kecerahan_halus = None                   # kecerahan yang dihaluskan supaya tidak kedip on/off
    gamma_aktif = 1.0

    posisi_awal = {}                         # titik pertama kali track terlihat
    terakhir = {}                            # track_id -> (frame_idx, titik tengah, ukuran kotak)
    sudah_disambung = set()                  # ID lama yang sudah "diwarisi" ID baru
    sudah_bergerak = set()                   # track yang sudah terbukti bergerak (bukan objek diam)
    posisi_lama = {}                         # titik bawah-tengah kendaraan di frame sebelumnya
    kilat_garis = 0                          # efek garis berkedip saat ada yang terhitung
    jumlah_frame = defaultdict(int)          # berapa frame tiap track_id terlihat
    voting = defaultdict(Counter)            # jenis kendaraan per track_id (ambil yang paling sering)
    tercatat = {}                            # track_id yang sudah ditulis ke CSV
    hitung = Counter()                       # total per jenis

    video_out = None
    mulai = time.time()
    frame_idx = 0
    gagal = 0
    fps_halus = 0.0                          # FPS pemrosesan (dihaluskan) untuk panel
    t_frame = time.time()
    t_lapor = time.time()

    print("Mulai deteksi... tekan 'q' di jendela video untuk berhenti.")
    try:
        while True:
            if args.minutes and time.time() - mulai > args.minutes * 60:
                print("Durasi selesai.")
                break

            ok, frame = cap.read()
            if not ok and os.path.isfile(args.url):
                print("Video selesai.")
                break
            if not ok:
                gagal += 1
                if gagal > 10:
                    print("Stream putus terus, berhenti.")
                    break
                print(f"Frame gagal dibaca, reconnect ({gagal}/10)...")
                cap.release()
                time.sleep(3)
                cap = buka_stream(args.url)
                continue
            gagal = 0

            frame_idx += 1
            if frame_idx % args.skip:
                continue

            # Stream HLS kadang mulai di resolusi rendah lalu pindah ke resolusi tinggi.
            # Kalau ukuran frame berubah, hitung ulang posisi garis supaya tetap di tempat yang benar.
            if frame.shape[:2] != (H, W):
                H, W = frame.shape[:2]
                garis_a = (int(garis_frac[0] * W), int(garis_frac[1] * H))
                garis_b = (int(garis_frac[2] * W), int(garis_frac[3] * H))
                posisi_lama.clear()
                print(f"Resolusi berubah jadi {W}x{H} | garis hitung: {garis_a} -> {garis_b}")

            # Frame asli (sebelum mode malam) disimpan untuk model asap, hanya saat waktunya cek
            cek_asap = asap is not None and asap.perlu_cek()
            frame_asli = frame.copy() if cek_asap else None

            # ---------- MODE MALAM: cerahkan frame sebelum masuk YOLO ----------
            if args.night != "off":
                kecerahan = ukur_kecerahan(frame)
                kecerahan_halus = kecerahan if kecerahan_halus is None else 0.9 * kecerahan_halus + 0.1 * kecerahan
                if args.night == "auto":
                    # histeresis: masuk mode malam di bawah ambang, keluar 20 poin di atasnya
                    if not mode_malam and kecerahan_halus < args.night_thresh:
                        mode_malam = True
                        print(f"[{datetime.now():%H:%M:%S}] Mode malam AKTIF (kecerahan {kecerahan_halus:.0f})")
                    elif mode_malam and kecerahan_halus > args.night_thresh + 20:
                        mode_malam = False
                        print(f"[{datetime.now():%H:%M:%S}] Mode malam MATI (kecerahan {kecerahan_halus:.0f})")
                if mode_malam:
                    frame, gamma_aktif = tingkatkan_malam(frame, kecerahan, clahe, args.night_target)

            # conf rendah (0.1) di sini sengaja: deteksi lemah tetap dikirim ke tracker untuk
            # menyambung track yang sudah ada. Kendaraan BARU tetap butuh >= --conf.
            hasil = model.track(frame, persist=True, tracker=tracker_yaml,
                                classes=list(KELAS), conf=0.1, iou=0.6,
                                imgsz=args.imgsz, verbose=False, device=args.device)[0]

            sekarang_t = time.time()
            fps_now = 1.0 / max(sekarang_t - t_frame, 1e-6)
            fps_halus = fps_now if fps_halus == 0 else 0.9 * fps_halus + 0.1 * fps_now
            t_frame = sekarang_t
            if sekarang_t - t_lapor > 10:
                print(f"[{datetime.now():%H:%M:%S}] FPS pemrosesan: {fps_halus:.1f}"
                      + ("  <- terlalu rendah, kendaraan dekat bisa lolos" if fps_halus < 8 else ""))
                t_lapor = sekarang_t

            # ---------- DETEKSI ASAP & API ----------
            if cek_asap:
                asap.cek(frame_asli, hasil.boxes.xyxy.int().tolist())

            if hasil.boxes.id is not None:
                ids = hasil.boxes.id.int().tolist()
                kelas = hasil.boxes.cls.int().tolist()
                kotak = hasil.boxes.xyxy.int().tolist()
                konf = hasil.boxes.conf.tolist()
                ids_aktif = set(ids)

                for tid, c, cf, (x1, y1, x2, y2) in zip(ids, kelas, konf, kotak):
                    jenis = KELAS.get(c, "lainnya")
                    jumlah_frame[tid] += 1
                    # Truk/bus dengan confidence rendah tidak ikut voting jenis kendaraan
                    if not (jenis in KELAS_BERAT and cf < args.heavy_conf):
                        voting[tid][jenis] += 1

                    tengah = ((x1 + x2) // 2, y2)   # titik bawah-tengah kotak (posisi ban di jalan)
                    lama = posisi_lama.get(tid)
                    posisi_lama[tid] = tengah
                    lewat_garis = lama is not None and memotong(lama, tengah, garis_a, garis_b)
                    ukuran = max(x2 - x1, y2 - y1)
                    posisi_awal.setdefault(tid, tengah)
                    terakhir[tid] = (frame_idx, tengah, ukuran)

                    # Objek yang tidak pernah berpindah (gerobak, lapak, mobil parkir) tidak dihitung
                    if math.dist(posisi_awal[tid], tengah) >= args.min_move * ukuran:
                        sudah_bergerak.add(tid)
                    jenis_yakin = len(voting[tid]) > 0

                    if args.mode == "garis":
                        boleh_hitung = lewat_garis and jumlah_frame[tid] >= args.min_frames and jenis_yakin
                    else:
                        boleh_hitung = (jumlah_frame[tid] >= args.min_frames and tid in sudah_bergerak
                                        and jenis_yakin)

                    # Mode masuk: cek apakah ini sebenarnya kendaraan lama yang ganti ID
                    if tid not in tercatat and boleh_hitung and args.mode == "masuk":
                        awal = posisi_awal[tid]
                        kandidat, jarak_min = None, None
                        for oid, (fi, pos, ukr) in terakhir.items():
                            if (oid == tid or oid in ids_aktif or oid not in tercatat
                                    or oid in sudah_disambung or frame_idx - fi > args.merge_gap):
                                continue
                            jarak = math.dist(pos, awal)
                            if jarak <= args.merge_dist * max(ukr, ukuran) and (jarak_min is None or jarak < jarak_min):
                                kandidat, jarak_min = oid, jarak
                        if kandidat is not None:
                            tercatat[tid] = tercatat[kandidat]   # warisi status, TIDAK dihitung lagi
                            sudah_disambung.add(kandidat)
                            boleh_hitung = False

                    if tid not in tercatat and boleh_hitung:
                        label = voting[tid].most_common(1)[0][0]
                        now = datetime.now(timezone.utc)
                        writer.writerow([
                            args.camera_id, label, tid,
                            now.strftime("%Y-%m-%d %H:%M:%S"),
                            now.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        ])
                        f.flush()
                        tercatat[tid] = label
                        hitung[label] += 1
                        kilat_garis = 4

                    # warna = jenis (pakai hasil voting supaya tidak berkedip ganti-ganti)
                    if tid in tercatat:
                        jenis_tampil = tercatat[tid]
                    elif voting[tid]:
                        jenis_tampil = voting[tid].most_common(1)[0][0]
                    else:
                        jenis_tampil = jenis + "?"          # belum yakin (mis. truk confidence rendah)
                    warna = WARNA.get(jenis_tampil, (180, 180, 180))
                    sudah = tid in tercatat
                    cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 3 if sudah else 1)
                    teks = f"{jenis_tampil} #{tid}" + (" [v]" if sudah else "")
                    cv2.putText(frame, teks, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)
                    cv2.circle(frame, tengah, 3, warna, -1)

            if asap is not None:
                asap.gambar(frame)

            # garis hitung hanya digambar di mode garis
            if args.mode == "garis":
                cv2.line(frame, garis_a, garis_b, (0, 255, 0) if kilat_garis else (0, 255, 255), 3)
            kilat_garis = max(kilat_garis - 1, 0)

            # panel jumlah per jenis (+ status mode malam)
            baris = [f"TOTAL: {sum(hitung.values())}"] + [f"{k}: {v}" for k, v in hitung.most_common()]
            if mode_malam:
                baris.append(f"MALAM g={gamma_aktif:.2f}")
            baris.append(f"FPS: {fps_halus:.1f}")
            cv2.rectangle(frame, (5, 5), (220, 15 + 22 * len(baris)), (0, 0, 0), -1)
            for i, teks in enumerate(baris):
                if teks.startswith("MALAM"):
                    warna_teks = (0, 255, 255)
                elif teks.startswith("FPS"):
                    warna_teks = (0, 255, 0) if fps_halus >= 8 else (0, 0, 255)   # merah = terlalu lambat
                else:
                    warna_teks = (255, 255, 255)
                cv2.putText(frame, teks, (12, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_teks, 2)

            if args.save_video:
                if video_out is None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    fps = max(fps / args.skip, 1)
                    video_out = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                video_out.write(frame)

            if not args.no_show:
                cv2.imshow(JUDUL, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("Dihentikan.")
    finally:
        cap.release()
        f.close()
        if asap is not None:
            asap.tutup()
        if video_out is not None:
            video_out.release()
        cv2.destroyAllWindows()

    print("\n=== JUMLAH KENDARAAN TERDETEKSI PER JENIS ===")
    for jenis, n in hitung.most_common():
        print(f"{jenis:8s}: {n}")
    print(f"TOTAL   : {sum(hitung.values())}")
    print(f"Data tersimpan di: {os.path.abspath(args.output)}")
    if asap is not None:
        print(f"\n=== KEJADIAN ASAP/API: {asap.jumlah_kejadian} ===")
        print(f"Catatan kejadian: {os.path.abspath(args.smoke_output)}")
        print(f"Foto bukti      : {os.path.abspath(args.smoke_snapshot_dir)}")


if __name__ == "__main__":
    main()