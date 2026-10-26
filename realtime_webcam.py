"""
realtime_webcam.py (versi bebas)
Pengenalan wajah real-time yang langsung berjalan tanpa tombol mulai dan tanpa batas percobaan.

Perubahan dari versi protokol:
  - Deteksi langsung berjalan begitu kamera terbuka (tanpa SPASI).
  - Tidak ada batas 20 percobaan. Berhenti hanya dengan q atau ESC.
  - Setiap wajah diberi kotak dan label berisi nama tebakan, status DITERIMA atau DITOLAK,
    jarak, dan waktu prediksi. Teks berwarna hitam di atas latar terang agar terbaca.
  - Pengenalan berjalan di thread terpisah, jadi tampilan kamera tetap lancar walau
    pengenalan butuh 1 sampai 2 detik per bingkai.

Contoh:
  python realtime_webcam.py --model FaceNet
  python realtime_webcam.py --model ArcFace --subjek rafi --kondisi jarak_1m

--subjek dan --kondisi bersifat opsional. Keduanya hanya dicatat ke log sebagai keterangan.
Log tersimpan ke hasil/realtime_log_bebas.csv (terpisah dari realtime_log.csv versi lama).
Threshold diambil dari hasil/metrik_ringkasan.csv. Bisa ditimpa dengan --tau.

Catatan: fungsi siapkan_gallery dan ambil_tau dipakai juga oleh realtime_multi.py,
jadi namanya jangan diubah.
"""
import argparse
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

import face_eval as fe

HITAM = (0, 0, 0)
PUTIH = (255, 255, 255)
HIJAU = (0, 170, 0)
MERAH = (0, 0, 255)
LATAR_HIJAU = (170, 240, 170)
LATAR_MERAH = (170, 170, 255)


def siapkan_gallery(data_dir, out_dir, nama):
    cache_file = out_dir / "cache_embedding.pkl"
    cache = pickle.load(open(cache_file, "rb")) if cache_file.exists() else {}
    df = fe.kumpulkan_data(data_dir)
    gal_rows = df[df.split == "gallery"]
    baru = False
    for r in gal_rows.itertuples():
        key = (nama, r.rel)
        if key not in cache:
            h = DeepFace.represent(img_path=str(r.path), model_name=fe.MODELS[nama],
                                   detector_backend=fe.DETECTOR, enforce_detection=True,
                                   align=fe.ALIGN)
            best = max(h, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"])
            cache[key] = dict(emb=np.asarray(best["embedding"], dtype=np.float32), ms=0.0, err="")
            baru = True
    if baru:
        out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(cache_file, "wb"))
    return fe.bangun_gallery(df, cache, nama)


def ambil_tau(out_dir, nama, tau_manual):
    if tau_manual is not None:
        return tau_manual
    f = out_dir / "metrik_ringkasan.csv"
    if not f.exists():
        raise SystemExit("hasil/metrik_ringkasan.csv belum ada. Jalankan face_eval.py evaluasi "
                         "dulu atau berikan --tau.")
    r = pd.read_csv(f)
    return float(r[r.model == nama].tau.iloc[0])


def perkecil(frame, lebar_proses):
    """Perkecil bingkai untuk diproses. Mengembalikan (gambar, skala)."""
    h, w = frame.shape[:2]
    if lebar_proses and w > lebar_proses:
        skala = lebar_proses / w
        return cv2.resize(frame, (lebar_proses, int(h * skala)), interpolation=cv2.INTER_AREA), skala
    return frame, 1.0


def pekerja(state, lock, stop, a, gal, tau, log):
    """Thread pengenalan: selalu mengambil bingkai terbaru, bukan antrean bingkai lama."""
    id_terakhir = -1
    galat_dicetak = False
    while not stop.is_set():
        with lock:
            frame, fid = state["frame"], state["id"]
        if frame is None or fid == id_terakhir:
            time.sleep(0.005)
            continue
        id_terakhir = fid
        t0 = time.perf_counter()
        try:
            kecil, skala = perkecil(frame, a.lebar_proses)
            try:
                h = DeepFace.represent(img_path=kecil, model_name=fe.MODELS[a.model],
                                       detector_backend=fe.DETECTOR, enforce_detection=True,
                                       align=fe.ALIGN)
            except ValueError:
                h = []
            h = sorted(h, key=lambda x: x["facial_area"]["x"])
            wajah = []
            for w in h:
                d = fe.jarak_per_subjek(np.asarray(w["embedding"], dtype=np.float32), gal)
                terdekat = min(d, key=d.get)
                final = terdekat if d[terdekat] <= tau else "unknown"
                k = w["facial_area"]
                wajah.append(dict(x=int(k["x"] / skala), y=int(k["y"] / skala),
                                  w=int(k["w"] / skala), h=int(k["h"] / skala),
                                  pred_final=final, pred_terdekat=terdekat,
                                  d_min=float(d[terdekat]),
                                  akses="DITOLAK" if final == "unknown" else "DITERIMA"))
            ms = (time.perf_counter() - t0) * 1000
            for w in wajah:
                w["ms"] = ms
            with lock:
                state["hasil"] = wajah
                state["waktu"] = time.time()
                state["proses"] = ms / 1000
            for i, w in enumerate(wajah):
                log.append(dict(waktu=datetime.now().isoformat(timespec="seconds"),
                                model=a.model, kondisi=a.kondisi, subjek_acuan=a.subjek,
                                n_wajah=len(wajah), posisi=i + 1, pred_terdekat=w["pred_terdekat"],
                                pred_final=w["pred_final"], d_min=w["d_min"], akses=w["akses"],
                                ms=ms, tau=tau, lebar_proses=a.lebar_proses))
        except Exception as e:  # jangan sampai thread mati karena satu bingkai bermasalah
            if not galat_dicetak:
                print(f"Peringatan pada thread pengenalan: {e!r}")
                galat_dicetak = True
            time.sleep(0.3)


def gambar_wajah(img, w):
    """Kotak wajah, lalu label dua baris berteks hitam di atas (atau di bawah) kotak."""
    diterima = w["akses"] == "DITERIMA"
    warna = HIJAU if diterima else MERAH
    latar = LATAR_HIJAU if diterima else LATAR_MERAH
    x, y, lebar, tinggi = w["x"], w["y"], w["w"], w["h"]
    cv2.rectangle(img, (x, y), (x + lebar, y + tinggi), warna, 2)

    baris1 = w["pred_final"]
    baris2 = f'{w["akses"]}  d={w["d_min"]:.2f}  {w["ms"]:.0f} ms'
    font = cv2.FONT_HERSHEY_SIMPLEX
    s1, t1, s2, t2 = 0.75, 2, 0.5, 1
    (w1, h1), _ = cv2.getTextSize(baris1, font, s1, t1)
    (w2, h2), _ = cv2.getTextSize(baris2, font, s2, t2)

    H, W = img.shape[:2]
    lbl_w = max(w1, w2) + 14
    lbl_h = h1 + h2 + 22
    x0 = int(min(max(x, 0), max(W - lbl_w, 0)))
    y0 = y - lbl_h
    if y0 < 0:
        y0 = min(y + tinggi, H - lbl_h)
    y0 = int(max(y0, 0))

    cv2.rectangle(img, (x0, y0), (x0 + lbl_w, y0 + lbl_h), latar, -1)
    cv2.rectangle(img, (x0, y0), (x0 + lbl_w, y0 + lbl_h), warna, 1)
    cv2.putText(img, baris1, (x0 + 7, y0 + h1 + 7), font, s1, HITAM, t1, cv2.LINE_AA)
    cv2.putText(img, baris2, (x0 + 7, y0 + lbl_h - 7), font, s2, HITAM, t2, cv2.LINE_AA)


def gambar_status(img, teks):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(teks, font, 0.55, 1)
    cv2.rectangle(img, (0, 0), (tw + 18, th + 18), PUTIH, -1)
    cv2.putText(img, teks, (9, th + 9), font, 0.55, HITAM, 1, cv2.LINE_AA)


def buka_kamera(indeks):
    cap = cv2.VideoCapture(indeks, cv2.CAP_DSHOW)  # DirectShow biasanya lebih lancar di Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(indeks)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera "
                         "atau coba --kamera 1.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # lebih ringan untuk 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # kurangi bingkai lama yang menumpuk
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(fe.MODELS))
    ap.add_argument("--subjek", default="-", help="opsional, hanya dicatat ke log")
    ap.add_argument("--kondisi", default="bebas", help="opsional, hanya dicatat ke log")
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--lebar_proses", type=int, default=640,
                    help="lebar bingkai saat dikenali (px). 0 berarti ukuran asli, lebih lambat")
    ap.add_argument("--kadaluarsa", type=float, default=3.0,
                    help="kotak disembunyikan bila hasil lebih tua dari ini (detik)")
    a = ap.parse_args()

    data_dir, out_dir = Path(a.data), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gal = siapkan_gallery(data_dir, out_dir, a.model)
    tau = ambil_tau(out_dir, a.model, a.tau)

    print("Memanaskan model, tunggu sebentar...")
    try:
        DeepFace.represent(img_path=np.zeros((160, 160, 3), np.uint8), model_name=fe.MODELS[a.model],
                           detector_backend=fe.DETECTOR, enforce_detection=False, align=fe.ALIGN)
    except Exception:
        pass
    print(f"Model {a.model}, tau {tau:.4f}. Tekan q atau ESC di jendela kamera untuk keluar.")

    cap = buka_kamera(a.kamera)
    state = dict(frame=None, id=0, hasil=[], waktu=0.0, proses=0.0)
    lock, stop, log = threading.Lock(), threading.Event(), []
    th = threading.Thread(target=pekerja, args=(state, lock, stop, a, gal, tau, log), daemon=True)
    th.start()

    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with lock:
                state["frame"] = frame
                state["id"] += 1
                hasil, waktu, proses = state["hasil"], state["waktu"], state["proses"]

            tampil = frame.copy()
            segar = (time.time() - waktu) <= a.kadaluarsa
            if hasil and segar:
                for w in hasil:
                    gambar_wajah(tampil, w)

            t_now = time.perf_counter()
            dt = max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps = (1 / dt) if fps == 0 else 0.9 * fps + 0.1 * (1 / dt)
            if waktu == 0.0:
                info = "menunggu hasil pertama"
            else:
                info = f"wajah {len(hasil) if segar else 0}"
            gambar_status(tampil, f"{a.model} | tau {tau:.4f} | kamera {fps:.0f} FPS | "
                                  f"proses {proses:.1f} s/bingkai | {info} | q: keluar")
            cv2.imshow("realtime", tampil)
            tombol = cv2.waitKey(1) & 0xFF
            if tombol in (ord("q"), 27):
                break
    finally:
        stop.set()
        th.join(timeout=3)
        cap.release()
        cv2.destroyAllWindows()
        if log:
            f = out_dir / "realtime_log_bebas.csv"
            d = pd.DataFrame(log)
            d.to_csv(f, mode="a", header=not f.exists(), index=False)
            print(f"{len(d)} baris tersimpan ke {f}. Ringkasan akses: {d.akses.value_counts().to_dict()}. "
                  f"Median waktu proses {d.ms.median():.0f} ms")


if __name__ == "__main__":
    main()