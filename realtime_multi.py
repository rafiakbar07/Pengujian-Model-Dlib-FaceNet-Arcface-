"""
realtime_multi.py
Pengujian real-time BEBERAPA orang sekaligus dalam satu bingkai.

Aturan pemetaan: wajah diurutkan dari kiri ke kanan menurut posisi di layar
(bukan kiri kanan badan orang). --subjek diisi sesuai urutan itu.

Contoh (rafi paling kiri di layar, adit paling kanan):
  python realtime_multi.py --model ArcFace --subjek rafi,roif,adit --kondisi tiga_orang --percobaan 20

Aturan percobaan:
  - Jumlah wajah terdeteksi harus SAMA dengan jumlah subjek. Bila berbeda,
    semua subjek pada percobaan itu dicatat gagal (terdeteksi=False), konsisten
    dengan realtime_webcam.py yang menghitung wajah tak terdeteksi sebagai gagal.
  - Satu percobaan menghasilkan satu baris per subjek. Kolom percobaan_ke
    dipakai untuk mengelompokkan baris yang berasal dari bingkai yang sama.
  - Kolom ms adalah waktu seluruh bingkai, bukan per wajah. Untuk rata-rata waktu,
    ambil satu baris per percobaan_ke.

Log disimpan ke hasil/realtime_log_multi.csv (file terpisah dari realtime_log.csv
karena kolomnya berbeda, jadi jangan digabung mentah).

Tekan SPASI untuk mulai, q untuk keluar.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

import face_eval as fe
from realtime_webcam import siapkan_gallery, ambil_tau


def percobaan_multi(frame, nama, gal, tau, urutan):
    """Mengembalikan (jumlah_wajah, ms, daftar hasil per subjek sesuai urutan)."""
    t0 = time.perf_counter()
    try:
        h = DeepFace.represent(img_path=frame, model_name=fe.MODELS[nama],
                               detector_backend=fe.DETECTOR, enforce_detection=True,
                               align=fe.ALIGN)
    except ValueError:
        h = []
    ms = (time.perf_counter() - t0) * 1000
    n = len(h)

    gagal = dict(terdeteksi=False, d_min=np.nan, pred_terdekat="", pred_final="unknown",
                 kotak=None)
    if n != len(urutan):
        return n, ms, [dict(subjek=s, posisi=i + 1, **gagal) for i, s in enumerate(urutan)]

    h = sorted(h, key=lambda x: x["facial_area"]["x"] + x["facial_area"]["w"] / 2)
    hasil = []
    for i, (s, w) in enumerate(zip(urutan, h)):
        d = fe.jarak_per_subjek(np.asarray(w["embedding"], dtype=np.float32), gal)
        terdekat = min(d, key=d.get)
        final = terdekat if d[terdekat] <= tau else "unknown"
        hasil.append(dict(subjek=s, posisi=i + 1, terdeteksi=True, d_min=d[terdekat],
                          pred_terdekat=terdekat, pred_final=final,
                          kotak=w["facial_area"]))
    return n, ms, hasil


def gambar_hasil(tampil, hasil):
    """Kotak dan label tetap tampil sampai percobaan berikutnya."""
    for r in hasil:
        k = r["kotak"]
        if not k:
            continue
        benar = r["pred_final"] == r["subjek"]
        warna = (0, 200, 0) if benar else (0, 0, 255)
        cv2.rectangle(tampil, (k["x"], k["y"]), (k["x"] + k["w"], k["y"] + k["h"]), warna, 2)
        cv2.putText(tampil, f"{r['pred_final']} {r['d_min']:.2f}", (k["x"], max(k["y"] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(fe.MODELS))
    ap.add_argument("--subjek", required=True,
                    help="nama subjek dipisah koma, urutan kiri ke kanan di layar")
    ap.add_argument("--kondisi", required=True)
    ap.add_argument("--percobaan", type=int, default=20)
    ap.add_argument("--interval", type=float, default=1.0, help="detik antar percobaan")
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--simpan_frame", action="store_true",
                    help="simpan tiap bingkai uji ke hasil/frames_multi agar bisa diuji ulang")
    a = ap.parse_args()

    urutan = [s.strip() for s in a.subjek.split(",") if s.strip()]
    if len(urutan) < 2:
        raise SystemExit("Isi minimal dua subjek, misalnya --subjek rafi,roif,adit")

    data_dir, out_dir = Path(a.data), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gal = siapkan_gallery(data_dir, out_dir, a.model)
    tau = ambil_tau(out_dir, a.model, a.tau)
    hilang = [s for s in urutan if s not in gal]
    if hilang:
        raise SystemExit(f"Subjek tidak ada di gallery: {hilang}. Tersedia: {sorted(gal)}")

    folder_frame = out_dir / "frames_multi"
    if a.simpan_frame:
        folder_frame.mkdir(parents=True, exist_ok=True)

    print(f"Model {a.model}, tau {tau:.4f}, urutan kiri ke kanan: {urutan}.")
    print("Tekan SPASI untuk mulai, q untuk keluar.")

    cap = cv2.VideoCapture(a.kamera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka.")

    f_log = out_dir / "realtime_log_multi.csv"
    log, rekam, terakhir, info, terakhir_hasil = [], False, 0.0, "", []
    n_percobaan = 0

    def simpan_log():
        if not log:
            return
        d = pd.DataFrame(log)
        d.to_csv(f_log, mode="a", header=not f_log.exists(), index=False)
        log.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tampil = frame.copy()

            if rekam and time.time() - terakhir >= a.interval:
                terakhir = time.time()
                n_percobaan += 1
                n_wajah, ms, hasil = percobaan_multi(frame, a.model, gal, tau, urutan)
                terakhir_hasil = hasil
                if a.simpan_frame:
                    nama_file = f"{a.model}_{a.kondisi}_{datetime.now():%Y%m%d_%H%M%S}_{n_percobaan:02d}.jpg"
                    cv2.imwrite(str(folder_frame / nama_file), frame)
                for r in hasil:
                    benar = r["pred_final"] == r["subjek"]
                    log.append(dict(waktu=datetime.now().isoformat(timespec="seconds"),
                                    model=a.model, kondisi=a.kondisi, percobaan_ke=n_percobaan,
                                    n_wajah=n_wajah, posisi=r["posisi"], subjek=r["subjek"],
                                    jenis="genuine", terdeteksi=r["terdeteksi"],
                                    d_min=r["d_min"], pred_terdekat=r["pred_terdekat"],
                                    pred_final=r["pred_final"], ms=ms,
                                    hasil="benar" if benar else "salah_atau_ditolak", tau=tau))
                simpan_log_sementara = len(log)
                info = (f"{n_percobaan}/{a.percobaan} wajah={n_wajah} "
                        f"benar={sum(r['pred_final'] == r['subjek'] for r in hasil)}/{len(urutan)}")
                if simpan_log_sementara and n_percobaan % 5 == 0:
                    simpan_log()  # simpan berkala agar data tidak hilang bila program berhenti
                if n_percobaan >= a.percobaan:
                    break

            gambar_hasil(tampil, terakhir_hasil)
            cv2.putText(tampil, f"{a.model} | {a.kondisi} | {'REKAM' if rekam else 'siap'} | {info}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("realtime_multi", tampil)
            tombol = cv2.waitKey(1) & 0xFF
            if tombol == ord("q"):
                break
            if tombol == 32:
                rekam, terakhir = True, 0.0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        simpan_log()

    if f_log.exists():
        d = pd.read_csv(f_log)
        d = d[(d.model == a.model) & (d.kondisi == a.kondisi)]
        print(f"Log tersimpan di {f_log}. Ringkasan sesi: {d.hasil.value_counts().to_dict()}")


if __name__ == "__main__":
    main()