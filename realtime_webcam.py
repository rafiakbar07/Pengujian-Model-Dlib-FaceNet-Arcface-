"""
realtime_webcam.py
Pengujian real-time dengan protokol tetap: satu sesi = satu model, satu kondisi,
satu orang, N percobaan berjarak sama. Setiap percobaan dicatat ke hasil/realtime_log.csv.

Contoh:
  python realtime_webcam.py --model ArcFace --subjek rafi --kondisi jarak_1m --percobaan 20
  python realtime_webcam.py --model ArcFace --subjek unknown --kondisi orang_asing
  python realtime_webcam.py --model ArcFace --subjek serangan --kondisi foto_layar_hp

--subjek : nama subjek terdaftar (percobaan genuine), "unknown" (orang tidak terdaftar),
           atau "serangan" (foto cetak atau layar ponsel yang ditunjukkan ke kamera).
Tekan SPASI untuk memulai rekaman percobaan, q untuk keluar.
Threshold diambil dari hasil/metrik_ringkasan.csv (hasil tahap evaluasi). Bisa ditimpa dengan --tau.
Ulangi sesi yang sama untuk ketiga model dengan kondisi yang persis sama.
"""
import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

import face_eval as fe


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


def satu_percobaan(frame, nama, gal, tau):
    t0 = time.perf_counter()
    try:
        h = DeepFace.represent(img_path=frame, model_name=fe.MODELS[nama],
                               detector_backend=fe.DETECTOR, enforce_detection=True,
                               align=fe.ALIGN)
        ms = (time.perf_counter() - t0) * 1000
        best = max(h, key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"])
        d = fe.jarak_per_subjek(np.asarray(best["embedding"], dtype=np.float32), gal)
        terdekat = min(d, key=d.get)
        final = terdekat if d[terdekat] <= tau else "unknown"
        return dict(terdeteksi=True, d_min=d[terdekat], pred_terdekat=terdekat,
                    pred_final=final, ms=ms, kotak=best["facial_area"])
    except ValueError:
        return dict(terdeteksi=False, d_min=np.nan, pred_terdekat="", pred_final="unknown",
                    ms=(time.perf_counter() - t0) * 1000, kotak=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(fe.MODELS))
    ap.add_argument("--subjek", required=True)
    ap.add_argument("--kondisi", required=True)
    ap.add_argument("--percobaan", type=int, default=20)
    ap.add_argument("--interval", type=float, default=1.0, help="detik antar percobaan")
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--tau", type=float)
    a = ap.parse_args()

    data_dir, out_dir = Path(a.data), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gal = siapkan_gallery(data_dir, out_dir, a.model)
    tau = ambil_tau(out_dir, a.model, a.tau)
    if a.subjek not in ("unknown", "serangan") and a.subjek not in gal:
        raise SystemExit(f"Subjek '{a.subjek}' tidak ada di gallery: {sorted(gal)}")
    jenis = "genuine" if a.subjek in gal else a.subjek
    print(f"Model {a.model}, tau {tau:.4f}, jenis {jenis}. Tekan SPASI untuk mulai, q untuk keluar.")

    cap = cv2.VideoCapture(a.kamera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka.")

    log, rekam, terakhir, info = [], False, 0.0, ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tampil = frame.copy()
        if rekam and time.time() - terakhir >= a.interval:
            terakhir = time.time()
            r = satu_percobaan(frame, a.model, gal, tau)
            if jenis == "genuine":
                hasil = "benar" if r["pred_final"] == a.subjek else "salah_atau_ditolak"
            else:
                hasil = "ditolak" if r["pred_final"] == "unknown" else "DITERIMA"
            log.append(dict(waktu=datetime.now().isoformat(timespec="seconds"), model=a.model,
                            kondisi=a.kondisi, subjek=a.subjek, jenis=jenis,
                            terdeteksi=r["terdeteksi"], d_min=r["d_min"],
                            pred_terdekat=r["pred_terdekat"], pred_final=r["pred_final"],
                            ms=r["ms"], hasil=hasil, tau=tau))
            info = f"{len(log)}/{a.percobaan} pred={r['pred_final']} d={r['d_min']:.3f} {hasil}"
            if r["kotak"]:
                k = r["kotak"]
                cv2.rectangle(tampil, (k["x"], k["y"]), (k["x"] + k["w"], k["y"] + k["h"]),
                              (0, 255, 0), 2)
            if len(log) >= a.percobaan:
                break
        cv2.putText(tampil, f"{a.model} | {a.kondisi} | {'REKAM' if rekam else 'siap'} | {info}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("realtime", tampil)
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        if tombol == 32:
            rekam, terakhir = True, 0.0
    cap.release()
    cv2.destroyAllWindows()

    if log:
        f = out_dir / "realtime_log.csv"
        d = pd.DataFrame(log)
        d.to_csv(f, mode="a", header=not f.exists(), index=False)
        ringkas = d.hasil.value_counts().to_dict()
        print(f"{len(d)} percobaan tersimpan ke {f}. Ringkasan: {ringkas}. "
              f"Rata-rata {d.ms.mean():.1f} ms")


if __name__ == "__main__":
    main()
