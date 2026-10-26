"""
update_spoof.py
Menjalankan anti-spoofing HANYA pada foto uji/serangan_foto, lalu:
  1) memperbarui hasil/hasil_prediksi.csv (serangan yang dinilai palsu -> pred_final = unknown)
  2) menghitung ulang APCER di hasil/metrik_ringkasan.csv dan hasil/metrik_per_kelompok.csv
  3) menimpa gambar serangan di foto_hasil_uji/<model>/... dengan label anti-spoof
File asli dicadangkan sekali sebagai *_sebelum_spoof.csv. Aman dijalankan berulang.

Pemakaian:
  python update_spoof.py --data DATASET
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from deepface import DeepFace

import face_eval as fe
from anotasi_hasil import HIJAU, MERAH, baca, cari_kotak, gambar, simpan, status_baris


def cadangkan(f):
    if not f.exists():
        raise SystemExit(f"{f} tidak ditemukan. Jalankan face_eval.py evaluasi dulu.")
    b = f.with_name(f.stem + "_sebelum_spoof" + f.suffix)
    if not b.exists():
        shutil.copy2(f, b)


def apcer(g):
    g = g[g.terdeteksi]
    return float((g.pred_final != "unknown").mean()) if len(g) else np.nan


def cek_spoof(path):
    """Mengembalikan is_real (True/False/None), skor, kotak wajah, dan waktu (ms)."""
    t0 = time.perf_counter()
    try:
        hasil = DeepFace.extract_faces(img_path=str(path), detector_backend=fe.DETECTOR,
                                       align=fe.ALIGN, enforce_detection=True,
                                       anti_spoofing=True)
    except ValueError as e:
        ms = (time.perf_counter() - t0) * 1000
        if "spoof" in str(e).lower():  # sebagian versi DeepFace melempar error saat spoof
            return dict(is_real=False, skor=np.nan, kotak=None, ms=ms)
        return dict(is_real=None, skor=np.nan, kotak=None, ms=ms)
    ms = (time.perf_counter() - t0) * 1000
    best = max(hasil, key=lambda h: h["facial_area"]["w"] * h["facial_area"]["h"])
    k = best["facial_area"]
    skor = best.get("antispoof_score")
    return dict(is_real=bool(best.get("is_real", True)),
                skor=float(skor) if skor is not None else np.nan,
                kotak=(int(k["x"]), int(k["y"]), int(k["w"]), int(k["h"])), ms=ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--hasil", default="hasil")
    ap.add_argument("--out", default="foto_hasil_uji")
    a = ap.parse_args()
    data_dir, hasil_dir, out_dir = Path(a.data), Path(a.hasil), Path(a.out)

    f_pred = hasil_dir / "hasil_prediksi.csv"
    f_ring = hasil_dir / "metrik_ringkasan.csv"
    f_kel = hasil_dir / "metrik_per_kelompok.csv"
    for f in (f_pred, f_ring, f_kel):
        cadangkan(f)

    p = pd.read_csv(f_pred)
    p["terdeteksi"] = p.terdeteksi.astype(bool)
    if "pred_final_sebelum_spoof" not in p:
        p["pred_final_sebelum_spoof"] = p.pred_final
    for kol, awal in (("spoof_vonis", ""), ("spoof_skor", np.nan), ("spoof_ms", np.nan)):
        if kol not in p:
            p[kol] = awal
    p["spoof_vonis"] = p.spoof_vonis.fillna("").astype(object)

    files = list(dict.fromkeys(p[p.split == "serangan"].file))
    if not files:
        raise SystemExit("Tidak ada foto serangan di hasil_prediksi.csv")
    models = list(dict.fromkeys(p.model))

    print(f"Memeriksa {len(files)} foto serangan (anti-spoof)...")
    info = {}
    for rel in files:
        img = baca(data_dir / rel)
        if img is None:
            print(f"  LEWAT (gagal dibaca): {rel}")
            continue
        r = cek_spoof(data_dir / rel)
        if r["kotak"] is None:
            r["kotak"] = cari_kotak(DeepFace, img)
        info[rel] = (r, img)
        label = {True: "ASLI (lolos)", False: "PALSU (ditolak)", None: "tak terdeteksi"}[r["is_real"]]
        print(f"  {Path(rel).name:26s} {label:16s} skor={r['skor']:.3f}  {r['ms']:.0f} ms")

    # 1) perbarui hasil_prediksi.csv
    for rel, (r, img) in info.items():
        m = (p.file == rel) & (p.split == "serangan")
        p.loc[m, "spoof_vonis"] = {True: "asli", False: "palsu", None: ""}[r["is_real"]]
        p.loc[m, "spoof_skor"] = r["skor"]
        p.loc[m, "spoof_ms"] = round(r["ms"], 1)
        if r["is_real"] is False:
            p.loc[m, "pred_final"] = "unknown"
        else:
            p.loc[m, "pred_final"] = p.loc[m, "pred_final_sebelum_spoof"]
    p.to_csv(f_pred, index=False)

    # 2) hitung ulang APCER
    ring = pd.read_csv(f_ring)
    lama = dict(zip(pd.read_csv(f_ring.with_name("metrik_ringkasan_sebelum_spoof.csv")).model,
                    pd.read_csv(f_ring.with_name("metrik_ringkasan_sebelum_spoof.csv")).APCER_serangan_foto))
    for i, row in ring.iterrows():
        ring.loc[i, "APCER_serangan_foto"] = apcer(p[(p.model == row.model) & (p.split == "serangan")])
    ring["anti_spoofing"] = "ya (foto serangan)"
    ring.to_csv(f_ring, index=False)

    kel = pd.read_csv(f_kel)
    for i, row in kel.iterrows():
        if row.kelompok == fe.KELOMPOK_SERANGAN:
            kel.loc[i, "FAR_atau_APCER"] = apcer(
                p[(p.model == row.model) & (p.kelompok == fe.KELOMPOK_SERANGAN)])
    kel.to_csv(f_kel, index=False)

    # 3) gambar
    for rel, (r, img) in info.items():
        if r["is_real"] is None:
            spoof = "Anti-spoof: -"
        else:
            spoof = (f"Anti-spoof: {'ASLI' if r['is_real'] else 'PALSU'} | "
                     f"skor {r['skor']:.2f} | {r['ms']:.0f} ms")
        for m in models:
            row = next(p[(p.file == rel) & (p.model == m)].itertuples())
            ok, status = status_baris(row)
            jarak = f"{row.d_min:.3f}" if pd.notna(row.d_min) else "-"
            baris = [f"{m} | pred: {row.pred_final} | asli: serangan",
                     f"{status} | {row.ms:.0f} ms | d={jarak}", spoof]
            simpan(out_dir / m / Path(rel).with_suffix(".jpg"),
                   gambar(img, r["kotak"], baris, HIJAU if ok else MERAH))

    print("\n=== APCER (serangan foto yang lolos) ===")
    for m in models:
        print(f"  {m:8s} sebelum: {lama.get(m, float('nan')):.4f}   sesudah: "
              f"{apcer(p[(p.model == m) & (p.split == 'serangan')]):.4f}")
    print(f"\nData diperbarui di {hasil_dir}, gambar serangan diperbarui di {out_dir}")


if __name__ == "__main__":
    main()