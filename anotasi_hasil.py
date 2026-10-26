"""
anotasi_hasil.py
Menyimpan semua foto yang sudah diuji dengan bounding box wajah.
Di atas kotak ada prediksi, label asli, status, jarak, dan waktu proses.

Keluaran: foto_hasil_uji/<model>/<path foto di dataset>.jpg

Pemakaian:
  python anotasi_hasil.py --data DATASET
  python anotasi_hasil.py --data DATASET --model FaceNet
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import face_eval as fe

HIJAU, MERAH = (0, 200, 0), (0, 0, 255)


def baca(path):
    """Baca gambar lewat numpy supaya aman untuk path Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simpan(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(str(path))


def cari_kotak(DeepFace, img):
    """Kotak wajah terbesar (aturan yang sama dengan face_eval.py), None bila tidak ada wajah."""
    try:
        hasil = DeepFace.extract_faces(img_path=img, detector_backend=fe.DETECTOR,
                                       align=fe.ALIGN, enforce_detection=True)
    except ValueError:
        return None
    best = max(hasil, key=lambda h: h["facial_area"]["w"] * h["facial_area"]["h"])
    k = best["facial_area"]
    return int(k["x"]), int(k["y"]), int(k["w"]), int(k["h"])


def status_baris(r):
    """Mengembalikan (berhasil, status) dengan aturan yang sama seperti evaluasi."""
    if r.split == "genuine":
        if r.pred_final == r.subjek:
            return True, "BENAR"
        if not r.terdeteksi:
            return False, "TAK TERDETEKSI"
        if r.pred_final == "unknown":
            return False, "DITOLAK"
        return False, "TERTUKAR"
    ok = r.pred_final == "unknown"
    return ok, "DITOLAK (benar)" if ok else "DITERIMA (salah)"


def gambar(img, kotak, baris, warna):
    h, w = img.shape[:2]
    skala = max(0.5, w / 1200)
    tebal = max(1, int(round(skala * 2)))
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    ukur = [cv2.getTextSize(t, font, skala, tebal) for t in baris]
    tinggi = [u[0][1] + u[1] + 8 for u in ukur]
    lebar_maks = max(u[0][0] for u in ukur) + 8

    if kotak:
        x, y, bw, bh = kotak
        cv2.rectangle(out, (x, y), (x + bw, y + bh), warna, tebal + 1)
        atas = y - sum(tinggi)
        if atas < 0:          # tidak muat di atas kotak, taruh di dalam kotak
            atas = y
        x0 = x
    else:
        atas, x0 = 10, 10
    x0 = max(0, min(x0, w - lebar_maks))

    cy = atas
    for t, u, th in zip(baris, ukur, tinggi):
        cv2.rectangle(out, (x0, cy), (x0 + u[0][0] + 8, cy + th), warna, -1)
        cv2.putText(out, t, (x0 + 4, cy + u[0][1] + 4), font, skala, (255, 255, 255),
                    tebal, cv2.LINE_AA)
        cy += th
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--hasil", default="hasil")
    ap.add_argument("--out", default="foto_hasil_uji")
    ap.add_argument("--model", choices=list(fe.MODELS), help="kosongkan untuk semua model")
    a = ap.parse_args()

    data_dir, out_dir = Path(a.data), Path(a.out)
    p = pd.read_csv(Path(a.hasil) / "hasil_prediksi.csv")
    p["terdeteksi"] = p.terdeteksi.astype(bool)
    models = [a.model] if a.model else list(dict.fromkeys(p.model))

    from deepface import DeepFace  # import di sini agar start lebih ringan

    files = list(dict.fromkeys(p.file))
    print(f"{len(files)} foto x {len(models)} model -> {out_dir}")
    jumlah = 0
    for i, rel in enumerate(files, 1):
        img = baca(data_dir / rel)
        if img is None:
            print(f"  LEWAT (gagal dibaca): {rel}")
            continue
        kotak = cari_kotak(DeepFace, img)  # sekali per foto, dipakai semua model
        for m in models:
            r = next(p[(p.file == rel) & (p.model == m)].itertuples())
            ok, status = status_baris(r)
            asli = "serangan" if r.split == "serangan" else r.subjek
            jarak = f"{r.d_min:.3f}" if pd.notna(r.d_min) else "-"
            baris = [f"{m} | pred: {r.pred_final} | asli: {asli}",
                     f"{status} | {r.ms:.0f} ms | d={jarak}"]
            hasil_img = gambar(img, kotak, baris, HIJAU if ok else MERAH)
            simpan(out_dir / m / Path(rel).with_suffix(".jpg"), hasil_img)
            jumlah += 1
        if i % 10 == 0:
            print(f"  {i}/{len(files)}")
    print(f"Selesai. {jumlah} gambar tersimpan di {out_dir}")


if __name__ == "__main__":
    main()