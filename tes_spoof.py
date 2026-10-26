"""
tes_spoof.py
Mengetes anti-spoofing DeepFace hanya pada foto serangan (tanpa model pengenalan).
Pemakaian:
  python tes_spoof.py --data DATASET
  python tes_spoof.py --data DATASET --sertakan-genuine   (opsional, lebih lama)
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from deepface import DeepFace

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def foto(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT) if folder.exists() else []


def periksa(path):
    t0 = time.perf_counter()
    try:
        hasil = DeepFace.extract_faces(img_path=str(path), detector_backend="mtcnn",
                                       align=True, enforce_detection=True,
                                       anti_spoofing=True)
    except ValueError as e:
        ms = (time.perf_counter() - t0) * 1000
        if "spoof" in str(e).lower():  # sebagian versi DeepFace melempar error saat spoof
            return dict(terdeteksi=True, is_real=False, skor=None, ms=ms, catatan="spoof (error)")
        return dict(terdeteksi=False, is_real=None, skor=None, ms=ms, catatan=str(e)[:60])
    ms = (time.perf_counter() - t0) * 1000
    best = max(hasil, key=lambda h: h["facial_area"]["w"] * h["facial_area"]["h"])
    return dict(terdeteksi=True, is_real=best.get("is_real"), skor=best.get("antispoof_score"),
                ms=ms, catatan="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--hasil", default="hasil")
    ap.add_argument("--sertakan-genuine", action="store_true")
    a = ap.parse_args()
    d = Path(a.data)

    daftar = [("serangan", p) for p in foto(d / "uji" / "serangan_foto")]
    if a.sertakan_genuine:
        for p in foto(d / "uji"):
            if "serangan_foto" not in [x.lower() for x in p.parts]:
                daftar.append(("genuine", p))
    if not daftar:
        raise SystemExit(f"Tidak ada foto di {d / 'uji' / 'serangan_foto'}")

    print(f"{len(daftar)} foto diperiksa...")
    rows = []
    for jenis, p in daftar:
        r = periksa(p)
        if not r["terdeteksi"]:
            vonis = "TAK TERDETEKSI"
        elif jenis == "serangan":
            vonis = "DITOLAK (benar)" if r["is_real"] is False else "LOLOS (salah)"
        else:
            vonis = "DITERIMA (benar)" if r["is_real"] is True else "SALAH DITOLAK"
        rows.append(dict(jenis=jenis, file=p.name, vonis=vonis, is_real=r["is_real"],
                         skor=r["skor"], ms=round(r["ms"]), catatan=r["catatan"]))
        print(f"  {jenis:9s} {p.name:28s} {vonis:18s} skor={r['skor']}  {r['ms']:.0f} ms")

    df = pd.DataFrame(rows)
    out = Path(a.hasil)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "tes_spoof.csv", index=False)

    print("\n=== Ringkasan ===")
    print(df.groupby(["jenis", "vonis"]).size().to_string())
    s = df[df.jenis == "serangan"]
    if len(s):
        lolos = (s.vonis == "LOLOS (salah)").sum()
        print(f"\nAPCER (serangan lolos): {lolos}/{len(s)} = {lolos / len(s):.3f}")
    print("Tersimpan di", out / "tes_spoof.csv")


if __name__ == "__main__":
    main()