"""
cek_dataset.py
Mengecek apakah struktur dataset cocok dengan face_eval.py.
Pemakaian: python cek_dataset.py --data DATASET
"""
import argparse
from collections import defaultdict
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fotos(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXT) if Path(folder).exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    d = Path(ap.parse_args().data)
    if not d.exists():
        raise SystemExit(f"Folder {d} tidak ditemukan.")

    print("Isi folder utama:", sorted(p.name for p in d.iterdir()))
    masalah = []

    # gallery
    gal = defaultdict(int)
    for p in fotos(d / "gallery"):
        rel = p.relative_to(d / "gallery").parts
        if len(rel) < 2:
            masalah.append(f"gallery: foto langsung di folder gallery, harus gallery/<subjek>/foto: {p.name}")
        else:
            gal[rel[0]] += 1
    print(f"\n[gallery] {len(gal)} subjek:", dict(gal))
    if not gal:
        masalah.append("gallery kosong atau tidak ada")

    # validasi
    val = defaultdict(int)
    for p in fotos(d / "validasi"):
        rel = p.relative_to(d / "validasi").parts
        if len(rel) >= 2:
            val[rel[0]] += 1
    print(f"[validasi] {len(val)} subjek:", dict(val) if val else "tidak ada (τ akan pakai bawaan DeepFace)")

    # uji
    uji = defaultdict(lambda: defaultdict(int))
    for p in fotos(d / "uji"):
        rel = p.relative_to(d / "uji").parts
        if len(rel) < 3:
            masalah.append(f"uji: harus uji/<kelompok>/<subjek>/foto, ditemukan: {'/'.join(rel)}")
        else:
            uji[rel[0]][rel[1]] += 1
    print(f"[uji] {len(uji)} kelompok:")
    for k, v in uji.items():
        print(f"   {k}: {sum(v.values())} foto, subjek {dict(v)}")
    if not uji:
        masalah.append("uji kosong atau tidak ada")

    # subjek uji harus ada di gallery
    semua_subjek_uji = {s for v in uji.values() for s in v}
    for s in sorted(semua_subjek_uji - set(gal)):
        if s != "unknown":
            masalah.append(f"subjek '{s}' ada di uji tapi tidak ada di gallery (nama harus sama persis)")

    # unknown
    unk = fotos(d / "unknown")
    print(f"[unknown] {len(unk)} foto" + ("" if unk else " (tidak ada, FAR akan NaN)"))

    print("\n=== HASIL ===")
    if masalah:
        for m in masalah:
            print(" -", m)
    else:
        print("Struktur cocok dengan face_eval.py.")


if __name__ == "__main__":
    main()