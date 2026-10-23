"""
perbaiki_struktur.py
Merapikan DATASET:
 1) nama folder (GALLERY, Validasi, RAFI, ...) jadi huruf kecil
 2) foto uji/<SUBJEK>/foto -> uji/<kelompok>/<subjek>/foto
    subjek di daftar UNKNOWN -> unknown/<subjek>/foto

Pemakaian:
  python perbaiki_struktur.py --data DATASET             (dry-run)
  python perbaiki_struktur.py --data DATASET --terapkan
"""
import argparse
import re
import shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUBJEK = ["roif", "rafi", "adit"]
UNKNOWN = ["adit"]  # subjek yang tidak punya gallery -> dijadikan orang asing

# kode (huruf kecil, persis seperti di nama file setelah "subjek_") -> kelompok
KODE = {
    "f": "baseline",
    "l": "pose", "r": "pose", "u": "pose", "d": "pose", "t": "pose",
    "tl": "kemiringan", "tr": "kemiringan",
    "lf": "pencahayaan", "lb": "pencahayaan", "ll": "pencahayaan",
    "lr": "pencahayaan", "dim": "pencahayaan", "dark": "pencahayaan",
    "d0.3": "jarak", "d1": "jarak", "d2": "jarak",
    "curang1": "serangan_foto", "curang2": "serangan_foto",
}


def pecah(stem):
    """'ROIF_CURANG1' -> ('roif', 'curang1'); 'adit_D0.3' -> ('adit', 'd0.3')."""
    s = stem.lower()
    for nama in sorted(SUBJEK, key=len, reverse=True):
        if s.startswith(nama):
            sisa = re.sub(r"^[\s_\-]+", "", s[len(nama):])
            if sisa in KODE:
                return nama, sisa
            return nama, re.sub(r"\d+$", "", sisa)  # cadangan: buang nomor urut
    return None, None


def huruf_kecil(p, terapkan):
    baru = p.name.lower()
    if p.name == baru:
        return
    print(f"  rename: {p.name} -> {baru}")
    if terapkan:
        tmp = p.with_name(p.name + "__tmp")  # perlu perantara di Windows
        p.rename(tmp)
        tmp.rename(p.with_name(baru))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="DATASET")
    ap.add_argument("--terapkan", action="store_true")
    a = ap.parse_args()
    root = Path(a.data)
    if not root.exists():
        raise SystemExit(f"{root} tidak ditemukan")

    print("Langkah 1: huruf kecil pada nama folder")
    for split in [p for p in root.iterdir() if p.is_dir()]:
        for sub in [p for p in split.iterdir() if p.is_dir()]:
            huruf_kecil(sub, a.terapkan)
        huruf_kecil(split, a.terapkan)

    print("\nLangkah 2: pindahkan foto uji")
    uji = next((p for p in root.iterdir() if p.is_dir() and p.name.lower() == "uji"), None)
    if uji is None:
        raise SystemExit("folder uji tidak ada")
    rencana, tak = [], {}
    for sub in [p for p in uji.iterdir() if p.is_dir() and p.name.lower() in SUBJEK]:
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() not in EXT:
                continue
            subjek, kode = pecah(f.stem)
            if subjek is None or kode not in KODE:
                tak.setdefault(kode or "(subjek tak dikenal)", []).append(f.name)
                continue
            if subjek in UNKNOWN:
                dst = root / "unknown" / subjek / f.name.lower()
            else:
                dst = root / "uji" / KODE[kode] / subjek / f.name.lower()
            rencana.append((f, dst))

    ringkas = {}
    for f, dst in rencana:
        kunci = "/".join(dst.relative_to(root).parts[:-1])
        ringkas[kunci] = ringkas.get(kunci, 0) + 1
        if a.terapkan:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
    for k, n in sorted(ringkas.items()):
        print(f"  {k}: {n} foto")

    if a.terapkan:
        for sub in [p for p in uji.iterdir() if p.is_dir() and p.name.lower() in SUBJEK]:
            if not any(sub.rglob("*.*")):
                shutil.rmtree(sub, ignore_errors=True)
    print(f"\n{len(rencana)} foto {'dipindah' if a.terapkan else 'akan dipindah'}.")
    if tak:
        print("\nTidak dipindah (kode belum terdaftar di KODE):")
        for k, v in sorted(tak.items()):
            print(f"  '{k}': {len(v)} foto, contoh {v[:3]}")
    if not a.terapkan:
        print("\nIni dry-run. Tambahkan --terapkan untuk menjalankan sungguhan.")


if __name__ == "__main__":
    main()