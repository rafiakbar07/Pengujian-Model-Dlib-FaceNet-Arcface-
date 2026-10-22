"""
siapkan_dataset.py
Langkah 0: menyusun foto mentah menjadi struktur yang dibaca face_eval.py.

Foto mentah harus berada di:  sumber/<subjek>/.../foto.jpg
Kondisi foto dikenali dari kata kunci pada nama subfolder atau nama file,
misalnya "pose_kiri_01.jpg", "redup/IMG_22.jpg", "jarak_2m.png", "kacamata.jpg".
Foto yang kondisinya tidak dikenali tidak disalin, hanya dilaporkan.

Pembagian foto baseline tiap subjek (urut nama file):
  N_GALLERY pertama  -> gallery  (referensi identitas)
  N_VALIDASI berikut -> validasi (mencari threshold)
  sisanya            -> uji/baseline
Usahakan foto gallery, validasi, dan uji baseline diambil di sesi yang berbeda,
supaya tidak mirip seperti hasil burst.

Orang tidak dikenal (VGGFace2): --vggface2 <folder berisi <id>/foto.jpg>

Pemakaian:
  python siapkan_dataset.py --sumber sumber --vggface2 vggface2        (dry-run)
  python siapkan_dataset.py --sumber sumber --vggface2 vggface2 --terapkan
"""
import argparse
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
N_GALLERY = 1
N_VALIDASI = 5

# urutan penting: aturan yang lebih spesifik ditaruh lebih dulu
ATURAN = [
    ("serangan_foto", ["cetak", "print", "layar", "hp", "ponsel", "spoof", "serangan", "attack"]),
    ("aksesori", ["kacamata", "topi", "glasses", "hat", "aksesori", "aksesoris"]),
    ("jarak", ["30cm", "1m", "2m", "jarak", "distance"]),
    ("pencahayaan", ["redup", "gelap", "backlight", "samping", "belakang", "cahaya", "light", "dim", "dark"]),
    ("kemiringan", ["miring", "tilt", "roll"]),
    ("pose", ["kiri", "kanan", "atas", "bawah", "menoleh", "pose", "left", "right", "up", "down"]),
    ("baseline", ["baseline", "frontal", "normal", "depan"]),
]


def tebak_kelompok(teks):
    teks = teks.lower()
    for kelompok, kata in ATURAN:
        for k in kata:
            if re.search(r"(?<![a-z0-9])" + re.escape(k), teks):
                return kelompok
    return None


def salin(src, dst_dir, terapkan):
    dst = dst_dir / src.name
    n = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{n}{src.suffix}"
        n += 1
    if terapkan:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", required=True)
    ap.add_argument("--tujuan", default="dataset")
    ap.add_argument("--vggface2")
    ap.add_argument("--n-orang", type=int, default=20)
    ap.add_argument("--n-foto", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--terapkan", action="store_true", help="salin file sungguhan")
    a = ap.parse_args()

    sumber, tujuan = Path(a.sumber), Path(a.tujuan)
    manifest, tak_dikenal = [], []

    for folder_subjek in sorted(p for p in sumber.iterdir() if p.is_dir()):
        subjek = folder_subjek.name
        per_kelompok = {}
        for f in sorted(folder_subjek.rglob("*")):
            if f.suffix.lower() not in EXT:
                continue
            rel = f.relative_to(folder_subjek)
            kelompok = tebak_kelompok("/".join(rel.parts))
            if kelompok is None:
                tak_dikenal.append(str(f))
                continue
            per_kelompok.setdefault(kelompok, []).append(f)

        base = per_kelompok.pop("baseline", [])
        if len(base) < N_GALLERY + N_VALIDASI + 1:
            print(f"PERINGATAN: baseline {subjek} hanya {len(base)} foto, "
                  f"butuh minimal {N_GALLERY + N_VALIDASI + 1}")
        pembagian = [("gallery", base[:N_GALLERY], tujuan / "gallery" / subjek),
                     ("validasi", base[N_GALLERY:N_GALLERY + N_VALIDASI], tujuan / "validasi" / subjek),
                     ("uji/baseline", base[N_GALLERY + N_VALIDASI:], tujuan / "uji" / "baseline" / subjek)]
        for kel, files in per_kelompok.items():
            pembagian.append((f"uji/{kel}", files, tujuan / "uji" / kel / subjek))

        for nama, files, dst_dir in pembagian:
            for f in files:
                dst = salin(f, dst_dir, a.terapkan)
                manifest.append(dict(subjek=subjek, bagian=nama, asal=str(f), tujuan=str(dst)))

    if a.vggface2:
        rng = random.Random(a.seed)
        ids = sorted(p for p in Path(a.vggface2).iterdir() if p.is_dir())
        terpilih = rng.sample(ids, min(a.n_orang, len(ids)))
        for folder in terpilih:
            fotos = sorted(f for f in folder.rglob("*") if f.suffix.lower() in EXT)
            for f in rng.sample(fotos, min(a.n_foto, len(fotos))):
                dst = salin(f, tujuan / "unknown" / folder.name, a.terapkan)
                manifest.append(dict(subjek="unknown", bagian="unknown", asal=str(f), tujuan=str(dst)))

    if not manifest:
        raise SystemExit("Tidak ada foto yang cocok. Periksa folder sumber dan kata kunci pada ATURAN.")

    m = pd.DataFrame(manifest)
    print("\nRingkasan jumlah foto:")
    print(pd.crosstab(m.subjek, m.bagian).to_string())
    if tak_dikenal:
        print(f"\n{len(tak_dikenal)} foto tidak dikenali kondisinya (tidak disalin):")
        for t in tak_dikenal[:30]:
            print("  ", t)
        if len(tak_dikenal) > 30:
            print(f"   ... dan {len(tak_dikenal) - 30} lainnya")
    if a.terapkan:
        tujuan.mkdir(parents=True, exist_ok=True)
        m.to_csv(tujuan / "manifest.csv", index=False)
        print(f"\nSelesai disalin ke {tujuan}. Jejak asal-tujuan: {tujuan / 'manifest.csv'}")
    else:
        print("\nIni dry-run. Tambahkan --terapkan untuk menyalin file.")


if __name__ == "__main__":
    main()
