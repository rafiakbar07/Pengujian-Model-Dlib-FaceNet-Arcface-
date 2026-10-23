"""
visualisasi_hasil.py
Membuat gambar dan laporan dari folder hasil/ (output face_eval.py evaluasi).

Keluaran di hasil/gambar/:
  confusion_matrix.png     confusion matrix 3 model
  keberhasilan_kelompok.png  % berhasil per kondisi x model
  peta_keberhasilan.png    tiap foto: berhasil (hijau) / gagal (merah) per model
  gagal_<model>.csv        daftar foto yang gagal beserta alasannya

Pemakaian:
  pip install matplotlib
  python visualisasi_hasil.py
  python visualisasi_hasil.py --hasil hasil_deepface
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def tambah_status(p, tau_map):
    ok, status = [], []
    for r in p.itertuples():
        if r.split == "genuine":
            if r.pred_final == r.subjek:
                o, s = True, "benar"
            elif not r.terdeteksi:
                o, s = False, "tak terdeteksi"
            elif r.pred_final == "unknown":
                o, s = False, "ditolak"
            else:
                o, s = False, "tertukar"
        else:  # serangan atau unknown: berhasil bila ditolak
            o = r.pred_final == "unknown"
            s = "ditolak" if o else "DITERIMA"
        ok.append(o)
        status.append(s)
    p = p.copy()
    p["ok"] = ok
    p["status"] = status
    p["tau"] = p.model.map(tau_map)
    p["selisih_ke_tau"] = p.d_min - p.tau  # positif = jarak melewati batas
    return p


def matriks(pm):
    g = pm[pm.split.isin(["genuine", "unknown"])]
    subjek = set(pm[pm.split == "genuine"].subjek) | (set(pm.pred_final) - {"unknown"})
    kolom = sorted(subjek) + ["unknown"]
    baris = [k for k in kolom if k in set(g.subjek)]
    return pd.crosstab(g.subjek, g.pred_final).reindex(index=baris, columns=kolom, fill_value=0)


def gambar_cm(ax, cm, judul):
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    mx = cm.values.max() or 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > mx / 2 else "black")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Sebenarnya")
    ax.set_title(judul)


def simpan_cm(p, models, out):
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        gambar_cm(ax, matriks(p[p.model == m]), m)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def simpan_kelompok(p, models, out):
    urut = list(dict.fromkeys(p.kelompok))
    pers = np.full((len(urut), len(models)), np.nan)
    teks = [[""] * len(models) for _ in urut]
    for i, k in enumerate(urut):
        for j, m in enumerate(models):
            g = p[(p.kelompok == k) & (p.model == m)]
            if len(g):
                pers[i, j] = g.ok.mean() * 100
                teks[i][j] = f"{int(g.ok.sum())}/{len(g)}"
    fig, ax = plt.subplots(figsize=(2 * len(models) + 3, 0.65 * len(urut) + 2))
    im = ax.imshow(pers, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(len(urut)))
    ax.set_yticklabels(urut)
    for i in range(len(urut)):
        for j in range(len(models)):
            ax.text(j, i, teks[i][j], ha="center", va="center", fontsize=11)
    ax.set_title("Berhasil per kondisi\n(genuine: dikenali benar; serangan/unknown: ditolak)")
    fig.colorbar(im, ax=ax, label="% berhasil")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def simpan_peta(p, models, out):
    base = p[p.model == models[0]][["kelompok", "subjek", "file"]].reset_index(drop=True)
    ok = np.zeros((len(base), len(models)), dtype=int)
    st = [[""] * len(models) for _ in range(len(base))]
    for j, m in enumerate(models):
        pm = p[p.model == m].set_index("file")
        for i, f in enumerate(base.file):
            ok[i, j] = int(pm.loc[f, "ok"])
            st[i][j] = pm.loc[f, "status"]
    fig, ax = plt.subplots(figsize=(2.2 * len(models) + 5, 0.3 * len(base) + 2))
    ax.imshow(ok, cmap=ListedColormap(["#d9534f", "#5cb85c"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(base)))
    ax.set_yticklabels([f"{k} / {Path(f).name}" for k, f in zip(base.kelompok, base.file)], fontsize=8)
    for i in range(len(base)):
        for j in range(len(models)):
            ax.text(j, i, st[i][j], ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Hijau = berhasil, merah = gagal", pad=28)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasil", default="hasil")
    d = Path(ap.parse_args().hasil)
    ring = pd.read_csv(d / "metrik_ringkasan.csv")
    models = list(ring.model)
    tau_map = dict(zip(ring.model, ring.tau))
    p = tambah_status(pd.read_csv(d / "hasil_prediksi.csv"), tau_map)
    p["terdeteksi"] = p.terdeteksi.astype(bool)

    gambar = d / "gambar"
    gambar.mkdir(exist_ok=True)
    simpan_cm(p, models, gambar / "confusion_matrix.png")
    simpan_kelompok(p, models, gambar / "keberhasilan_kelompok.png")
    simpan_peta(p, models, gambar / "peta_keberhasilan.png")

    pd.set_option("display.width", 220)
    print("\n=== Jumlah per status ===")
    print(p.groupby(["model", "status"]).size().unstack(fill_value=0).to_string())

    print("\n=== Berhasil/total per kondisi ===")
    tab = p.groupby(["kelompok", "model"], sort=False).ok.agg(["sum", "count"])
    tab["teks"] = tab["sum"].astype(int).astype(str) + "/" + tab["count"].astype(str)
    print(tab["teks"].unstack("model")[models].to_string())

    print("\n=== Foto yang gagal ===")
    for m in models:
        g = p[(p.model == m) & (~p.ok)][["kelompok", "subjek", "file", "pred_final",
                                         "status", "d_min", "tau", "selisih_ke_tau"]]
        g.round(4).to_csv(gambar / f"gagal_{m}.csv", index=False)
        print(f"\n[{m}] {len(g)} foto gagal")
        if len(g):
            g2 = g.copy()
            g2["file"] = g2.file.apply(lambda f: Path(f).name)
            print(g2.round(3).to_string(index=False))

    print("\nGambar tersimpan di:", gambar)


if __name__ == "__main__":
    main()