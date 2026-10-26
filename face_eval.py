"""
face_eval.py
Komparasi Dlib, FaceNet (512d), dan ArcFace lewat DeepFace + MTCNN.

Struktur folder dataset (ubah nama kelompok bila perlu):

dataset/
  gallery/<subjek>/foto.jpg              foto referensi (kondisi normal)
  validasi/<subjek>/foto.jpg             untuk mencari threshold (tidak ikut diuji)
  uji/<kelompok>/<subjek>/foto.jpg       kelompok: baseline, pose, kemiringan,
                                         pencahayaan, jarak, aksesori, serangan_foto
  unknown/<orang>/foto.jpg               orang tidak dikenal (VGGFace2)

Pemakaian:
  pip install deepface tf-keras dlib pandas numpy
  python face_eval.py ekstrak                  # tahap 1, lambat, hasil disimpan di cache
  python face_eval.py evaluasi                 # tahap 2, cepat, bisa diulang
  python face_eval.py evaluasi --threshold deepface
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- konfigurasi
MODELS = {"Dlib": "Dlib", "FaceNet": "Facenet512", "ArcFace": "ArcFace"}
DETECTOR = "mtcnn"
ALIGN = True
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KELOMPOK_SERANGAN = "serangan_foto"

# cadangan bila find_threshold milik DeepFace tidak tersedia di versi kamu.
# CEK ULANG nilai ini di kode sumber DeepFace sebelum dipakai di paper.
TAU_DEEPFACE_CADANGAN = {"Dlib": 0.07, "Facenet512": 0.30, "ArcFace": 0.68}


# ---------------------------------------------------------------- data
def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)


def kumpulkan_data(data_dir):
    """Mengembalikan DataFrame: split, kelompok, subjek, rel, path."""
    rows = []
    for p in list_images(data_dir / "gallery"):
        rel = p.relative_to(data_dir / "gallery").parts
        rows.append(dict(split="gallery", kelompok="gallery", subjek=rel[0], path=p))
    for p in list_images(data_dir / "validasi"):
        rel = p.relative_to(data_dir / "validasi").parts
        rows.append(dict(split="validasi", kelompok="validasi", subjek=rel[0], path=p))
    for p in list_images(data_dir / "uji"):
        rel = p.relative_to(data_dir / "uji").parts
        if len(rel) < 3:
            raise SystemExit(f"Foto uji harus di uji/<kelompok>/<subjek>/foto : {p}")
        split = "serangan" if rel[0] == KELOMPOK_SERANGAN else "genuine"
        rows.append(dict(split=split, kelompok=rel[0], subjek=rel[1], path=p))
    for p in list_images(data_dir / "unknown"):
        rows.append(dict(split="unknown", kelompok="unknown", subjek="unknown", path=p))
    if not rows:
        raise SystemExit(f"Tidak ada foto ditemukan di {data_dir}")
    df = pd.DataFrame(rows)
    df["rel"] = df["path"].apply(lambda p: p.relative_to(data_dir).as_posix())
    return df


# ---------------------------------------------------------------- tahap 1
def tahap_ekstrak(data_dir, out_dir):
    from deepface import DeepFace  # import di sini agar tahap 2 tidak butuh TensorFlow

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "cache_embedding.pkl"
    cache = pickle.load(open(cache_file, "rb")) if cache_file.exists() else {}
    df = kumpulkan_data(data_dir)
    print(f"{len(df)} foto ditemukan.\n", df["split"].value_counts().to_string(), "\n")

    for nama, df_name in MODELS.items():
        print(f"=== Model {nama} ({df_name}) ===")
        # pemanasan: tanpa try/except supaya error instalasi (mis. dlib) langsung terlihat
        DeepFace.represent(img_path=str(df.iloc[0]["path"]), model_name=df_name,
                           detector_backend=DETECTOR, enforce_detection=False, align=ALIGN)
        for i, r in enumerate(df.itertuples(), 1):
            key = (nama, r.rel)
            if key in cache:
                continue
            t0 = time.perf_counter()
            try:
                hasil = DeepFace.represent(img_path=str(r.path), model_name=df_name,
                                           detector_backend=DETECTOR,
                                           enforce_detection=True, align=ALIGN,
                                           anti_spoofing=True)
                ms = (time.perf_counter() - t0) * 1000
                # bila ada beberapa wajah, ambil yang terbesar
                best = max(hasil, key=lambda h: h["facial_area"]["w"] * h["facial_area"]["h"])
                cache[key] = dict(emb=np.asarray(best["embedding"], dtype=np.float32),
                                  ms=ms, err="")
            except ValueError as e:  # DeepFace melempar ValueError bila wajah tidak terdeteksi
                cache[key] = dict(emb=None, ms=(time.perf_counter() - t0) * 1000,
                                  err=str(e)[:80])
            if i % 50 == 0:
                pickle.dump(cache, open(cache_file, "wb"))
                print(f"  {i}/{len(df)}")
        pickle.dump(cache, open(cache_file, "wb"))
    print("Selesai. Cache:", cache_file)


# ---------------------------------------------------------------- tahap 2
def l2(v):
    return v / np.linalg.norm(v)


def bangun_gallery(df, cache, nama):
    gal = {}
    for r in df[df.split == "gallery"].itertuples():
        c = cache.get((nama, r.rel))
        if c is None or c["emb"] is None:
            print(f"  PERINGATAN: foto gallery tanpa wajah terdeteksi: {r.rel}")
            continue
        gal.setdefault(r.subjek, []).append(l2(c["emb"]))
    return {s: np.vstack(v) for s, v in gal.items()}


def jarak_per_subjek(emb, gal):
    e = l2(emb)
    return {s: float(1.0 - (M @ e).max()) for s, M in gal.items()}


def tau_eer(genuine, impostor):
    g, im = np.sort(genuine), np.sort(impostor)
    cand = np.unique(np.concatenate([g, im]))
    far = np.searchsorted(im, cand, side="right") / len(im)
    frr = 1.0 - np.searchsorted(g, cand, side="right") / len(g)
    k = int(np.argmin(np.abs(far - frr)))
    return float(cand[k]), float(far[k]), float(frr[k])


def tau_deepface(df_name):
    try:
        from deepface.modules import verification
        return float(verification.find_threshold(df_name, "cosine"))
    except Exception:
        return TAU_DEEPFACE_CADANGAN[df_name]


def tentukan_tau(df, cache, nama, gal, mode):
    df_name = MODELS[nama]
    if mode == "deepface":
        return tau_deepface(df_name), "bawaan DeepFace"
    gen, imp = [], []
    for r in df[df.split == "validasi"].itertuples():
        c = cache.get((nama, r.rel))
        if c is None or c["emb"] is None or r.subjek not in gal:
            continue
        for s, d in jarak_per_subjek(c["emb"], gal).items():
            (gen if s == r.subjek else imp).append(d)
    if len(gen) < 5 or len(imp) < 5:
        print(f"  PERINGATAN: data validasi {nama} kurang, pakai threshold bawaan DeepFace")
        return tau_deepface(df_name), "bawaan DeepFace (validasi kurang)"
    tau, far, frr = tau_eer(np.array(gen), np.array(imp))
    return tau, f"EER validasi (FAR={far:.3f}, FRR={frr:.3f})"


def prediksi(df, cache, nama, gal, tau):
    rows = []
    for r in df[df.split.isin(["genuine", "unknown", "serangan"])].itertuples():
        c = cache.get((nama, r.rel))
        if c is None:
            raise SystemExit(f"Cache belum lengkap untuk {nama}: {r.rel}. Jalankan tahap ekstrak.")
        if c["emb"] is None:
            rows.append(dict(model=nama, split=r.split, kelompok=r.kelompok, subjek=r.subjek,
                             file=r.rel, terdeteksi=False, d_min=np.nan,
                             pred_terdekat="", pred_final="unknown", ms=c["ms"]))
            continue
        d = jarak_per_subjek(c["emb"], gal)
        terdekat = min(d, key=d.get)
        rows.append(dict(model=nama, split=r.split, kelompok=r.kelompok, subjek=r.subjek,
                         file=r.rel, terdeteksi=True, d_min=d[terdekat],
                         pred_terdekat=terdekat,
                         pred_final=terdekat if d[terdekat] <= tau else "unknown",
                         ms=c["ms"]))
    return pd.DataFrame(rows)


def metrik_klasifikasi(p):
    """Accuracy, error rate, macro precision/recall/F1 (genuine + unknown, tanpa serangan)."""
    p = p[p.split.isin(["genuine", "unknown"])]
    y, h = p.subjek.values, p.pred_final.values
    subjek = sorted(set(p[p.split == "genuine"].subjek))
    pr, rc, f1 = [], [], []
    for c in subjek:
        tp = ((y == c) & (h == c)).sum()
        fp = ((y != c) & (h == c)).sum()
        fn = ((y == c) & (h != c)).sum()
        a = tp / (tp + fp) if tp + fp else 0.0
        b = tp / (tp + fn) if tp + fn else 0.0
        pr.append(a), rc.append(b), f1.append(2 * a * b / (a + b) if a + b else 0.0)
    acc = float((y == h).mean())
    return dict(accuracy=acc, error_rate=1 - acc, macro_precision=np.mean(pr),
                macro_recall=np.mean(rc), macro_f1=np.mean(f1))


def metrik_biometrik(p):
    g = p[p.split == "genuine"]
    u = p[(p.split == "unknown") & p.terdeteksi]
    a = p[(p.split == "serangan") & p.terdeteksi]
    tar = float((g.pred_final == g.subjek).mean()) if len(g) else np.nan
    return dict(TAR=tar, FRR=1 - tar if len(g) else np.nan,
                FAR=float((u.pred_final != "unknown").mean()) if len(u) else np.nan,
                APCER_serangan_foto=float((a.pred_final != "unknown").mean()) if len(a) else np.nan)


def tabel_per_kelompok(p):
    rows = []
    for k, g in p.groupby("kelompok", sort=False):
        r = dict(model=g.model.iloc[0], kelompok=k, n_foto=len(g),
                 detection_rate=g.terdeteksi.mean(), rata_ms=g.ms.mean())
        sp = g.split.iloc[0]
        if sp == "genuine":
            benar = (g.pred_final == g.subjek).sum()
            tertukar = ((g.pred_final != g.subjek) & (g.pred_final != "unknown")).sum()
            r.update(TAR=benar / len(g), FRR=1 - benar / len(g),
                     salah_identitas=tertukar / len(g))
        else:  # unknown atau serangan: yang dihitung adalah yang salah diterima
            d = g[g.terdeteksi]
            r["FAR_atau_APCER"] = (d.pred_final != "unknown").mean() if len(d) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def tahap_evaluasi(data_dir, out_dir, mode):
    cache = pickle.load(open(out_dir / "cache_embedding.pkl", "rb"))
    df = kumpulkan_data(data_dir)
    ringkasan, kelompok, semua = [], [], []
    for nama in MODELS:
        print(f"=== {nama} ===")
        gal = bangun_gallery(df, cache, nama)
        tau, cara = tentukan_tau(df, cache, nama, gal, mode)
        print(f"  tau = {tau:.4f}  ({cara})")
        p = prediksi(df, cache, nama, gal, tau)
        semua.append(p)
        kelompok.append(tabel_per_kelompok(p))
        uji = p
        ringkasan.append(dict(model=nama, tau=tau, cara_tau=cara, n_foto=len(uji),
                              detection_rate=uji.terdeteksi.mean(),
                              **metrik_klasifikasi(p), **metrik_biometrik(p),
                              rata_ms=uji.ms.mean(), median_ms=uji.ms.median()))
        cm = pd.crosstab(p[p.split.isin(["genuine", "unknown"])].subjek,
                         p[p.split.isin(["genuine", "unknown"])].pred_final)
        cm.to_csv(out_dir / f"confusion_matrix_{nama}.csv")

    pd.concat(semua).to_csv(out_dir / "hasil_prediksi.csv", index=False)
    pd.concat(kelompok).to_csv(out_dir / "metrik_per_kelompok.csv", index=False)
    r = pd.DataFrame(ringkasan)
    r.to_csv(out_dir / "metrik_ringkasan.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", r.round(4).T.to_string())
    print("\nFile hasil ada di:", out_dir)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tahap", choices=["ekstrak", "evaluasi"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--threshold", choices=["eer", "deepface"], default="eer")
    a = ap.parse_args()
    d, o = Path(a.data), Path(a.out)
    if a.tahap == "ekstrak":
        tahap_ekstrak(d, o)
    else:
        tahap_evaluasi(d, o, a.threshold)
