# Komparasi Dlib, FaceNet, dan ArcFace untuk Pengawasan Karyawan

Paket ini mengikuti alur Bab 2 pada paper. Jalankan berurutan.

## 0. Persiapan

```
pip install deepface tf-keras dlib opencv-python pandas numpy matplotlib
```

Di Windows, `dlib` sering gagal terpasang tanpa CMake dan Visual C++ Build Tools. Pasang keduanya lebih dulu bila itu terjadi. Versi Python dan TensorFlow yang cocok dengan DeepFace sebaiknya dicek di halaman repositori DeepFace.

## 1. Rapikan dataset (siapkan_dataset.py)

Taruh foto mentah di `sumber/<subjek>/...`. Kondisi foto dikenali dari kata kunci pada nama subfolder atau nama file, misalnya `pose_kiri_01.jpg`, `redup/IMG_22.jpg`, `jarak_2m.png`, `kacamata.jpg`, `layar_hp.jpg`. Daftar kata kunci ada di variabel `ATURAN` dan boleh disesuaikan.

```
python siapkan_dataset.py --sumber sumber --vggface2 vggface2
python siapkan_dataset.py --sumber sumber --vggface2 vggface2 --terapkan
```

Perintah pertama hanya simulasi dan menampilkan jumlah foto per bagian. Perintah kedua menyalin file ke `dataset/` dan menulis `dataset/manifest.csv` sebagai jejak asal tiap foto. Foto baseline tiap subjek dibagi menjadi 1 gallery, 5 validasi, dan sisanya uji. Ambil foto ketiga bagian itu pada sesi yang berbeda agar tidak saling mirip.

Struktur akhir yang dibaca skrip berikutnya:

```
dataset/gallery/<subjek>/
dataset/validasi/<subjek>/
dataset/uji/<kelompok>/<subjek>/
dataset/unknown/<orang>/
```

## 2. Ekstraksi embedding (face_eval.py ekstrak)

```
python face_eval.py ekstrak
```

Tahap paling lama. Hasil disimpan di `hasil/cache_embedding.pkl` dan bisa dilanjutkan bila terhenti.

## 3. Evaluasi (face_eval.py evaluasi)

```
python face_eval.py evaluasi
python face_eval.py evaluasi --threshold deepface
```

Threshold dicari per model pada titik EER dari data validasi. Opsi kedua memakai nilai bawaan DeepFace sebagai pembanding. Keluaran: `metrik_ringkasan.csv`, `metrik_per_kelompok.csv`, `confusion_matrix_<model>.csv`, `hasil_prediksi.csv`.

## 4. Uji real-time (realtime_webcam.py)

Satu sesi sama dengan satu model, satu kondisi, satu orang, dan N percobaan.

```
python realtime_webcam.py --model ArcFace --subjek rafi --kondisi jarak_1m --percobaan 20
python realtime_webcam.py --model ArcFace --subjek unknown --kondisi orang_asing
python realtime_webcam.py --model ArcFace --subjek serangan --kondisi foto_layar_hp
```

Tekan SPASI untuk mulai merekam dan q untuk keluar. Ulangi kondisi yang sama untuk ketiga model dengan jumlah percobaan dan posisi kamera yang sama. Hasil ditambahkan ke `hasil/realtime_log.csv`. Skrip ini butuh hasil tahap 3 untuk membaca threshold.


## Catatan untuk penulisan

- Bab 3 sebaiknya memakai dua subbab saja sesuai template, misalnya 3.1 Hasil Pengujian Foto dan 3.2 Hasil Pengujian Real-time. Pembahasan menyusul setelah angkanya ada.
- Tulis di paper bahwa jumlah subjek, jumlah foto per kondisi, dan nilai tau tiap model berasal dari `metrik_ringkasan.csv` dan `manifest.csv`.
- Sebutkan keterbatasan: foto VGGFace2 berasal dari domain yang berbeda dari webcam, dan subjek terdaftar berasal dari dataset pribadi dalam jumlah terbatas.
- Peringkat di `peringkat.txt` hanya menerapkan urutan prioritas secara harfiah. Bila selisih dua model sangat tipis, nilai itu perlu dipertimbangkan secara manual.
