# Structural Damage Classification

Dalam proyek ini, saya mengeksplorasi pendekatan machine learning tradisional untuk mengklasifikasikan tingkat kerusakan bangunan ke dalam **Grade 1–5** pada kompetisi Kaggle Structural Damage Classification. Fokus saya adalah membangun alur eksperimen yang dapat direproduksi dan mengevaluasi model menggunakan **F1-macro**, sehingga setiap kelas memiliki kontribusi yang sama terhadap skor.

Model akhir menggunakan ensemble LightGBM dengan preprocessing melalui **Scikit-Learn Pipeline dan ColumnTransformer**. Proyek ini tidak menggunakan neural network.

## Hasil utama

| Metrik | Hasil |
| --- | ---: |
| F1-macro leaderboard terbaik | **0.59578** |
| F1-macro out-of-fold, 5 fold | **0.588670** |
| F1-macro gabungan fold konfirmasi 1–4 | **0.588776** |
| Jumlah data training | 533.465 |
| Jumlah data test | 228.629 |

Submission terbaik adalah `submissions/submission_diverse.csv`, yang dihasilkan oleh `models/diverse_ensemble.pkl`. Riwayat skor dan identitas SHA-256 submission tersedia di [leaderboard_history.json](reports/leaderboard_history.json) dan [leaderboard_best.json](reports/leaderboard_best.json).

Skor leaderboard dicatat dari hasil submission Kaggle, sedangkan skor CV dihitung dari prediksi validasi lokal. Keduanya menggunakan data evaluasi berbeda dan tidak diharapkan bernilai sama.

## Struktur repositori

```text
README.md
.gitignore
requirements.txt
data/
    train.csv
    test.csv
    sample_submission.csv
notebooks/
    01_eda.ipynb
    02_modeling.ipynb
    03_submission.ipynb
    04_improve.ipynb
    05_calibrate.ipynb
    06_regional.ipynb
    07_blend.ipynb
    08_diverse.ipynb
scripts/
    __init__.py
    preprocess.py
    feature_engineering.py
    train.py
    predict.py
models/
submissions/
reports/
    figures/
```

Folder `models/` menyimpan model `.pkl`, sedangkan `submissions/` menyimpan file prediksi. Dataset, metadata kompetisi, model biner, prediksi mentah, dan submission tetap disimpan secara lokal serta dikecualikan dari Git. File `.gitkeep` mempertahankan folder kosong pada repositori.

## Data

Saya menggunakan `train.csv` untuk pengembangan dan validasi model, `test.csv` untuk prediksi submission, serta `sample_submission.csv` sebagai acuan kolom dan urutan ID. Informasi tambahan variabel tersedia pada `metadata.csv`.

Untuk menyiapkan data:

1. Masuk ke Kaggle dan buka kompetisi **Structural Damage Classification** sesuai tautan kelas. Halaman kompetisi dapat dicari melalui [Kaggle Competitions](https://www.kaggle.com/competitions).
2. Setujui aturan kompetisi, kemudian unduh dataset melalui tab **Data**.
3. Ekstrak `train.csv`, `test.csv`, dan `sample_submission.csv` langsung ke folder `data/`.
4. Jika tersedia, simpan `metadata.csv` di `reports/metadata.csv`.

Dataset tidak disertakan di GitHub. Label target training berbentuk `Grade N` dipetakan ke integer **1–5**. Submission menggunakan angka 1–5 karena evaluator menolak label teks seperti `Grade 1`. Label pada sample submission hanya merupakan contoh format, bukan target untuk training.

## Instalasi

Gunakan Python 3.11 atau lebih baru. Jalankan perintah berikut dari root repositori:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Untuk membuka notebook:

```powershell
python -m jupyterlab
```

## Metode

### Preprocessing dan feature engineering

Baseline umum menggunakan `BuildingFeatures` → `ColumnTransformer` → classifier dalam satu Pipeline. Kolom numerik memakai median imputation dan scaling, sedangkan kolom kategorikal memakai imputasi modus dan one-hot encoding. Random Forest, Logistic Regression, dan LightGBM tersedia sebagai pilihan classifier. Tuning baseline menggunakan `RandomizedSearchCV` dengan scoring `f1_macro` pada training folds.

Pada eksperimen LightGBM native, saya mempertahankan kategori sebagai tipe kategorikal pandas melalui `NativeCategories`. Vocabulary kategori hanya dipelajari dari training fold; kategori baru menjadi missing. Umur sentinel ≥995 diperlakukan sebagai nilai tidak diketahui dan ditangani oleh LightGBM.

Fitur yang digunakan meliputi:

- Rasio tinggi dan luas per lantai, volume proxy, rasio luas terhadap tinggi, serta transformasi log.
- Kelompok umur, jumlah lantai, material, dan kategori wilayah.
- Prefix dan suffix ID sebagai hipotesis pola data. Fitur ID tidak dianggap sebagai koordinat geografis yang telah diverifikasi.
- Median umur, luas, dan tinggi per ward; rasio terhadap median; serta jumlah baris training per ward.
- Kombinasi district-material, foundation-material, dan roof-ground-floor.

Statistik regional dihitung dari fitur training fold tanpa menggunakan label. Ward yang belum dikenal memakai median global training. Seluruh function dan class pada `scripts/` memiliki docstring.

### Validasi dan pemilihan model

Baseline one-hot memakai holdout stratified 20%. Eksperimen native dan ensemble memakai **StratifiedKFold(5, shuffle=True, random_state=42)** agar perbandingan antarversi menggunakan split yang sama.

Fold 0 digunakan untuk memilih kandidat atau proporsi ensemble. Pilihan kemudian dibekukan sebelum evaluasi fold konfirmasi 1–4. Bobot loss menggunakan akar inverse frekuensi kelas yang dihitung dari training fold. Bobot keputusan kelas dioptimalkan pada tahap awal; eksperimen regional dan sampled mempertahankan bobot dari ensemble improved.

Fold yang sama telah digunakan berulang dalam pengembangan, sehingga hasil ini merupakan **validasi internal**, bukan benchmark baru yang sepenuhnya bebas bias. Skor OOF seluruh fold juga mencakup fold pemilihan kandidat. Kenaikan CV tidak menjamin kenaikan leaderboard.

### Ensemble akhir

Model akhir berisi 20 Pipeline: lima model untuk setiap keluarga berikut.

| Keluarga model | Kontribusi probabilitas |
| --- | ---: |
| Sampled LightGBM | 50% |
| Regional LightGBM | 25% |
| Native LightGBM | 12.5% |
| Wide LightGBM | 12.5% |

Komponen sampled menggunakan seed 137, 900 pohon, learning rate 0.03, 127 leaves, sampling baris 0.85 setiap iterasi, sampling fitur 0.75, `min_child_samples=60`, dan `reg_lambda=5`. Probabilitas tiap keluarga dirata-ratakan antar-fold, digabungkan sesuai proporsi di atas, lalu dikalikan dengan bobot keputusan kelas yang dibekukan.

## Ringkasan eksperimen

| Eksperimen | F1-macro lokal | F1-macro leaderboard |
| --- | ---: | ---: |
| V1 historis, CV | 0.572537 | 0.57842 |
| V2 historis, CV | 0.582838 | — |
| Baseline one-hot, holdout | 0.508215 | — |
| Native Pipeline, CV | 0.583325 | 0.59005 |
| Improved ensemble, CV | 0.585262 | 0.59242 |
| Kalibrasi bobot, CV | 0.585680 | 0.59225 |
| Regional ensemble, CV | 0.586396 | 0.59326 |
| Sampled ensemble, CV | **0.588670** | **0.59578** |

Angka holdout baseline tidak dibandingkan langsung dengan CV sebagai pengukuran pada protokol identik. Baseline one-hot juga membatasi kategori per kolom menjadi 128 dan tidak memakai fitur ID secara default, sehingga metode tersebut berbeda dari model native.

Dua hasil negatif menjadi bagian penting dalam evaluasi saya:

- Kalibrasi bobot meningkatkan CV, tetapi leaderboard turun dari 0.59242 ke 0.59225. Karena itu, bobot tersebut tidak digunakan dalam ensemble akhir.
- Pencarian ulang proporsi tiga keluarga model pada notebook 07 menurunkan skor konfirmasi dari 0.586500 ke 0.586320. Kandidat ini tidak direkomendasikan untuk submission.

Penambahan komponen sampled meningkatkan skor CV dari 0.586396 ke 0.588670, skor konfirmasi dari 0.586500 ke 0.588776, dan leaderboard dari 0.59326 ke 0.59578. F1 kelima kelas serta kelima fold meningkat pada validasi lokal.

Laporan lengkap tersedia di `reports/*_model_metrics.json`, `reports/*_fold_scores.csv`, dan [diverse_class_comparison.csv](reports/diverse_class_comparison.csv).

## Interpretasi analisis

Distribusi kelas tidak seimbang; Grade 5 merupakan kelas terbesar. Saya menggunakan F1-macro agar peningkatan pada kelas mayoritas tidak menutupi kinerja kelas lain. Grade 2 dan 3 masih relatif sulit dibedakan, sehingga menjadi perhatian utama untuk pengembangan selanjutnya.

Fitur wilayah dan pola ID memberikan informasi prediktif yang kuat pada split saat ini. Namun, pola tersebut tidak membuktikan hubungan sebab-akibat dan belum menjamin model dapat digeneralisasikan ke wilayah baru. Median regional memberikan konteks ukuran bangunan terhadap lingkungannya, sedangkan kombinasi material membantu model menangkap hubungan antarjenis konstruksi.

Hasil ensemble menunjukkan bahwa penggabungan model dapat membantu meskipun suatu komponen lebih lemah ketika digunakan sendiri. Peningkatan sampled ensemble konsisten dengan manfaat kombinasi model, tetapi bukan bukti kausal bahwa sampling saja menyebabkan seluruh kenaikan: beberapa parameter berubah bersama-sama.

Grafik analisis tersedia di [reports/figures](reports/figures), termasuk distribusi target, distribusi umur, hubungan material-target, dan perbandingan eksperimen. Confusion matrix holdout pada folder tersebut berasal dari baseline one-hot, bukan model akhir.

## Reproduksi model terbaik

Setelah data dan dependensi siap, jalankan urutan berikut:

1. Buka `notebooks/01_eda.ipynb` untuk eksplorasi data.
2. Jalankan baseline native:

   ```powershell
   python -m scripts.train --restore-v2
   ```

3. Jalankan seluruh cell `notebooks/04_improve.ipynb` untuk ensemble improved.
4. Jalankan seluruh cell `notebooks/06_regional.ipynb` untuk ensemble regional.
5. Jalankan seluruh cell `notebooks/08_diverse.ipynb` untuk ensemble sampled dan submission terbaik.

Notebook 02 dan 03 menjelaskan modeling serta pembuatan submission native. Notebook 05 dan 07 menyimpan eksperimen pembanding dan tidak diperlukan untuk menghasilkan ensemble akhir. Training dan prediksi penuh dapat membutuhkan waktu lama.

Setelah seluruh model tersedia, submission dapat dibuat ulang melalui:

```powershell
python -m scripts.predict --model models/diverse_ensemble.pkl --output submissions/submission_diverse.csv
```

Pertahankan `diverse_ensemble.pkl` bersama kelima file fold untuk masing-masing prefix `diverse_sampled_fold_`, `regional_medium_fold_`, `native_fold_`, dan `improved_wide_fold_`. File ensemble merujuk model komponen di folder `models/`; file ensemble saja tidak cukup untuk melakukan prediksi.

Checkpoint menggunakan manifest SHA-256 data/kode untuk mencegah pemakaian hasil lama dengan konfigurasi berbeda. Jangan mengubah notebook atau modul saat training berjalan. Manifest lama dapat menolak resume setelah kode berubah. Untuk mereproduksi checkpoint sebelum penambahan fitur regional, gunakan commit `9fb4dda` di checkout terpisah. Clone baru tanpa checkpoint dapat menjalankan urutan di atas dari awal.

Untuk mencoba baseline lain pada subset kecil:

```powershell
python -m scripts.train --model logistic_regression --sample-size 2000 --output models/smoke.pkl
```

Model subset hanya untuk pemeriksaan alur, bukan submission utama. Perintah predict tanpa argumen memakai `models/model.pkl` dan menulis ulang `submissions/submission.csv`; gunakan path model/output eksplisit untuk memilih eksperimen.

## Verifikasi hasil

Pemeriksaan pada model akhir mencakup:

- 228.629 baris submission dengan ID unik dan urutan yang sama dengan sample submission.
- Label integer 1–5 tanpa nilai kosong.
- Perhitungan ulang F1-macro menggunakan Scikit-Learn.
- Seluruh 20 model menggunakan Pipeline dan ColumnTransformer.
- Kesesuaian probabilitas model yang dimuat ulang pada 128 baris test.
- Pemeriksaan docstring modul dan sintaks kode notebook.

Hasilnya tersimpan di [diverse_verification.json](reports/diverse_verification.json). Prediksi model yang dimuat ulang berbeda paling banyak sekitar 3.33e-16 dari probabilitas tersimpan, sesuai toleransi floating-point. Submission eksperimen sebelumnya tetap dipertahankan untuk perbandingan.

## Pengelolaan Git dan data

`.gitignore` mengecualikan seluruh `data/*.csv`, metadata kompetisi, model biner, checkpoint probabilitas, submission, environment, dan kredensial. Dataset tidak boleh di-commit, termasuk file yang berukuran kurang dari 100 MB.

Sebelum commit, periksa:

```powershell
git check-ignore data/train.csv data/test.csv data/sample_submission.csv
git status --short
git ls-files "data/*.csv"
```

Perintah terakhir harus tidak menghasilkan output. Jika dataset pernah masuk index, gunakan `git rm --cached -- <path>` untuk menghapusnya dari index tanpa menghapus file lokal. Jangan memakai `git add -f` pada dataset. Mengabaikan file tidak menghapusnya dari riwayat commit yang sudah ada.

Repositori ini menyimpan kode, notebook, dokumentasi, laporan agregat, dan gambar analisis. Dataset harus diunduh melalui Kaggle, sedangkan model dan submission dihasilkan kembali melalui workflow yang tersedia.
