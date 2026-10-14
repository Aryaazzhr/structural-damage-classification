# Structural Damage Classification

Proyek akademik machine learning tradisional untuk klasifikasi kerusakan bangunan
Grade 1–5. Metrik kompetisi adalah **F1-macro**: F1 dihitung per kelas, kemudian
dirata-ratakan tanpa pembobotan jumlah sampel. **Tidak menggunakan neural network.**

## Hasil terbaik saat ini

- F1-macro leaderboard: **0.59242**, dilaporkan pengguna (sebelumnya 0.59005).
- F1-macro CV: **0.585262**; fold konfirmasi: **0.585248**.
- Submission: `submissions/submission_improved.csv`.
- Model: `models/improved_ensemble.pkl`, beserta lima fold native dan lima fold improved_wide.
- Catatan identitas file dan riwayat: `reports/leaderboard_best.json` serta `reports/leaderboard_history.json`.

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
scripts/
    __init__.py
    preprocess.py
    feature_engineering.py
    train.py
    predict.py
models/                 # pipeline .pkl; tidak di-commit
submissions/            # submission.csv; tidak di-commit
reports/
    figures/            # gambar hasil analisis
```

Folder kosong memakai `.gitkeep` agar ikut tersimpan di Git. Metadata tambahan,
ringkasan EDA, dan laporan historis berada di reports/. Arsip eksperimen sebelumnya
tersimpan di folder saudara `../structural-damage-classification-archive-20261012`,
bukan bagian dari repositori akademik ini.

## Mengunduh data Kaggle

1. Masuk ke akun Kaggle dan buka halaman kompetisi **Structural Damage Classification**
   yang diberikan pengajar. Cari melalui [Kaggle Competitions](https://www.kaggle.com/competitions)
   jika belum memiliki tautannya. URL/slug kompetisi tidak diasumsikan dari nama folder.
2. Setujui peraturan kompetisi jika diminta, buka tab **Data**, lalu unduh berkas data.
3. Ekstrak `train.csv`, `test.csv`, dan `sample_submission.csv` langsung ke `data/`.
   Jangan meletakkannya dalam `data/raw/` atau folder ZIP bertingkat.
4. Jika tersedia, simpan `metadata.csv` sebagai informasi variabel di `reports/metadata.csv`.
5. Dataset tidak disertakan dalam Git. Pengguna repositori harus mengunduhnya sendiri.

## Instalasi dan menjalankan kode

Gunakan Python 3.11 atau lebih baru. Jalankan perintah berikut **dari direktori proyek**:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.train --restore-v2
python -m scripts.predict --model models/native_ensemble.pkl --output submissions/submission_native.csv
```

Mode native menyimpan lima pipeline `models/native_fold_*.pkl` dan pemuat ensemble
`models/native_ensemble.pkl`. Hasilnya ada di `reports/native_model_metrics.json`
dan `submissions/submission_native.csv`. Perintah predict di atas membuat ulang
submission native dengan urutan ID sesuai contoh.

Mode baseline terpisah (`--model lightgbm`, `--model random_forest`, atau
`--model logistic_regression`) menyimpan `models/model.pkl`,
`models/model.metrics.json`, `reports/model_metrics.json`, dan confusion matrix
holdout. Perintah predict tanpa argumen memakai model baseline tersebut dan
**mengganti** `submissions/submission.csv`; gunakan argumen model native seperti
contoh di atas untuk workflow yang dipulihkan.

Pilihan algoritma: `random_forest` (default), `logistic_regression`, dan `lightgbm`.
Opsi `--use-id-features` mengaktifkan hipotesis pola ID; default nonaktif karena ID
bukan koordinat geografis yang terverifikasi. Tuning dapat membutuhkan waktu lama.
Untuk uji alur singkat tanpa mengganti model utama:

```powershell
python -m scripts.train --model logistic_regression --sample-size 2000 --output models/smoke.pkl
```

Model dari subset hanya untuk pengujian program, bukan submission utama.
Pipeline `.pkl` dapat dibuka dengan `joblib.load`; muat hanya model yang dipercaya.

Kerangka telah diuji pada 1.000 baris: tuning F1-macro, serialisasi pipeline,
prediksi kategori baru/missing, serta penyelarasan ID. Random Forest dan LightGBM
juga diuji fit/predict pada subset kecil. Hasil pemeriksaan tercatat di
`reports/framework_validation.json`; pengujian ini bukan evaluasi performa final.

Untuk notebook:

```powershell
python -m jupyterlab
```

Jalankan berurutan: `01_eda.ipynb`, `02_modeling.ipynb`, `03_submission.ipynb`.
Notebook menggunakan modul yang sama, sehingga implementasi tidak diduplikasi.

## Workflow dan pencegahan kebocoran data

1. Baca label training dan ubah `Grade N` menjadi integer 1–5. Label dalam sample
   submission adalah contoh format, **bukan** label untuk training/evaluasi.
2. Sisihkan 20% data sebagai holdout stratified dengan seed 42.
3. Gunakan satu `Pipeline`: `BuildingFeatures` → `ColumnTransformer` → classifier.
   Fitur meliputi rasio tinggi/luas per lantai dan penanda umur sentinel >=995.
4. Cabang numerik menggunakan median imputation dan scaling; cabang kategori
   menggunakan imputation modus dan `OneHotEncoder` yang menangani kategori baru.
   Parameter preprocessing dipelajari hanya dari training fold.
5. `RandomizedSearchCV` menala parameter pipeline dengan 3-fold stratified CV,
   scoring `f1_macro`, hanya pada 80% bagian training. Holdout tidak dipakai tuning.
6. Laporkan F1-macro, classification report, dan confusion matrix pada holdout.
   Kemudian refit pipeline terpilih pada seluruh baris training dan simpan `.pkl`.
7. Prediksi test melalui pipeline yang sama. Validasi ID unik, urutan ID, tidak ada
   nilai kosong, dan label numerik 1–5 sebelum menulis submission.

Semua function dan class di scripts/ memiliki docstring. Penambahan model harus
tetap memakai pipeline ini; neural network dilarang.

## Interpretasi hasil analisis

Dataset yang tersedia memiliki 533.465 baris training dan 228.629 baris test.
Training tidak memiliki nilai kosong pada kolom asli, tetapi umur sentinel perlu
ditangani sebagai nilai tidak diketahui. Distribusi kelas tidak seimbang: Grade 5
paling dominan. Karena itu accuracy/F1-micro tidak dapat menggantikan F1-macro.
Pembobotan kelas dipakai untuk memberi perhatian pada kelas yang lebih jarang.

Grafik distribusi target, umur, dan hubungan material bangunan dengan target
tersimpan di `reports/figures/`. Pola asosiasi pada grafik tidak membuktikan sebab-akibat.
Interpretasi dan angka rinci tersedia di notebook EDA dan `reports/eda_summary.json`.

### Hasil eksperimen historis

| Eksperimen lama | CV F1-macro | Leaderboard dilaporkan pengguna |
| --- | ---: | ---: |
| V1 | 0.572537 | 0.57842 |
| V2 | 0.582838 | belum diketahui |

V2 lama memperbaiki F1-macro sekitar 1,03 poin persentase dengan pembobotan kelas
dan keputusan prediksi. Lihat `reports/legacy_v2_report.json`. Angka F1-micro lama
0.607106 bukan metrik kompetisi. Submission yang dipertahankan saat penataan berasal
dari eksperimen V2 lama. Baseline one-hot tersedia di `models/model.pkl`; hasil pemulihan native terbaru tersedia di `models/native_ensemble.pkl`.

Baseline akademik one-hot menggunakan preprocessing dan skema holdout yang berbeda.
Jangan mengatribusikan skor historis tersebut ke pipeline baru atau membandingkan
CV dan holdout seolah-olah protokolnya sama. Hasil baseline tersedia pada `reports/model_metrics.json`;
hasil leaderboard hanya diketahui setelah upload.

## Aturan Git/GitHub

`.gitignore` mengecualikan **semua `data/*.csv`**, termasuk train/test/sample, ZIP,
model besar, submission, environment, dan kredensial. Dataset tetap tidak boleh
di-commit meskipun ukurannya kurang dari 100 MB. Jangan gunakan `git add -f` pada data.

Jika belum berupa repositori Git:

```powershell
git init
git check-ignore data/train.csv data/test.csv data/sample_submission.csv
git add README.md .gitignore requirements.txt notebooks scripts reports data/.gitkeep models/.gitkeep submissions/.gitkeep
git status --short
git ls-files "data/*.csv"
```

Perintah terakhir harus kosong. Jika CSV pernah dilacak, hapus dari index dengan
`git rm --cached -- data/train.csv data/test.csv data/sample_submission.csv` sebelum
commit; ini tidak menghapus file lokal. Mengabaikan file tidak menghapusnya dari
riwayat commit lama. Audit riwayat terlebih dahulu jika dataset pernah di-push.
Commit dan push hanya sesudah memeriksa staged files. Repositori ini menyimpan kode, notebook, laporan agregat, dan gambar; dataset, metadata kompetisi, model biner, dan prediksi mentah tetap lokal.

## Hasil baseline akademik one-hot

LightGBM default tanpa tuning dan tanpa fitur ID telah dijalankan pada seluruh
533.465 baris. Holdout 106.693 baris menghasilkan **F1-macro 0.508215**. Setelah
evaluasi, pipeline di-refit pada seluruh training dan disimpan di `models/model.pkl`.

Hasil ini **belum sama** dengan CV V2 historis 0.582838. Pipeline akademik mengganti
kategori native menjadi one-hot encoding (maksimum 128 kategori per kolom),
menonaktifkan fitur ID, dan menggunakan parameter/pembobotan serta keputusan kelas
yang berbeda. Protokol holdout juga berbeda dari CV lama, sehingga selisih angka
0.074624 bukan estimasi perbandingan dengan protokol identik. Tuning baru belum
dijalankan. Submission V2 lama tidak ditimpa. Lihat `reports/score_comparison.json`.

## Pemulihan performa V2 dengan Pipeline akademik

Mode yang direkomendasikan untuk membandingkan dengan V2 lama adalah
`python -m scripts.train --restore-v2`. Mode baseline one-hot tetap tersedia
melalui `--model lightgbm`, tetapi tidak mereproduksi metode V2.

Pipeline native menggunakan `NativeBuildingFeatures`, `ColumnTransformer`
dengan vocabulary kategori per training fold, dan LightGBM native categorical.
Tidak ada pemangkasan kategori menjadi 128 level. Fitur log, rasio, bin umur,
serta pola awalan/akhiran ID dipulihkan. Umur sentinel tetap menjadi missing,
yang ditangani langsung oleh LightGBM. Tidak ada neural network.

Validasi memakai StratifiedKFold(5, shuffle=True, random_state=42) seperti V2.
Pembobotan loss menggunakan akar inverse frekuensi kelas dari training fold.
Setelan model menggunakan konfigurasi V2 dengan 500 pohon tetap per fold;
berbeda dari early stopping historis. Pengali keputusan dioptimalkan hanya
pada fold 0, kemudian dibekukan untuk empat fold konfirmasi. Ini bukan nested CV
atau benchmark baru yang bebas bias karena konfigurasi historis berasal dari
percobaan terdahulu pada dataset yang sama. Skor tidak dijamin identik.

Setiap fold benar-benar dilatih melalui Pipeline; tidak mengimpor model arsip.
Output baru terpisah dari model/submission baseline:

- `models/native_fold_0.pkl` sampai `native_fold_4.pkl`: lima pipeline lengkap.
- `models/native_ensemble.pkl`: pemuat kelima pipeline dan bobot keputusan.
  Semua file fold harus dipertahankan bersama file ensemble.
- `reports/native_model_metrics.json`: skor CV dan empat fold konfirmasi.
- `reports/native_fold_scores.csv`: skor per fold.
- `submissions/submission_native.csv`: prediksi integer 1–5.

Jalankan kembali perintah yang sama untuk melanjutkan checkpoint. Manifest
memeriksa SHA-256 data/kode dan versi library agar hasil lama tidak dipakai
untuk konfigurasi baru. Jangan mengubah kode selama training berjalan.


## Hasil pemulihan native: run lengkap 13 Oktober 2026

Kelima fold telah dilatih ulang pada 533.465 baris menggunakan Pipeline dan
ColumnTransformer. F1-macro dihitung ulang secara independen dengan sklearn.

| Evaluasi | V2 historis | Pipeline native baru |
| --- | ---: | ---: |
| OOF seluruh 5 fold | 0.582838 | **0.583325** |
| Gabungan fold konfirmasi 1–4 | 0.582800 | **0.583310** |

Peningkatan OOF sebesar 0.000487 (sekitar 0.049 poin persentase) bersifat kecil;
ini menunjukkan performa berhasil dipulihkan, bukan bukti peningkatan yang
signifikan secara statistik. Split sama dengan V2, tetapi jumlah pohon tetap dan
vocabulary kategori per fold berbeda. Fold 0 digunakan untuk memilih pengali
keputusan sehingga skor OOF keseluruhan mencakup fold pemilihan tersebut.

Skor per fold (0 sampai 4): 0.583386, 0.582378, 0.583208, 0.585077, 0.582574.
Laporan rinci: `reports/native_model_metrics.json` dan
`reports/native_verification.json`. Prediksi dari ensemble yang dimuat ulang
identik dengan hasil training pada 256 baris test yang tersebar (selisih maksimum
probabilitas 0). Kelima model terverifikasi menggunakan Pipeline/ColumnTransformer.

**File yang direkomendasikan untuk upload: `submissions/submission_native.csv`.**
Seluruh 228.629 baris telah diperiksa: urutan ID sama dengan sample submission,
ID unik, tidak ada missing, serta damage_grade berupa integer 1–5. Label tidak
mengandung teks "Grade". `submissions/submission.csv` historis tetap dipertahankan.
Skor leaderboard submission native kemudian dilaporkan pengguna sebesar **0.59005**. Skor 0.57842 adalah hasil V1 yang lebih awal.


## Eksperimen peningkatan setelah leaderboard 0.59005

Pengguna melaporkan F1-macro leaderboard **0.59005** untuk
`submissions/submission_native.csv`. Identitas file (SHA-256) disimpan di
`reports/leaderboard_history.json`; file dan kelima model native dipertahankan.

Notebook `notebooks/04_improve.ipynb` menguji LightGBM native dengan kapasitas
63/127/255 leaves, 700–1000 pohon, dan regularisasi lambda 5. Setiap kandidat
menggunakan Pipeline serta ColumnTransformer yang sama. Semua kategori dan
bobot kelas dipelajari hanya dari training fold. Pemilihan kandidat, campuran
probabilitas (50% atau 100% kandidat baru), dan pengali keputusan dilakukan pada
fold 0. Selanjutnya konfigurasi dibekukan untuk fold 1–4. Karena dataset pernah
dipakai untuk eksperimen sebelumnya, hasil ini tetap merupakan validasi internal.

Jalankan semua cell notebook, atau dari root proyek di PowerShell:

```powershell
python -u -c "import json; from pathlib import Path; nb=json.loads(Path('notebooks/04_improve.ipynb').read_text()); exec(compile(''.join(nb['cells'][1]['source']), '04_improve.ipynb', 'exec'))"
```

Checkpoint `improved_*` dapat dilanjutkan dengan menjalankan notebook kembali.
Jangan mengubah notebook/data/kode selama run; manifest mendeteksi perubahan
untuk mencegah pemakaian checkpoint yang tidak cocok. Hasil seleksi tersimpan di
`reports/improved_selection.json`; hasil akhir, jika kandidat mengalahkan baseline
pada fold seleksi, di `reports/improved_model_metrics.json`. Model ensemble baru
memerlukan semua file fold yang dirujuk di atribut `model_files`.


Kandidat terpilih pada fold 0: **255 leaves, 700 pohon, min_child_samples=100,
reg_lambda=5**, dengan penggabungan probabilitas 50:50 terhadap model native lama.
Skor seleksi 0.585319 versus baseline 0.583386. Seluruh setelan lain dan fitur tetap
sama. Tabel seluruh kandidat tersedia di `reports/improved_selection.json` dan
grafiknya di `reports/figures/improved_selection.png`.

Setelah training selesai, prediksi dapat dibuat ulang melalui:

```powershell
python -m scripts.predict --model models/improved_ensemble.pkl --output submissions/submission_improved.csv
```

Ensemble ini membutuhkan lima `models/improved_wide_fold_*.pkl` dan lima
`models/native_fold_*.pkl` di direktori models/. Dua model kandidat seleksi lainnya
(compact dan long) tidak dipakai dalam ensemble akhir.


### Hasil eksperimen peningkatan lengkap

| Metrik | Model leaderboard 0.59005 | Ensemble baru |
| --- | ---: | ---: |
| OOF F1-macro (5 fold) | 0.583325 | **0.585262** |
| F1-macro gabungan fold konfirmasi 1–4 | 0.583310 | **0.585248** |

Kelima fold meningkat dibanding baseline. Selisih OOF +0.001937 (sekitar 0.194
poin persentase); selisih konfirmasi +0.001938. Ini hasil validasi internal,
bukan jaminan kenaikan leaderboard. F1 grade 1, 2, 3, dan 5 meningkat, tetapi
F1 grade 4 turun dari 0.559855 menjadi 0.550310. Lihat
`reports/improved_class_comparison.csv` untuk trade-off per kelas serta
`reports/figures/improved_comparison.png` untuk grafik perbandingan.

**Submission terbaik terkonfirmasi pengguna: `submissions/submission_improved.csv` (leaderboard 0.59242).**
Seluruh 228.629 baris lulus validasi ID, urutan sample submission, nilai yang
hilang, dan label integer 1–5. Kesepuluh model terverifikasi sebagai sklearn
Pipeline dengan ColumnTransformer. Uji pemuatan ulang ensemble pada 256 baris
test cocok dengan probabilitas saat training (selisih maksimum 2.23e-16).
Detail pemeriksaan ada di `reports/improved_verification.json`.

File `submissions/submission_native.csv` yang memperoleh 0.59005 tetap utuh
(terverifikasi SHA-256). Skor leaderboard ensemble baru **0.59242**, dilaporkan pengguna; naik **0.00237** dari 0.59005. File terbaik terbaru adalah `submissions/submission_improved.csv`. Riwayat hasil tersimpan di `reports/leaderboard_history.json`.


## Optimasi bobot keputusan lanjutan

Notebook `notebooks/05_calibrate.ipynb` mengoptimalkan empat pengali probabilitas
kelas secara bersama-sama menggunakan differential evolution (tiga seed tetap).
Grade 5 menjadi acuan dengan bobot 1. Optimasi hanya memakai fold 0; bobot terpilih
dibekukan sebelum evaluasi fold 1–4. Classifier tetap ensemble sepuluh sklearn
Pipeline/ColumnTransformer sebelumnya, tanpa neural network atau training baru.

| Evaluasi | Ensemble leaderboard 0.59242 | Bobot baru |
| --- | ---: | ---: |
| OOF F1-macro | 0.585262 | **0.585680** |
| Gabungan fold konfirmasi 1–4 | 0.585248 | **0.585604** |

Peningkatan konfirmasi +0.000356 sangat kecil. Fold historis telah dipakai berulang
kali sehingga ini validasi internal, bukan bukti peningkatan yang signifikan atau
jaminan leaderboard. Pengguna melaporkan skor leaderboard **0.59225** untuk
`submissions/submission_calibrated.csv`, turun **0.00017** dari hasil terbaik.
Kenaikan CV tidak berlanjut pada leaderboard. Submission yang direkomendasikan
tetap **`submissions/submission_improved.csv` (0.59242)** dengan model
`models/improved_ensemble.pkl`. Eksperimen calibrated dipertahankan untuk
reproduksi dan perbandingan, bukan sebagai model terbaik.

Reproduksi dari clone baru:

1. Unduh data ke data/ dan instal requirements.txt sesuai petunjuk di atas.
2. Jalankan `python -m scripts.train --restore-v2` untuk menghasilkan baseline
   lima fold beserta probabilitas lokal.
3. Jalankan seluruh cell `notebooks/04_improve.ipynb` untuk menghasilkan ensemble
   improved dan `reports/improved_predictions.npz`.
4. Jalankan seluruh cell `notebooks/05_calibrate.ipynb` untuk optimasi bobot dan
   menghasilkan `models/calibrated_ensemble.pkl` serta submission baru.

Setelah model tersedia, buat ulang prediksi dengan:

```powershell
python -m scripts.predict --model models/calibrated_ensemble.pkl --output submissions/submission_calibrated.csv
```

Model calibrated tetap memerlukan lima file native_fold dan lima file
improved_wide_fold. Verifikasi seluruh baris submission dan uji pemuatan ulang
model tersedia di `reports/calibrated_verification.json`. Dataset CSV, metadata
kompetisi, model, checkpoint probabilitas NPZ, dan submission tidak disertakan
dalam Git; pengguna perlu menjalankan workflow untuk mereproduksinya.


## Eksperimen fitur regional

`notebooks/06_regional.ipynb` menambahkan fitur dengan `RegionalBuildingFeatures`:
median umur/luas/tinggi per ward, rasio terhadap median, jumlah baris training per
ward, prefix ID 9/10 digit, serta kombinasi district-material, foundation-material,
dan roof-ground-floor. Statistik dipelajari dari training fold saja tanpa label;
ward baru memakai median global training. Prefix ID adalah hipotesis pola data,
bukan koordinat yang telah diverifikasi.

Dua kapasitas LightGBM (127/255 leaves, 700 pohon) dan blend 50%/100% dipilih pada
fold 0. Bobot keputusan kelas tetap berasal dari ensemble leaderboard 0.59242.
Pilihan dibekukan untuk fold konfirmasi 1–4. Jika model regional terpilih,
`WeightedFoldPipelineEnsemble` menjaga bobot campuran yang tepat meskipun jumlah
model komponennya berbeda. Hasil seleksi dan konfirmasi tersimpan terpisah di
`reports/regional_selection.json` dan `reports/regional_model_metrics.json`.

Jalankan notebook 06 setelah workflow 04 menghasilkan probabilitas/model improved.
Penambahan kelas fitur/ensemble ini tidak mengubah perilaku kelas lama, sehingga
model lama tetap dapat dipakai untuk prediksi. Manifest lama secara sengaja menolak
resume training ketika hash modul berubah. Untuk mereproduksi checkpoint eksperimen
lama persis, gunakan commit `9fb4dda` di checkout terpisah; jangan menghapus model
terbaik untuk mengatasi ketidakcocokan manifest. Clone baru tanpa checkpoint dapat
menjalankan urutan training dari awal.


### Hasil eksperimen regional lengkap

Kandidat terpilih memakai **127 leaves, 700 pohon**, dan blend probabilitas
**50% model regional + 50% ensemble terbaik sebelumnya**. Bobot keputusan kelas
tidak dioptimalkan ulang. Ensemble berisi 15 pipeline: lima regional, lima native,
dan lima improved_wide.

| Evaluasi | Ensemble leaderboard 0.59242 | Ensemble regional |
| --- | ---: | ---: |
| OOF F1-macro | 0.585262 | **0.586396** |
| Gabungan fold konfirmasi 1–4 | 0.585248 | **0.586500** |

Kelima fold meningkat. Kenaikan konfirmasi **+0.001251** merupakan hasil validasi
internal pada fold yang sudah pernah digunakan; belum membuktikan kenaikan
leaderboard. Best leaderboard tetap **0.59242** sampai ada hasil upload baru.

Kandidat upload: **`submissions/submission_regional.csv`**. Seluruh 228.629 baris
sudah diperiksa untuk label integer 1–5, ID unik, urutan sample, dan nilai kosong.
Lihat `reports/regional_verification.json` dan `reports/regional_class_comparison.csv`.
Buat ulang prediksi dengan:

```powershell
python -m scripts.predict --model models/regional_ensemble.pkl --output submissions/submission_regional.csv
```

Pertahankan kelima `regional_medium_fold_*.pkl`, kelima `native_fold_*.pkl`, dan
kelima `improved_wide_fold_*.pkl` bersama `regional_ensemble.pkl`. File submission
improved berskor 0.59242 dan identitas SHA-256-nya tetap dipertahankan.
