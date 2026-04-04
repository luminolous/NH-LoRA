# IMPLEMENT.md

## Cara Kerja Codex pada Repo Ini

Kamu sedang mengimplementasikan repo riset untuk NH-LoRA.
Tujuanmu adalah menyelesaikan implementasi secara penuh sesuai design paper, tetapi dengan workflow yang aman, modular, dan dapat diaudit.

## Aturan Paling Penting

1. Ikuti design paper NH-LoRA sebagai sumber kebenaran utama.
2. Jangan mengubah metode menjadi metode lain.
3. Jangan menghapus modul inti hanya demi menyederhanakan implementasi.
4. Jika ada bagian yang belum 100% eksplisit, pilih implementasi paling sederhana yang tetap setia pada maksud paper.
5. Milestone digunakan untuk kontrol progres, tetapi target akhir tetap implementasi penuh.

## Aturan Teknis Repo

- Gunakan YAML untuk konfigurasi di `configs/`
- Gunakan bash runner di `scripts/`
- Simpan log ke `outputs/logs/`
- Simpan summary metrics ke `outputs/summaries/`
- Simpan checkpoint ringan ke `outputs/checkpoints/`
- Simpan metrics mentah ke `outputs/metrics/`

## Bahasa

- Komunikasi/dokumen kerja boleh berbahasa Indonesia
- Semua source code harus berbahasa Inggris
- Semua code comments harus berbahasa Inggris
- Semua identifiers harus English-friendly

## Larangan Komputasi

Codex **jangan**:
- menjalankan training berat
- menjalankan benchmark penuh
- melakukan install package baru
- menjalankan job yang membutuhkan GPU lokal
- melakukan eksperimen besar di lingkungan kerja saat ini

Jika perlu menguji kode:
- gunakan import checks
- gunakan config parsing checks
- gunakan synthetic tensors kecil
- gunakan dummy forward pass
- gunakan smoke test yang ringan

## Urutan Operasi yang Diharapkan

Untuk setiap milestone:
1. Baca file terkait dan pahami dependency
2. Tulis rencana perubahan singkat
3. Implementasikan perubahan kecil dan modular
4. Jalankan sanity check ringan
5. Simpan status perubahan ke log dokumentasi
6. Baru lanjut ke perubahan berikutnya

## Status Log

Setelah setiap perubahan besar, update file status, misalnya:
- `documentation.md`
atau
- `outputs/logs/dev_status.md`

Isi minimal:
- apa yang baru diubah
- file apa saja yang tersentuh
- apa yang sudah selesai
- apa yang masih pending
- asumsi yang diambil
- risiko atau potensi bug yang perlu dicek nanti

## Ketentuan Dataset

- Jangan mengubah kontrak dataset tanpa alasan kuat
- Loader benchmark harus tetap konsisten dengan data contract repo
- Jika ada ketidakjelasan path atau split, dokumentasikan asumsi secara eksplisit

## Ketentuan Arsitektur

Implementasikan modul berikut:
- frozen ViT wrapper
- shared core LoRA
- slot bank
- rank mask fixed-capacity
- task-state encoder
- history bank summaries
- horizon planner
- materialize action
- instance router
- incremental cosine head
- CHU
- incremental training loop

## Ketentuan Bootstrap

Task pertama harus:
- tanpa teacher model
- tanpa history-aware similarity
- tanpa KD
- tanpa feature retention
- tetap memiliki warm-up sensing
- tetap membentuk task_state
- menginisialisasi shared core dan bootstrap slot
- menjalankan light consolidation setelah selesai

## Ketentuan Planner

Planner minimal harus:
- layer-wise
- menerima task embedding dan history summaries
- menghasilkan novelty, conflict, rank budget, consolidation, shared gate
- memisahkan planner outputs dari materialized structural config

## Ketentuan CHU

Implementasi awal CHU harus heuristik:
- usage
- stability
- redundancy
- consolidation flag

Jangan mengganti CHU awal menjadi modul learned yang kompleks.

## Ketentuan Pengujian

Pengujian lokal yang diperbolehkan:
- import tests
- constructor tests
- YAML loading tests
- dummy batch forward
- planner output shape checks
- slot expansion checks
- classifier expansion checks
- logging path creation
- summary writer checks

Pengujian lokal yang tidak diperbolehkan:
- full training
- multi-epoch heavy benchmark runs
- package install tanpa persetujuan
- rebuild environment

## Ketentuan Kualitas Kode

- modul kecil dan jelas
- dependency antar-file eksplisit
- hindari file monster yang terlalu panjang jika bisa dipisah logis
- gunakan typing jika memungkinkan
- error message harus informatif
- hindari shortcut yang menyulitkan debugging

## Ketentuan Deliverable

Di akhir pekerjaan, repo harus memiliki:
- struktur rapi
- kode inti NH-LoRA
- configs YAML
- bash runners
- logging ke `outputs/logs/`
- mean/std summarization
- dokumentasi cara run di server SSH
- ringkasan final tentang apa yang sudah selesai dan apa yang belum divalidasi penuh

## Jika Menemui Ambiguitas

Jika menemukan detail yang ambigu:
1. jangan mengarang metode baru,
2. pilih implementasi paling konservatif,
3. catat asumsi di dokumentasi,
4. lanjutkan pekerjaan tanpa memblokir seluruh progres.
