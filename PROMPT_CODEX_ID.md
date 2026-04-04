# PROMPT_CODEX_ID.md

Anda sedang mengerjakan implementasi repo riset **NH-LoRA**.

Tujuan utama:
Membangun repo NH-LoRA yang bersih, fokus, dan paper-oriented untuk rehearsal-free class-incremental learning berbasis frozen ViT, sesuai design paper NH-LoRA terbaru.

## Sumber kebenaran
1. Design paper NH-LoRA terbaru adalah sumber kebenaran utama.
2. File `SPEC.md`, `PLANS.md`, dan `IMPLEMENT.md` adalah aturan implementasi yang harus diikuti.
3. Kode yang sudah ada di repo adalah baseline kerja, tetapi jika bertentangan dengan paper, ikuti paper.

## Target akhir
Milestone digunakan untuk mengontrol progres, tetapi **target akhir tetap implementasi penuh NH-LoRA**, bukan versi mini.

## Modul inti yang wajib diimplementasikan
- Frozen Vision Transformer backbone
- Shared Core LoRA
- Expandable Task Slot Bank
- Dynamic rank mask dengan fixed max rank
- Task-State Encoder
- Horizon Planner
- Materialize Action
- Instance Router
- Incremental cosine classifier head
- Consolidation and Homeostasis Unit
- Bootstrap mode untuk task pertama
- History bank berbasis summary statistics
- Training loop rehearsal-free CIL lengkap
- Mean/std summarization untuk multi-seed runs

## Benchmark yang harus didukung
- CIFAR-100
- CUB-200-2011
- ImageNet-R
- OmniBenchmark

## Aturan run
- Setiap benchmark harus punya file bash sendiri di `scripts/`
- Setiap benchmark harus punya config YAML sendiri di `configs/`
- Semua log train/eval harus disimpan ke `outputs/logs/`
- Semua summary hasil harus disimpan ke `outputs/summaries/`
- Jalankan multi-seed logic untuk 5 seeds
- Buat ringkasan mean dan std otomatis

## Aturan penting
- Jangan ubah metode menjadi metode lain
- Jangan buang modul inti NH-LoRA
- Jangan ubah repo menjadi toolbox umum continual learning
- Jangan menambah baseline atau framework besar yang tidak diminta
- Jangan install package baru
- Jangan menjalankan komputasi berat
- Jangan menjalankan benchmark penuh di lingkungan kerja saat ini
- Jangan mengasumsikan GPU lokal tersedia

## Bahasa
- Komunikasi dan dokumen kerja boleh dalam Bahasa Indonesia
- Semua source code harus ditulis dalam bahasa Inggris
- Semua komentar di kode harus dalam bahasa Inggris
- Nama class/function/module harus English-friendly

## Strategi kerja
1. Pertama, inspeksi repo dan pahami struktur saat ini.
2. Lalu cocokkan repo dengan `SPEC.md`.
3. Implementasikan pekerjaan milestone demi milestone sesuai `PLANS.md`.
4. Untuk setiap milestone:
   - tulis rencana singkat,
   - implementasikan perubahan modular,
   - lakukan sanity check ringan,
   - update status log.
5. Jika ada detail paper yang ambigu, pilih implementasi paling sederhana yang tetap setia pada maksud paper, lalu catat asumsi tersebut.
6. Jangan melakukan pekerjaan yang membuat repo sulit diaudit.

## Sanity checks yang diperbolehkan
- import checks
- config parsing checks
- synthetic tensor checks
- forward smoke tests kecil
- slot expansion checks
- planner output checks
- classifier expansion checks
- logging path checks
- summary writer checks

## Deliverable yang diharapkan
- repo rapi dan fokus
- implementasi inti NH-LoRA lengkap
- configs YAML
- bash runners
- logging ke `outputs/logs/`
- ringkasan mean/std
- dokumentasi run di server SSH
- status akhir yang menjelaskan:
  - apa yang sudah selesai,
  - apa yang masih pending,
  - apa yang belum divalidasi penuh di benchmark nyata

Jika ada konflik antara kode lama dan design paper NH-LoRA, ikuti design paper.
Jika ada konflik antara simplifikasi implementasi dan inti metode, pertahankan inti metode.
