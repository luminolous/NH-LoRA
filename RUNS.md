# RUNS.md

Dokumen ini menjelaskan cara menjalankan eksperimen NH-LoRA dengan pola:
- **1 bash file per benchmark**
- **1 YAML config per benchmark**
- **log disimpan ke `outputs/logs/`**
- **hasil per-seed disimpan terstruktur**
- **ringkasan mean/std dibuat otomatis**

> Catatan penting:
> - Training penuh dijalankan di mesin SSH utama, bukan di mesin lokal ringan.
> - Bash scripts di sini diasumsikan hanya memanggil kode repo yang sudah siap.
> - Gunakan `tmux`, `screen`, atau `nohup` di server SSH agar proses tetap berjalan saat koneksi terputus.

---

## Struktur Folder yang Disarankan

```text
configs/
  base.yaml
  cifar100.yaml
  cub200.yaml
  imagenet_r.yaml
  omnibenchmark.yaml
scripts/
  run_cifar100.sh
  run_cub200.sh
  run_imagenet_r.sh
  run_omnibenchmark.sh
  run_all.sh
outputs/
  logs/
  metrics/
  summaries/
  checkpoints/
```

---

## Konvensi Run

Setiap bash script benchmark harus:
1. membuat folder output jika belum ada,
2. membaca config YAML benchmark terkait,
3. menjalankan **5 seeds** secara default,
4. menyimpan log stdout/stderr ke `outputs/logs/`,
5. menyimpan metrics per-seed ke folder benchmark masing-masing,
6. memanggil summarizer untuk membuat mean dan std.

Contoh struktur hasil run:

```text
outputs/
  logs/
    cifar100_seed1.log
    cifar100_seed2.log
    ...
  metrics/
    cifar100/
      seed_1.json
      seed_2.json
      ...
    cub200/
      seed_1.json
      ...
  summaries/
    cifar100_summary.json
    cub200_summary.json
    imagenet_r_summary.json
    omnibenchmark_summary.json
  checkpoints/
    cifar100/
      seed_1/
      seed_2/
    cub200/
      seed_1/
```

---

## Cara Menjalankan per Benchmark

### CIFAR-100

```bash
bash scripts/run_cifar100.sh
```

### CUB-200-2011

```bash
bash scripts/run_cub200.sh
```

### ImageNet-R

```bash
bash scripts/run_imagenet_r.sh
```

### OmniBenchmark

```bash
bash scripts/run_omnibenchmark.sh
```

### Menjalankan Semua Benchmark Secara Berurutan

```bash
bash scripts/run_all.sh
```

---

## Menjalankan di Server SSH dengan `nohup`

Contoh:

```bash
nohup bash scripts/run_cifar100.sh > outputs/logs/nohup_cifar100.out 2>&1 &
```

Contoh untuk benchmark lain:

```bash
nohup bash scripts/run_cub200.sh > outputs/logs/nohup_cub200.out 2>&1 &
nohup bash scripts/run_imagenet_r.sh > outputs/logs/nohup_imagenet_r.out 2>&1 &
nohup bash scripts/run_omnibenchmark.sh > outputs/logs/nohup_omnibenchmark.out 2>&1 &
```

---

## Menjalankan di `tmux`

Contoh ringkas:

```bash
tmux new -s nhlora_cifar
bash scripts/run_cifar100.sh
```

Detach:

```bash
Ctrl+B lalu D
```

Attach kembali:

```bash
tmux attach -t nhlora_cifar
```

---

## Catatan Config

Semua config benchmark mewarisi ide dari `configs/base.yaml`.
Isi benchmark-specific YAML hanya override hal-hal yang berbeda, misalnya:
- nama benchmark
- path dataset
- jumlah tasks
- classes per task
- epoch
- batch size
- selected blocks
- output directory

Jika implementasi parser belum mendukung inheritance YAML otomatis, lakukan salah satu:
1. flatten seluruh config per benchmark, atau
2. tambahkan helper merge config secara eksplisit di kode.

---

## Contract Output yang Disarankan

Per-seed metrics JSON minimal memuat:
- `benchmark`
- `seed`
- `final_avg_acc`
- `avg_inc_acc`
- `forgetting`
- `backward_transfer`
- `last_task_acc`
- `parameter_growth`
- `total_active_rank`
- `opened_slots`
- `pruned_slots`
- `train_time_sec`
- `notes` (opsional)

Summary JSON minimal memuat:
- `benchmark`
- `num_seeds`
- mean dan std untuk semua metrik utama
- timestamp
- config name

---

## Catatan Validasi

Selama pengembangan lokal, yang boleh diuji hanya:
- config loading,
- output path creation,
- logging setup,
- dummy argument wiring,
- smoke test ringan.

Training nyata dilakukan di mesin SSH utama.
