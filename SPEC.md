# SPEC.md

## Tujuan Repo

Repo ini adalah implementasi **NH-LoRA** untuk **rehearsal-free class-incremental learning** berbasis **frozen Vision Transformer**.

Repo ini **bukan** toolbox continual learning umum.
Repo ini harus tetap bersih, fokus, dan paper-oriented.
Hanya file benchmark/data loading yang benar-benar diperlukan yang boleh diadopsi dari sumber lain.

## Sumber Kebenaran Metode

Urutan prioritas sumber kebenaran:
1. Design paper NH-LoRA terbaru (otoritatif)
2. Keputusan eksplisit di file markdown repo ini
3. Implementasi sederhana yang paling dekat dengan maksud paper, jika ada detail kecil yang belum sepenuhnya eksplisit

Jika ada konflik antara kode dan design paper, **ikuti design paper**.

## Tujuan Akhir

Milestone dibagi untuk pengendalian progres, tetapi **target akhir tetap implementasi penuh NH-LoRA** sesuai design paper, bukan versi yang dipangkas secara konseptual.

## Cakupan Metode yang Wajib Ada

NH-LoRA wajib mengimplementasikan komponen berikut:

1. Frozen Vision Backbone
   - Main target: ViT-B/16-IN21K
   - Backbone fully frozen
   - Trainable parameters hanya modul NH-LoRA dan classifier head

2. Shared Core LoRA
   - Shared low-rank adapter per selected block
   - Bersifat reusable lintas task
   - Slow plasticity
   - Shared rank kecil

3. Expandable Task Slot Bank
   - Slot adapter per selected block
   - Slot dapat bertambah
   - Dynamic rank menggunakan fixed-capacity rank mask
   - Tidak boleh realokasi tensor tiap kali rank berubah

4. Task-State Encoder (TSE)
   - Warm-up sensing sebelum full training task
   - Input statistik task: feature mean, feature dispersion, gradient sketch, similarity, uncertainty
   - Output task embedding `z_t`

5. Horizon Planner (HP)
   - Layer-wise planner
   - Output minimal:
     - novelty
     - conflict
     - rank budget
     - consolidation signal
     - shared gate
   - Planner harus memakai task embedding dan history summaries
   - Materialisasi aksi struktural harus dipisah dari sinyal planner mentah

6. Materialize Action
   - Mengubah sinyal planner menjadi konfigurasi struktural konkret
   - Minimal menghasilkan:
     - shared gate
     - active slot candidates
     - rank configuration
     - consolidation flag
   - Action space:
     - `reuse_shared`
     - `expand_rank_existing_slot`
     - `open_new_slot`
     - `freeze_old_strong_retention`

7. Instance Router
   - Sparse routing per instance
   - Top-K slot activation
   - Menghasilkan routing coefficients
   - Dipakai dalam forward layer NH-LoRA

8. Incremental Cosine Classifier Head
   - Cosine classifier
   - Head expand saat task baru datang
   - Support weight imprinting atau warm-start initialization

9. Consolidation and Homeostasis Unit (CHU)
   - Post-task merge / prune / keep-or-freeze
   - Heuristic-based initial implementation
   - Menggunakan usage, stability, redundancy, dan consolidation flag

10. History Bank
   - Menyimpan summary statistik task lama
   - Tidak boleh menyimpan raw data task lama

## Bootstrap Task Pertama

Task pertama harus menggunakan mode bootstrap khusus:
- no teacher model
- no history-aware similarity
- no old classes
- no KD loss
- no feature retention loss
- initialize small shared memory + bootstrap slot
- deterministic routing jika hanya ada satu slot bootstrap
- light consolidation setelah task pertama

## Objective / Losses yang Wajib Didukung

Untuk task > 1:
- `L_cls`
- `L_kd`
- `L_feat`
- `L_orth`
- `L_rank`
- `L_grow`
- `L_route`

Untuk task 1:
- `L_cls`
- `L_orth`
- `L_rank`
- `L_route`

Dengan:
- `L_kd = 0` pada task 1
- `L_feat = 0` pada task 1
- `L_grow` boleh dimatikan atau sangat kecil pada task 1

## Benchmark yang Didukung

Repo harus mendukung benchmark berikut:
- CIFAR-100
- CUB-200-2011
- ImageNet-R
- OmniBenchmark

Catatan:
- OmniBenchmark boleh menjadi benchmark paling akhir yang dieksekusi
- Namun support kodenya tetap harus ada

## Evaluasi dan Output

Minimal metrik yang perlu dicatat:
- final average accuracy
- average incremental accuracy
- forgetting
- backward transfer
- last-task accuracy

Minimal metrik efisiensi:
- parameter growth per task
- total active rank
- number of opened slots
- number of pruned slots
- training time per task
- inference overhead (ringan / estimatif jika belum detail)

## Format Run

Setiap benchmark dijalankan menggunakan:
- 1 file bash khusus
- 1 file YAML config khusus

Semua run harus mendukung:
- multi-seed
- default target: 5 seeds
- summary mean dan std otomatis

## Struktur Folder yang Diinginkan

```text
configs/
scripts/
src/
  datasets/
  backbones/
  models/
  engine/
  utils/
outputs/
  logs/
  metrics/
  summaries/
  checkpoints/
```

## Ketentuan Logging

Semua proses training dan evaluasi harus menyimpan log ke:
- `outputs/logs/`

Tujuannya:
- menjaga progres jika komputer crash
- menjaga progres jika koneksi SSH putus
- mempermudah audit dan debugging

## Ketentuan Bahasa

- Prompt kerja dan dokumen repo ini boleh dalam Bahasa Indonesia
- Semua kode sumber harus ditulis dalam English-friendly style
- Semua comments di kode harus berbahasa Inggris
- Nama function/class/module harus berbahasa Inggris

## Larangan

Codex tidak boleh:
- mengubah metode menjadi metode lain
- menghapus modul inti NH-LoRA
- menambah framework besar yang tidak perlu
- menginstal package baru tanpa alasan yang sangat kuat
- menjalankan training berat / benchmark penuh di lingkungan lokal
- mengasumsikan resource GPU lokal tersedia
- mengubah data contract seenaknya

## Prinsip Implementasi

Jika ada ambiguitas kecil, pilih implementasi yang:
1. paling sederhana,
2. paling stabil,
3. paling dekat dengan maksud design paper,
4. paling mudah diuji tanpa komputasi berat.
