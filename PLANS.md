# PLANS.md

## Prinsip Umum

Milestone dibagi untuk kontrol pekerjaan, tetapi **final milestone tetap full implementation NH-LoRA**.

Milestone bukan pemangkasan konsep.
Milestone adalah urutan kerja agar implementasi tetap stabil, dapat diverifikasi, dan tidak menyimpang dari design paper.

## Final Target

Pada akhir seluruh milestone, repo harus memiliki:
- implementasi penuh modul inti NH-LoRA
- support 4 benchmark
- bash runner per benchmark
- YAML config per benchmark
- logging ke `outputs/logs/`
- summary mean/std untuk 5 seeds
- sanity checks ringan
- dokumentasi cara run

---

## Milestone 1 — Repo Skeleton dan Kontrak Dasar

### Tujuan
Membangun kerangka repo NH-LoRA yang bersih dan fokus.

### Hasil yang harus ada
- struktur folder final
- migrasi/rapikan dataset loader yang dibutuhkan
- config loader YAML
- bash runner skeleton
- logger dasar
- output folders dibuat otomatis
- status log kerja

### Acceptance criteria
- repo structure sesuai SPEC
- `configs/` dan `scripts/` aktif
- `outputs/logs/` dibuat otomatis saat run
- minimal satu command dry-run bisa membaca config tanpa error impor besar

---

## Milestone 2 — Data Pipeline dan Benchmark Wiring

### Tujuan
Memastikan benchmark dapat dimuat dengan kontrak yang konsisten.

### Hasil yang harus ada
- dataset wrapper untuk CIFAR-100
- dataset wrapper untuk CUB-200-2011
- dataset wrapper untuk ImageNet-R
- dataset wrapper untuk OmniBenchmark
- task split support
- data manager yang rapi
- dokumentasi path dataset

### Acceptance criteria
- setiap benchmark bisa diinisialisasi dari YAML
- split task dapat dibentuk
- shape / number of tasks / class split dapat dicek dengan dry test
- tidak ada training berat dijalankan

---

## Milestone 3 — Backbone dan Layer NH-LoRA Dasar

### Tujuan
Membangun frozen ViT + komponen low-rank inti.

### Hasil yang harus ada
- frozen ViT wrapper
- selected block handling
- shared core LoRA module
- slot bank module
- fixed-capacity rank mask
- helper untuk open slot / expand rank / freeze slot

### Acceptance criteria
- forward backbone tanpa NH-LoRA tetap jalan
- forward backbone dengan NH-LoRA layer wrapper jalan
- rank mask tidak mengubah tensor shape
- slot bank bisa menambah slot baru secara terkontrol

---

## Milestone 4 — Task-State Encoder dan Bootstrap Mode

### Tujuan
Mengimplementasikan warm-up sensing dan bootstrap task pertama.

### Hasil yang harus ada
- warm-up feature statistics
- gradient sketch ringan
- uncertainty estimate
- similarity placeholder
- auxiliary warm-up head
- task-state encoder
- bootstrap planner policy
- bootstrap routing rule

### Acceptance criteria
- task 1 dapat menghasilkan `task_state`
- task 1 berjalan tanpa teacher model
- task 1 berjalan tanpa history bank aktif
- task 1 tidak memanggil KD/feature retention
- bootstrap shared core + bootstrap slot benar-benar dimaterialisasi

---

## Milestone 5 — Horizon Planner dan Materialize Action

### Tujuan
Mengimplementasikan planner layer-wise sesuai paper.

### Hasil yang harus ada
- history bank summaries
- history-aware aggregation
- planner outputs:
  - novelty
  - conflict
  - rank budget
  - consolidation signal
  - shared gate
- materialize action logic
- support action:
  - reuse_shared
  - expand_rank_existing_slot
  - open_new_slot
  - freeze_old_strong_retention

### Acceptance criteria
- planner dapat menerima `task_state` + history summaries
- planner menghasilkan action valid per selected layer
- materialize action menghasilkan config konkret untuk forward
- fallback saat slot count mencapai batas maksimum

---

## Milestone 6 — Instance Router dan Forward Lengkap

### Tujuan
Menghubungkan planner output dengan forward pass NH-LoRA penuh.

### Hasil yang harus ada
- instance query projection
- top-k slot selection
- routing coefficients
- effective weight construction:
  - frozen weight
  - shared update
  - slot updates
- route info untuk loss logging

### Acceptance criteria
- current_model forward mengembalikan:
  - logits
  - features
  - route_info
- routing sparse berjalan
- planner config dipakai di forward
- active slots dan rank config benar-benar mempengaruhi update efektif

---

## Milestone 7 — Training Loop Lengkap dan Losses

### Tujuan
Membangun training loop utama NH-LoRA.

### Hasil yang harus ada
- task loop incremental
- classifier expansion
- teacher snapshot untuk task > 1
- losses:
  - cls
  - kd
  - feat
  - orth
  - rank
  - grow
  - route
- optimizer wiring
- lightweight checkpoint save

### Acceptance criteria
- task 1 dan task > 1 memakai cabang loss berbeda
- old classes digunakan untuk KD
- classifier head expand saat task baru
- checkpoint ringan dan logs tersimpan
- log train minimal per epoch tersimpan ke file

---

## Milestone 8 — CHU, Summary, dan Run Interface Final

### Tujuan
Menyelesaikan post-task logic dan antarmuka run final.

### Hasil yang harus ada
- slot usage estimation
- slot stability estimation
- slot redundancy estimation
- heuristic merge/prune/keep-or-freeze
- save task summary ke history bank
- bash runner per benchmark
- 5-seed summary mean/std
- result summarizer

### Acceptance criteria
- CHU dapat dipanggil setelah task selesai
- history bank bertambah tiap task
- bash runner benchmark aktif
- multi-seed loop aktif
- mean/std summary dibuat otomatis
- logs tersimpan di `outputs/logs/`

---

## Milestone 9 — Validasi Ringan dan Dokumentasi Run

### Tujuan
Menyelesaikan repo agar siap dipindah dan dijalankan di mesin SSH utama.

### Hasil yang harus ada
- dry tests ringan
- import tests
- config parsing tests
- shape/forward sanity checks
- README / RUNS documentation
- daftar asumsi implementasi

### Acceptance criteria
- tidak ada install tambahan yang diwajibkan oleh Codex
- tidak ada benchmark berat dijalankan lokal
- ada panduan jelas cara run di server SSH
- status akhir menjelaskan:
  - yang sudah selesai
  - yang belum divalidasi penuh
  - yang perlu dijalankan di server SSH

---

## Catatan Eksekusi

Codex harus bekerja milestone per milestone.
Namun, milestone terakhir **tetap berarti full implementation NH-LoRA**, bukan versi mini.
Jika waktu mepet, yang boleh dikurangi adalah tingkat validasi eksperimen lokal, bukan inti metode.
