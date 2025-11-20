# Dung Beetle Optimization Algorithm (DBO) Scheduler
**Distributed Task Scheduling for Multi-VM Execution**

Repository ini berisi implementasi penjadwalan tugas terdistribusi menggunakan **Dung Beetle Optimization Algorithm (DBO)**. Algoritma ini digunakan untuk mengalokasikan task ke VM secara optimal berdasarkan kapasitas CPU dan MIPS, dengan tujuan meminimalkan makespan, mengurangi bottleneck, dan meningkatkan utilisasi sumber daya.

Struktur repository telah dirancang modular, terdiri dari:
- `dbo_algorithm.py` → Implementasi algoritma DBO 
- `scheduler.py` → Eksekusi scheduling + simulasi waktu
- `dbo_results.csv` → Hasil penjadwalan DBO
- `dataset.txt` → Dataset tugas untuk penjadwalan

## Repository Structure

```
├── __pycache__/
├── .gitignore
├── README.md
├── dbo_algorithm.py
├── dbo_results.csv
├── dataset.txt
├── requirements.txt
└── scheduler.py
```

## 1. Overview Sistem

Hasil akhir berupa file `dbo_results.csv` yang berisi jadwal lengkap dengan kolom:
- **index**: Indeks task
- **task_name**: Nama task  
- **vm_assigned**: VM yang ditugaskan untuk task tersebut
- **start_time**: Waktu mulai eksekusi
- **exec_time**: Waktu eksekusi task
- **finish_time**: Waktu selesai eksekusi
- **wait_time**: Waktu tunggu sebelum eksekusi dimulai

## 2. Dung Beetle Optimization Algorithm (DBO)

DBO bekerja dengan prinsip meniru perilaku kumbang kotoran (*Scarabaeus sacer*) dalam mencari makanan:

### Operator Utama DBO:
1. **Ball-rolling** (25%): Eksplorasi berdasarkan defect dan jarak ke worst solution
2. **Dancing** (20%): Diversifikasi menggunakan fungsi trigonometri  
3. **Breeding** (20%): Eksploitasi di sekitar global best dengan shrinking region
4. **Foraging** (20%): Pencarian makanan dengan Gaussian perturbation
5. **Stealing** (15%): Interaksi sosial antar individu untuk mencuri posisi terbaik

### Parameter Algoritma:
```python
# Parameter utama dari paper
alpha = 0.6          # Defect coefficient
sigma = 0.4          # Random coefficient  
b_const = 0.5        # Ball constant
k_defect = 0.05      # Defect factor
S_steal = 0.5        # Stealing factor

# Probabilitas operator
p_roll = 0.25        # Ball-rolling probability
p_dance = 0.20       # Dancing probability  
p_breed = 0.20       # Breeding probability
p_forage = 0.20      # Foraging probability
p_steal = 0.15       # Stealing probability
```

Contoh kode inti (disederhanakan):
```python
def dbo_algorithm(tasks, vms, population=30, max_iter=200):
    # Inisialisasi populasi posisi kontinu
    population_cont = [[random.uniform(0, n_vm-1) for _ in range(n_tasks)] 
                      for _ in range(population)]
    
    for t in range(max_iter):
        R = 1.0 - (t / max_iter)  # Shrinking factor
        
        for i in range(population):
            # Pilih operator berdasarkan probabilitas
            if r < p_roll:
                # Ball-rolling operator
                new_x[d] = xi[d] + alpha * k_defect * (xi[d] - xi_prev[d]) + b_const * abs(xi[d] - X_worst[d])
            elif r < p_roll + p_dance:
                # Dancing operator  
                new_x[d] = xi[d] + tan(theta) * abs(xi[d] - xi_prev[d])
            # ... operator lainnya
```

Implementasi lengkap ada di `dbo_algorithm.py`.

## 3. Arsitektur dan Alur Kerja Algoritma DBO untuk Penjadwalan Tugas

### 3.1. Pemodelan Masalah
Setiap elemen dalam masalah penjadwalan dipetakan ke konsep perilaku kumbang kotoran:

| Konsep Perilaku DBO | Implementasi pada Penjadwalan Tugas |
|---------------------|--------------------------------------|
| **Posisi Kumbang** | Vektor Solusi - array dimana vektor[i] adalah indeks VM untuk task_i |
| **Bola Kotoran** | Resource VM yang dicari untuk mengeksekusi tugas |
| **Territory** | Ruang Pencarian - semua kemungkinan kombinasi jadwal |
| **Food Source** | Solusi Optimal dengan makespan terendah |
| **Population** | Kumpulan kandidat solusi jadwal |

### 3.2. Fungsi Fitness: Estimasi Makespan
Tujuan utama algoritma adalah meminimalkan makespan. Fungsi `fitness_from_cont` menghitung:

1. **Konversi Posisi Kontinu**: `vm_idx = round(position[i])` untuk task i
2. **Beban VM**: Untuk setiap VM, hitung total beban = Σ(task.cpu_load / vm.mips)  
3. **Makespan**: Nilai maksimum beban di antara semua VM

```python
def fitness_from_cont(pos):
    vm_loads = [0.0] * n_vm
    for i, task in enumerate(tasks):
        idx = int(round(pos[i]))
        vm_loads[idx] += task.cpu_load / vm_mips[idx]
    return max(vm_loads)  # Makespan estimate
```

### 3.3. Fase Pencarian dalam DBO

Algoritma secara dinamis menggunakan 5 operator berbeda:

#### 1. Ball-rolling (Eksplorasi)
```python
def_term = alpha * k_defect * (xi[d] - xi_prev[d])
Dx = abs(xi[d] - X_worst[d])
new_x[d] = xi[d] + def_term + b_const * Dx
```

#### 2. Dancing (Diversifikasi)  
```python
theta = random.uniform(0, π)
new_x[d] = xi[d] + tan(theta) * abs(xi[d] - xi_prev[d])
```

#### 3. Breeding (Eksploitasi)
```python
# Shrinking region around global best
Lb_star = max(X_star[d] * (1-R), 0)
Ub_star = min(X_star[d] * (1+R), n_vm-1)
new_x[d] = X_star[d] + b1 * (xi[d] - Lb_star) + b2 * (xi[d] - Ub_star)
```

#### 4. Foraging (Pencarian Makanan)
```python
C1 = random.gauss(0, 1)  # Gaussian noise
C2 = random.random()
new_x[d] = xi[d] + C1 * (xi[d] - Lbb[d]) + C2 * (xi[d] - Ubb[d])
```

#### 5. Stealing (Interaksi Sosial)
```python
X_bj = random_other_beetle_position
term = abs(xi[d] - X_star[d]) + abs(xi[d] - X_bj[d])
new_x[d] = X_best[d] + S_steal * random() * term
```

## 4. Cara Menjalankan

### 1. Setup Environment

```powershell
# Buat virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies (jika ada)
pip install -r requirements.txt
```

### 2. Buat File Environment (.env)
Buat file `.env` dengan konfigurasi VM:

```env
VM1_IP=10.15.42.77
VM2_IP=10.15.42.78  
VM3_IP=10.15.42.79
VM4_IP=10.15.42.80
VM_PORT=5000
DATASET_FILE=dataset.txt
RESULTS_FILE=dbo_results.csv
DBO_POPULATION=30
DBO_ITERATIONS=200
DBO_PROB_LOCAL=0.7
```

### 3. Siapkan Dataset

File `dataset.txt` berisi indeks tugas (1-10):
```
6
5
8
2
10
3
4
4
7
3
9
1
7
9
1
8
2
5
6
10
```

### 4. Jalankan Scheduler

```powershell
python scheduler.py
```

## 5. Konfigurasi VM

| VM | CPU Cores | RAM (GB) | MIPS | Karakteristik |
|----|-----------|----------|------|---------------|
| vm1 | 1 | 1 | 1000 | Light workload |
| vm2 | 2 | 2 | 2000 | Medium workload |  
| vm3 | 4 | 4 | 4000 | Heavy workload |
| vm4 | 8 | 4 | 6000 | Highest performance |


### Parameter Tuning:
```python
# Experiment dengan parameter ini
DBO_POPULATION = 50    
DBO_ITERATIONS = 300   
alpha = 0.8            
```

## 6. Dependencies

Jika menggunakan distributed execution:
```txt
asyncio
httpx  
pandas
python-dotenv
```

