# DBO Scheduler for Distributed Task Execution

Sistem penjadwalan tugas terdistribusi menggunakan algoritma Dung Beetle Optimizer (DBO) untuk mengoptimalkan eksekusi task pada multiple VM.

## Prerequisites

- Python 3.8 atau lebih tinggi
- Akses jaringan ke VM server yang ditentukan
- Virtual environment (recommended)

##  Quick Start



### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
.\venv\Scripts\Activate.ps1

# Jika ada error execution policy, jalankan:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/macOS:**
```bash
# Buat virtual environment
python3 -m venv venv

# Aktivasi virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Configuration
```bash
# Copy template .env (sudah disediakan)
# Edit .env jika perlu mengubah konfigurasi default
```

### 5. Prepare Dataset
Buat file `dataset.txt` dengan format:
```
1
2
3
4
5
```
Setiap baris berisi satu integer yang merepresentasikan task index.

### 6. Run the Scheduler
```bash
python scheduler.py
```

## ⚙️ Configuration

### Environment Variables (file .env)

| Variable | Default | Description |
|----------|---------|-------------|
| `VM1_IP` | 10.15.42.77 | IP VM dengan 1 CPU, 1GB RAM |
| `VM2_IP` | 10.15.42.78 | IP VM dengan 2 CPU, 2GB RAM |
| `VM3_IP` | 10.15.42.79 | IP VM dengan 4 CPU, 4GB RAM |
| `VM4_IP` | 10.15.42.80 | IP VM dengan 8 CPU, 4GB RAM |
| `VM_PORT` | 5000 | Port untuk HTTP requests |
| `DATASET_FILE` | dataset.txt | Path ke file dataset |
| `RESULTS_FILE` | dbo_results.csv | Output CSV file |

### DBO Algorithm Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `DBO_POPULATION` | 30 | Ukuran populasi untuk algoritma DBO |
| `DBO_MAX_ITER` | 200 | Maksimum iterasi algoritma |
| `DBO_PROB_LOCAL` | 0.7 | Probabilitas local search |
| `BASE_MIPS_PER_CORE` | 1000 | Base MIPS per core CPU |

## Output

Setelah eksekusi selesai, sistem akan menghasilkan:

1. **Console Output**: Metrik real-time dan progress
2. **CSV File** (`dbo_results.csv`): Detail eksekusi setiap task
3. **Performance Metrics**:
   - Makespan (total execution time)
   - Throughput (tasks/second)
   - Resource utilization
   - Load balancing metrics


## Usage
```bash
python scheduler.py
```
