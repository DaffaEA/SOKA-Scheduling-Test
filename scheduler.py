#!/usr/bin/env python3
"""
dbo_scheduler.py

Scheduler asynchronous yang menggunakan algoritma DBO untuk
menugaskan task ke 4 VM fisik via HTTP GET /task/{index}.

- Dataset: file berisi satu integer indeks per baris.
- Task length (MI): index^2 * 10000
- VM MIPS (estimasi): cpu_cores * BASE_MIPS_PER_CORE (default 1000)
- Menghasilkan CSV results dan menampilkan metrik
"""

import asyncio
import httpx
import time
from datetime import datetime
import csv
import pandas as pd
import sys
import os
from dotenv import load_dotenv
from collections import namedtuple
import random
from typing import List, Dict

# --- Konfigurasi Lingkungan ---
load_dotenv()

VM_SPECS = {
    'vm1': {'ip': os.getenv("VM1_IP"), 'cpu': 1, 'ram_gb': 1},
    'vm2': {'ip': os.getenv("VM2_IP"), 'cpu': 2, 'ram_gb': 2},
    'vm3': {'ip': os.getenv("VM3_IP"), 'cpu': 4, 'ram_gb': 4},
    'vm4': {'ip': os.getenv("VM4_IP"), 'cpu': 8, 'ram_gb': 4},
}

VM_PORT = int(os.getenv("VM_PORT", "5000"))
DATASET_FILE = os.getenv("DATASET_FILE", "dataset.txt")
RESULTS_FILE = os.getenv("RESULTS_FILE", "dbo_results.csv")

# DBO Parameters (adapted from Java example)
POPULATION = int(os.getenv("DBO_POPULATION", "30"))
MAX_ITER = int(os.getenv("DBO_MAX_ITER", "200"))
PROB_LOCAL = float(os.getenv("DBO_PROB_LOCAL", "0.7"))
BASE_MIPS_PER_CORE = int(os.getenv("BASE_MIPS_PER_CORE", "1000"))

rng = random.Random(42)

VM = namedtuple('VM', ['name', 'ip', 'cpu_cores', 'ram_gb', 'mips'])
Task = namedtuple('Task', ['id', 'name', 'index', 'cpu_load'])

# --- Fungsi Helper & Definisi Task ---

def get_task_load(index: int) -> int:
    """Rumus panjang task: index^2 * 10000 (dalam MI)."""
    return index * index * 10000

def load_tasks(dataset_path: str) -> List[Task]:
    if not os.path.exists(dataset_path):
        print(f"Error: File dataset '{dataset_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
        
    tasks = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                index = int(s)
                # tidak memaksa rentang — biarkan user menentukan
                cpu_load = get_task_load(index)
                task_name = f"task-{index}-{i}"
                tasks.append(Task(
                    id=i,
                    name=task_name,
                    index=index,
                    cpu_load=cpu_load,
                ))
            except ValueError:
                print(f"Peringatan: Mengabaikan baris {i+1} yang tidak valid: '{s}'")
    print(f"Berhasil memuat {len(tasks)} tugas dari {dataset_path}")
    return tasks

# ====================================================================
# ================== DBO ALGORITHM (Python version) ===================
# ====================================================================

def dbo_algorithm(tasks: List[Task], vms: List[VM],
                  population: int = POPULATION,
                  max_iter: int = MAX_ITER,
                  prob_local: float = PROB_LOCAL) -> Dict[int, str]:
    """
    DBO untuk meminimalkan makespan.
    Mengembalikan dict mapping task.id -> vm.name
    """

    n_cloud = len(tasks)
    n_vm = len(vms)
    if n_cloud == 0 or n_vm == 0:
        return {}

    # compute VM "total_mips" used in fitness (here MIPS * number_of_pes ~ mips)
    vm_mips = [vm.mips for vm in vms]

    # create index mapping for vm id -> vm name
    vm_names = [vm.name for vm in vms]

    # Initialize population (each individual: list of vm indices for each task)
    population_list = []
    for _ in range(population):
        assign = [rng.randrange(n_vm) for _ in range(n_cloud)]
        population_list.append(assign)

    # fitness function: makespan (lower is better)
    def fitness(assign):
        vm_loads = [0.0] * n_vm
        for i, t in enumerate(tasks):
            vm_idx = assign[i]
            # ET = task MI / total_mips(vm)
            et = t.cpu_load / vm_mips[vm_idx]
            vm_loads[vm_idx] += et
        return max(vm_loads)

    # initial best
    best = None
    best_fit = float('inf')
    for ind in population_list:
        f = fitness(ind)
        if f < best_fit:
            best_fit = f
            best = ind.copy()

    # DBO iterations: simple discrete operations inspired by the Java version
    for it in range(max_iter):
        for ind_idx in range(len(population_list)):
            ind = population_list[ind_idx]
            new_ind = ind.copy()
            for k in range(n_cloud):
                r = rng.random()
                if r < 0.2:
                    # follow best (exploit)
                    new_ind[k] = best[k]
                else:
                    if rng.random() < prob_local:
                        # local random modification (explore)
                        new_ind[k] = rng.randrange(n_vm)
                    # else keep same
            new_fit = fitness(new_ind)
            old_fit = fitness(ind)
            if new_fit < old_fit:
                population_list[ind_idx] = new_ind

        # update best
        for ind in population_list:
            f = fitness(ind)
            if f < best_fit:
                best_fit = f
                best = ind.copy()

    # build mapping task.id -> vm.name
    assignment = {}
    for i, vm_idx in enumerate(best):
        assignment[tasks[i].id] = vm_names[vm_idx]

    print(f"DBO: Optimized makespan (estimated) = {best_fit:.4f} seconds (based on MIPS model)")
    return assignment

# --- Eksekutor Tugas Asinkron ---

async def execute_task_on_vm(task: Task, vm: VM, client: httpx.AsyncClient, 
                              vm_semaphore: asyncio.Semaphore, results_list: list):
    """
    Mengirim request GET ke VM yang ditugaskan, dibatasi oleh semaphore VM.
    Mencatat hasil dan waktu.
    """
    url = f"http://{vm.ip}:{VM_PORT}/task/{task.index}"
    task_start_time = None
    task_finish_time = None
    task_exec_time = -1.0
    task_wait_time = -1.0
    
    wait_start_mono = time.monotonic()
    
    try:
        async with vm_semaphore:
            # Waktu tunggu selesai, eksekusi dimulai
            task_wait_time = time.monotonic() - wait_start_mono
            
            print(f"Mengeksekusi {task.name} (idx: {task.id}, endpoint: {task.index}) di {vm.name} (IP: {vm.ip})...")
            
            # Catat waktu mulai
            task_start_mono = time.monotonic()
            task_start_time = datetime.now()
            
            # Kirim request GET
            response = await client.get(url, timeout=300.0) # Timeout 5 menit
            response.raise_for_status()
            
            # Catat waktu selesai
            task_finish_time = datetime.now()
            task_exec_time = time.monotonic() - task_start_mono
            
            print(f"Selesai {task.name} di {vm.name}. Waktu eksekusi: {task_exec_time:.4f}s")
            
    except httpx.HTTPStatusError as e:
        print(f"Error HTTP pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    except httpx.RequestError as e:
        print(f"Error Request pada {task.name} di {vm.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error tidak diketahui pada {task.name} di {vm.name}: {e}", file=sys.stderr)
        
    finally:
        if task_start_time is None:
            task_start_time = datetime.now()
        if task_finish_time is None:
            task_finish_time = datetime.now()
            
        results_list.append({
            "index": task.id,
            "task_name": task.name,
            "vm_assigned": vm.name,
            "start_time": task_start_time,
            "exec_time": task_exec_time,
            "finish_time": task_finish_time,
            "wait_time": task_wait_time
        })

# --- Fungsi Paska-Proses & Metrik (sama seperti contohmu) ---

def write_results_to_csv(results_list: list, filename: str = RESULTS_FILE):
    """Menyimpan hasil eksekusi ke file CSV."""
    if not results_list:
        print("Tidak ada hasil untuk ditulis ke CSV.", file=sys.stderr)
        return

    # Urutkan berdasarkan 'index' untuk keterbacaan
    results_list.sort(key=lambda x: x['index'])

    headers = ["index", "task_name", "vm_assigned", "start_time", "exec_time", "finish_time", "wait_time"]
    
    # Format datetime agar lebih mudah dibaca di CSV (relatif terhadap min start)
    formatted_results = []
    min_start = min(item['start_time'] for item in results_list)
    for r in results_list:
        new_r = r.copy()
        new_r['start_time'] = (r['start_time'] - min_start).total_seconds()
        new_r['finish_time'] = (r['finish_time'] - min_start).total_seconds()
        formatted_results.append(new_r)

    formatted_results.sort(key=lambda item: item['start_time'])

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(formatted_results)
        print(f"\nData hasil eksekusi disimpan ke {filename}")
    except IOError as e:
        print(f"Error menulis ke CSV {filename}: {e}", file=sys.stderr)

def calculate_and_print_metrics(results_list: list, vms: List[VM], total_schedule_time: float):
    try:
        df = pd.DataFrame(results_list)
    except Exception as e:
        print("Error: Gagal membuat DataFrame dari hasil.", file=sys.stderr)
        return

    # Konversi kolom waktu
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['finish_time'] = pd.to_datetime(df['finish_time'])
    
    # Filter 'failed' tasks (exec_time < 0)
    success_df = df[df['exec_time'] > 0].copy()
    
    if success_df.empty:
        print("Tidak ada tugas yang berhasil diselesaikan. Metrik tidak dapat dihitung.")
        return

    num_tasks = len(success_df)
    
    # Hitung metrik
    total_cpu_time = success_df['exec_time'].sum()
    total_wait_time = success_df['wait_time'].sum()
    
    avg_exec_time = success_df['exec_time'].mean()
    avg_wait_time = success_df['wait_time'].mean()
    
    # Waktu mulai & selesai relatif terhadap awal
    min_start = success_df['start_time'].min()
    success_df['rel_start_time'] = (success_df['start_time'] - min_start).dt.total_seconds()
    success_df['rel_finish_time'] = (success_df['finish_time'] - min_start).dt.total_seconds()
    
    avg_start_time = success_df['rel_start_time'].mean()
    avg_finish_time = success_df['rel_finish_time'].mean()
    
    makespan = total_schedule_time # Waktu dari eksekusi pertama hingga terakhir
    throughput = num_tasks / makespan if makespan > 0 else 0
    
    # Imbalance Degree (Degree of Imbalance)
    vm_exec_times = success_df.groupby('vm_assigned')['exec_time'].sum()
    max_load = vm_exec_times.max()
    min_load = vm_exec_times.min()
    avg_load = vm_exec_times.mean()
    imbalance_degree = (max_load - min_load) / avg_load if avg_load > 0 else 0
    
    # Resource Utilization
    total_cores = sum(vm.cpu_cores for vm in vms)
    total_available_cpu_time = makespan * total_cores
    resource_utilization = total_cpu_time / total_available_cpu_time if total_available_cpu_time > 0 else 0

    # Tampilkan Metrik
    print("\n--- Hasil ---")
    print(f"Total Tugas Selesai       : {num_tasks}")
    print(f"Makespan (Waktu Total)    : {makespan:.4f} detik")
    print(f"Throughput                : {throughput:.4f} tugas/detik")
    print(f"Total CPU Time            : {total_cpu_time:.4f} detik")
    print(f"Total Wait Time           : {total_wait_time:.4f} detik")
    print(f"Average Start Time (rel)  : {avg_start_time:.4f} detik")
    print(f"Average Execution Time    : {avg_exec_time:.4f} detik")
    print(f"Average Finish Time (rel) : {avg_finish_time:.4f} detik")
    print(f"Imbalance Degree          : {imbalance_degree:.4f}")
    print(f"Resource Utilization (CPU): {resource_utilization:.4%}")

# --- 6. Fungsi Main ---

async def main():
    # 1. Inisialisasi VM list (tambah field mips)
    vms = []
    for name, spec in VM_SPECS.items():
        ip = spec.get('ip')
        if not ip:
            print(f"Error: IP tidak ditemukan untuk {name}. Pastikan .env memiliki {name.upper()}_IP.", file=sys.stderr)
            sys.exit(1)
        cpu = int(spec.get('cpu', 1))
        ram = int(spec.get('ram_gb', 1))
        mips = cpu * BASE_MIPS_PER_CORE
        vms.append(VM(name, ip, cpu, ram, mips))
    
    tasks = load_tasks(DATASET_FILE)
    if not tasks:
        print("Tidak ada tugas untuk dijadwalkan. Keluar.", file=sys.stderr)
        return
        
    tasks_dict = {task.id: task for task in tasks}
    vms_dict = {vm.name: vm for vm in vms}

    # 2. Jalankan Algoritma Penjadwalan (DBO)
    print("\nMenjalankan DBO scheduling...")
    best_assignment = dbo_algorithm(tasks, vms, POPULATION, MAX_ITER, PROB_LOCAL)
    
    print("\nPenugasan Tugas Terbaik Ditemukan (contoh 20 pertama):")
    for i in range(min(20, len(tasks))):
        vm_name = best_assignment.get(i, "UNASSIGNED")
        print(f"  - Tugas {i} -> {vm_name}")
    if len(tasks) > 20:
        print("  - ... etc.")

    # 3. Siapkan Eksekusi
    results_list = []
    
    # Buat semaphore untuk setiap VM berdasarkan core CPU
    vm_semaphores = {vm.name: asyncio.Semaphore(vm.cpu_cores) for vm in vms}
    
    # Buat satu HTTP client untuk semua request
    async with httpx.AsyncClient() as client:
        
        # Siapkan semua coroutine tugas
        all_task_coroutines = []
        for task_id, vm_name in best_assignment.items():
            task = tasks_dict[task_id]
            vm = vms_dict[vm_name]
            sem = vm_semaphores[vm_name]
            
            all_task_coroutines.append(
                execute_task_on_vm(task, vm, client, sem, results_list)
            )
            
        print(f"\nMemulai eksekusi {len(all_task_coroutines)} tugas secara paralel...")
        
        # 4. Jalankan Semua Tugas dan Ukur Waktu Total
        schedule_start_time = time.monotonic()
        
        # gather runs until all finished
        await asyncio.gather(*all_task_coroutines)
        
        schedule_end_time = time.monotonic()
        total_schedule_time = schedule_end_time - schedule_start_time
        
        print(f"\nSemua eksekusi tugas selesai dalam {total_schedule_time:.4f} detik.")
    
    # 5. Simpan Hasil dan Hitung Metrik
    write_results_to_csv(results_list)
    calculate_and_print_metrics(results_list, vms, total_schedule_time)

if __name__ == "__main__":
    asyncio.run(main())
