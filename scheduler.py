from apscheduler.schedulers.background import BackgroundScheduler
import time
import psutil
import torch

def decide_allocation_for_model(
    model_name: str,
    usage_counts: dict,
    total_gpu_budget: float = 70.0,   # total % of model size we can put on GPU if usage is max
    total_cpu_budget: float = 50.0,   # total % of model size we can put on CPU if usage is max
    min_gpu_percent: float = 5.0,
    max_gpu_percent: float = 80.0,
    min_cpu_percent: float = 5.0,
    max_cpu_percent: float = 80.0,
) -> (float, float):
    """
    Decide gpu_percent & cpu_percent for `model_name` by comparing its usage 
    to other models, plus checking available GPU & CPU memory.
    """
    total_usage = sum(usage_counts.values())
    if total_usage == 0:
        # If no usage at all, fallback to minimal GPU usage
        return min_gpu_percent, min_cpu_percent
    
    # Model usage
    model_usage = usage_counts.get(model_name, 0)
    
    # Fraction of total usage
    usage_fraction = model_usage / total_usage  # e.g. if model_usage=50, total=200 => 0.25

    # Proposed GPU usage
    proposed_gpu = usage_fraction * total_gpu_budget  # e.g. 0.25 * 70 => 17.5
    gpu_percent = max(min_gpu_percent, min(proposed_gpu, max_gpu_percent))

    # Proposed CPU usage
    proposed_cpu = usage_fraction * total_cpu_budget
    cpu_percent = max(min_cpu_percent, min(proposed_cpu, max_cpu_percent))

    # Now let's consider available GPU memory
    if torch.cuda.is_available():
        free_gpu_mem, total_gpu_mem = torch.cuda.mem_get_info(device=0)
        free_gpu_gb = free_gpu_mem / (1024**3)
    else:
        free_gpu_gb = 0

    # If GPU memory is quite low, reduce GPU usage by some chunk
    if free_gpu_gb < 2:
        gpu_percent = max(0, gpu_percent - 10)

    # Check CPU memory
    mem_info = psutil.virtual_memory()
    free_cpu_gb = mem_info.available / (1024**3)
    # If CPU memory is low (<2 GB for example), reduce CPU usage
    if free_cpu_gb < 2:
        cpu_percent = max(0, cpu_percent - 10)

    # Ensure the sum doesn't exceed 100
    total_percent = gpu_percent + cpu_percent
    if total_percent > 100:
        # We'll reduce CPU to keep total under 100, 
        # but you could also reduce GPU or do something more advanced
        leftover = 100 - gpu_percent
        cpu_percent = min(cpu_percent, leftover)

    return gpu_percent, cpu_percent

def scheduler_update_allocations(all_model_names):
    """
    Example function that runs in a background job:
    1. Reads usage counts for all known models.
    2. For each model, decides how much should be on GPU/CPU.
    3. Re-saves the partial distribution if it changed significantly.
    """
    usage_counts = load_usage_counts()  # e.g., from a JSON or DB
    total_usage = sum(usage_counts.values())

    for model_name in all_model_names:
        # 1) Decide new gpu_percent, cpu_percent
        gpu_percent, cpu_percent = decide_allocation_for_model(model_name, usage_counts)
        
        # 2) Compare to the last known distribution (optional) to avoid thrashing
        old_gpu, old_cpu = get_last_distribution_for_model(model_name)
        if (abs(gpu_percent - old_gpu) < 5) and (abs(cpu_percent - old_cpu) < 5):
            print(f"No major change for {model_name}. Skipping re-save.")
            continue
        
        # 3) Actually load the model from HF or from disk, then re-save
        model = load_full_model_from_hub_or_disk(model_name)
        model_path = f"./shards_{model_name}"
        save_model(model, model_path, gpu_percent, cpu_percent)

        # 4) Store the new distribution
        store_distribution_for_model(model_name, gpu_percent, cpu_percent)

# Create a scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(scheduler_update_allocations, 'interval', hours=1)
scheduler.start()

# Keep script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    scheduler.shutdown()
