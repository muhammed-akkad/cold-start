import json
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Rename "S" => "S (Safetensors)"
# ---------------------------------------------------------------------
STRATEGY_RENAME = {
    "S": "S (Safetensors)",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D"
}

def load_results_json(path, model_name_hint, strategy_hint):
    """
    path: path to the JSON file
    model_name_hint: e.g. "VGG19", "ResNet50", "MobileNetV2"
    strategy_hint: one of "S", "A", "B", "C", "D"
    Returns a pandas DataFrame
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Flatten the "profiling" keys if they exist
    for entry in data:
        profiling = entry.get("profiling", {})
        entry["read_index_sec"] = profiling.get("read_index_sec", None)
        entry["instantiate_model_sec"] = profiling.get("instantiate_model_sec", None)
        entry["load_params_sec"] = profiling.get("load_params_sec", None)
        entry["assign_params_sec"] = profiling.get("assign_params_sec", None)
        entry["dispatch_model_sec"] = profiling.get("dispatch_model_sec", None)
        entry["profiling_total_sec"] = profiling.get("total_sec", None)

        # Rename S => S (Safetensors) in the final DataFrame
        renamed_strat = STRATEGY_RENAME.get(strategy_hint, strategy_hint)
        entry["model"] = model_name_hint
        entry["strategy"] = renamed_strat

    return pd.DataFrame(data)

# ---------------------------------------------------------------------
# Files: VGG19, ResNet50, and MobileNetV2
# ---------------------------------------------------------------------
FILES = [
    # VGG19
    ("results_VGG19_S.json",       "VGG19",       "S"),
    ("results_VGG19_A.json",       "VGG19",       "A"),
    ("results_VGG19_B.json",       "VGG19",       "B"),
    ("results_VGG19_C.json",       "VGG19",       "C"),
    ("results_VGG19_D.json",       "VGG19",       "D"),
    
    # ResNet50
    ("results_resnet50_S.json",    "ResNet50",    "S"),
    ("results_resnet50_A.json",    "ResNet50",    "A"),
    ("results_resnet50_B.json",    "ResNet50",    "B"),
    ("results_resnet50_C.json",    "ResNet50",    "C"),
    ("results_resnet50_D.json",    "ResNet50",    "D"),
    
    # MobileNetV2
    ("results_mobilenet_v2_S.json","MobileNetV2", "S"),
    ("results_mobilenet_v2_A.json","MobileNetV2", "A"),
    ("results_mobilenet_v2_B.json","MobileNetV2", "B"),
    ("results_mobilenet_v2_C.json","MobileNetV2", "C"),
    ("results_mobilenet_v2_D.json","MobileNetV2", "D"),
]

# ---------------------------------------------------------------------
# 1) Load all JSON data into one big DataFrame
# ---------------------------------------------------------------------
df_list = []
for (filename, model_hint, strategy_hint) in FILES:
    df_temp = load_results_json(filename, model_hint, strategy_hint)
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)

# ---------------------------------------------------------------------
# 2) Plot: Mean loading_time_sec by (model, strategy)
# ---------------------------------------------------------------------
grouped_loading = df.groupby(["model", "strategy"])["loading_time_sec"].mean().reset_index()
pivot_loading = grouped_loading.pivot(index="model", columns="strategy", values="loading_time_sec")

plt.figure(figsize=(6,4))
pivot_loading.plot(kind="bar")
plt.title("Mean loading_time_sec by Model and Strategy")
plt.ylabel("Average loading_time_sec")
plt.xlabel("Model")
plt.legend(title="Strategy")
plt.tight_layout()
plt.savefig("plot_loading_time_sec.png")
plt.close()

# ---------------------------------------------------------------------
# 3) Plot: Mean inference_time_sec by (model, strategy)
# ---------------------------------------------------------------------
grouped_inference = df.groupby(["model", "strategy"])["inference_time_sec"].mean().reset_index()
pivot_inference = grouped_inference.pivot(index="model", columns="strategy", values="inference_time_sec")

plt.figure(figsize=(6,4))
pivot_inference.plot(kind="bar")
plt.title("Mean inference_time_sec by Model and Strategy")
plt.ylabel("Average inference_time_sec")
plt.xlabel("Model")
plt.legend(title="Strategy")
plt.tight_layout()
plt.savefig("plot_inference_time_sec.png")
plt.close()

# ---------------------------------------------------------------------
# 4) Plot: Mean throughput by (model, strategy)
# ---------------------------------------------------------------------
grouped_throughput = df.groupby(["model", "strategy"])["throughput"].mean().reset_index()
pivot_throughput = grouped_throughput.pivot(index="model", columns="strategy", values="throughput")

plt.figure(figsize=(6,4))
pivot_throughput.plot(kind="bar")
plt.title("Mean throughput by Model and Strategy")
plt.ylabel("Average throughput (samples/sec)")
plt.xlabel("Model")
plt.legend(title="Strategy")
plt.tight_layout()
plt.savefig("plot_throughput.png")
plt.close()

# ---------------------------------------------------------------------
# 5) Plot: Stacked bar of profiling sub-steps for each model
# ---------------------------------------------------------------------
profiling_cols = [
    "read_index_sec",
    "instantiate_model_sec",
    "load_params_sec",
    "assign_params_sec",
    "dispatch_model_sec"
]
df_profiling = df.dropna(subset=profiling_cols, how="all")
profiling_means = df_profiling.groupby(["model", "strategy"])[profiling_cols].mean()

models_in_data = profiling_means.index.get_level_values("model").unique()
for model_name in models_in_data:
    sub_means = profiling_means.loc[model_name]  # slice out that model
    plt.figure(figsize=(6,4))
    sub_means.plot(kind="bar", stacked=True)
    plt.title(f"Profiling Breakdown: {model_name}")
    plt.ylabel("Time (sec)")
    plt.xlabel("Strategy")
    plt.tight_layout()
    outname = f"profiling_breakdown_{model_name}.png"
    plt.savefig(outname)
    plt.close()

# ---------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------
print("All done! Saved figures:")
print("  plot_loading_time_sec.png")
print("  plot_inference_time_sec.png")
print("  plot_throughput.png")
for model_name in models_in_data:
    print(f"  profiling_breakdown_{model_name}.png")
