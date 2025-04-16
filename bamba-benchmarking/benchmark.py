import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Set constants ---
# MODEL_PATH = "./Bamba-9B-fp8"   #Cannot use this on cpu or T4 :/ 
MODEL_PATH = "sshleifer/tiny-gpt2" 
PROMPT_CSV = "bamba-benchmarking/prompts.csv" #data
MAX_NEW_TOKENS = 10

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = torch.device("cpu") #force it to use cpu (MPS error)
print(f"Using device: {device}")

# --- Load model and tokenizer ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=torch.float16,).to(device).eval()

# --- Load prompts  ---
df = pd.read_csv(PROMPT_CSV)
# Source: https://huggingface.co/datasets/fka/awesome-chatgpt-prompts
# Pulled in as a csv 
# Add prompt length as a feature
df["prompt_length"] = df["prompt"].apply(lambda x: len(tokenizer(x)["input_ids"]))
results = {}

# --- Benchmarking  ---
print("Running benchmark...")
for i, row in df.iterrows():
    prompt = row["prompt"]
    prompt_len = row["prompt_length"]

    #Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    #Timing model
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)
    end_time = time.time()

    #Get latency and throughput
    total_time = end_time - start_time
    generated_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    throughput = generated_tokens / total_time

    #Bandwidth (only applicable to GPU studd)
    if device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated()
        bandwidth = memory_used / total_time / 1e9  # in GB/s
    else:
        memory_used = 0
        bandwidth = 0
    
    #Store in dictionary for easy plotting, rounding 
    results[prompt_len] = {
        "latency": round(total_time, 4),
        "throughput": round(throughput, 2),
        "bandwidth": round(bandwidth, 2)
    }

    print(f"[{i+1}/{len(df)}] Length: {prompt_len}, Latency: {total_time:.4f}s, Throughput: {throughput:.2f} tok/s, Bandwidth: {bandwidth:.2f} GB/s")

# --- Save results in dataframe and to csv ---
results_df = pd.DataFrame([
    {"prompt_length": k, **v} for k, v in results.items()
])
results_df.to_csv("bamba_benchmark_results.csv", index=False)

# --- Plot results ---
plt.figure()
plt.scatter(results_df["prompt_length"], results_df["latency"], marker="o")
plt.title("Latency vs Prompt Length")
plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Latency (s)")
plt.grid(True)
plt.savefig("latency_vs_prompt_length.png")

plt.figure()
plt.scatter(results_df["prompt_length"], results_df["throughput"], marker="o")
plt.title("Throughput vs Prompt Length")
plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Throughput (tokens/sec)")
plt.grid(True)
plt.savefig("throughput_vs_prompt_length.png")

plt.figure()
plt.scatter(results_df["prompt_length"], results_df["bandwidth"], marker="o")
plt.title("Bandwidth vs Prompt Length")
plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Bandwidth (GB/s)")
plt.grid(True)
plt.savefig("bandwidth_vs_prompt_length.png")
print("Results saved")
