import csv
import argparse
import time
from vllm import LLM, SamplingParams

# Argument Parser
parser = argparse.ArgumentParser(description="Benchmark LLaMA Model SLA with vLLM")
parser.add_argument("--model", type=str, choices=["8B", "70B"], required=True, help="Choose model size: 8B or 70B")
parser.add_argument("--batch_min", type=int, required=True, help="Minimum batch size (power of 2)")
parser.add_argument("--batch_max", type=int, required=True, help="Maximum batch size (power of 2)")
args = parser.parse_args()

# Model selection
model_mapping = {
    "8B": "meta-llama/Meta-Llama-3.1-8B",
    "70B": "meta-llama/Meta-Llama-3.1-70B"
}
model_name = model_mapping[args.model]

# Initialize vLLM
llm = LLM(model=model_name, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)

# Generate batch sizes in powers of 2
batch_sizes = [2**i for i in range(args.batch_min.bit_length() - 1, args.batch_max.bit_length())]

# CSV Setup
csv_filename = f"sla_results_{args.model}.csv"
csv_headers = ["Model Size", "Batch Size", "Measured SLA (tokens/sec/user)"]

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

    # Run inference for each batch size
    for batch_size in batch_sizes:
        input_text = "Hello world! " * (1000 // 2)  # Approx. 1000 tokens
        prompts = [input_text] * batch_size

        start_time = time.time()  # Start time measurement
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()  # End time measurement

        # Calculate tokens/sec manually
        total_time = end_time - start_time
        total_tokens = batch_size * 1000  # Batch size * max_tokens per prompt
        tokens_per_sec = total_tokens / total_time
        sla = tokens_per_sec / batch_size  # Tokens/sec per user

        # Save to CSV
        writer.writerow([args.model, batch_size, sla])

        print(f"Model: {args.model}, Batch Size: {batch_size}, SLA: {sla:.2f} tokens/sec/user")

print(f"Results saved to {csv_filename}.")
