import time
from vllm import LLM, SamplingParams

log_file_name = "./nsys.txt"

batch_size = 32
num_words = 32
prompts = [
    "The main challenges of ML and AI are" * num_words
] * batch_size  # Adjust batch size as needed

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
llm = LLM(model="meta-llama/Meta-Llama-3-8B", max_model_len=350 ,compilation_config=3)

# Start measuring time
t0 = time.time()

# Run LLM inference
outputs = llm.generate(prompts, sampling_params)
num_tokens = sum([len(output.outputs[0].text.split(" ")) - num_words * 0 for output in outputs]) / 0.75

tout = time.time()
tokens_per_sec = num_tokens / (tout - t0)

# Save log
log_content = (f"Num Tokens Generated: {num_tokens:.2f}, "
               f"Time {1000.0*(tout-t0):.3f} ms, "
               f"Input length {len(prompts[0])}, Batch {batch_size}, "
               f"TP {tokens_per_sec} [tokens/sec]\n")

print(f"Writing log to {log_file_name}")
with open(log_file_name, "w") as f:
    f.write(log_content)

print(f"\nLog saved to: {log_file_name}")
