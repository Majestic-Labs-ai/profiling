from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is"
    ]*128 #adjust batch_size as required
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
llm = LLM(model="meta-llama/Meta-Llama-3.1-8B", max_model_len=2048)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
