for running TensorRT tests - 

cd projects/TensorRT-LLM/
docker ps

To start a new docker:
sudo systemctl start docker
sudo systemctl enable docker
make -C docker release_run

To attach to an existing docker:
docker exec -it tensorrt_llm-release-Neta /bin/bash OR docker attach <docker_name>

Restart an existing docker:
docker restart tensorrt_llm-release-Neta

Running TRT-LLM benchmark:
      
      # Prepare the dataset:
      
      python benchmarks/cpp/prepare_dataset.py --tokenizer=meta-llama/Llama-3.1-8B-Instruct --stdout token-norm-dist --num-requests=30000 --input-mean=128 --output-mean=128 --input-stdev=0 --output-stdev=0 > 128_128.json 
      
      # Build the Engine:
      
      trtllm-bench --model meta-llama/Llama-3.1-8B-Instruct build --tp_size 1 --pp_size 1 --quantization FP8 --dataset 128_128.json 
      
      # Run the Engine:
      
      trtllm-bench --model meta-llama/Llama-3.1-8B-Instruct throughput --dataset 128_128.json --engine_dir /tmp/meta-llama/Llama-3.1-8B-Instruct/tp_1_pp_1 


Running TRT-LLM Inference (Llama-3.1-8B):

      cd /app/tensorrt_llm/examples/llama/
      git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
      huggingface-cli login --token *****
      
      1. Convert Checkpoints:
      
      python3 convert_checkpoint.py --model_dir Meta-Llama-3.1-8B-Instruct --output_dir llama-3.1-8b-ckpt
      
      2. uild Engine:
      
      trtllm-build --checkpoint_dir llama-3.1-8b-ckpt \
          --gemm_plugin float16 \
          --output_dir ./llama-3.1-8b-engine
      
      3. Run Inference:
      
      python3 ../run.py --engine_dir ./llama-3.1-8b-engine  --max_output_len 100 --tokenizer_dir Meta-Llama-3.1-8B-Instruct --input_text "How do I count to nine in French?"

Runnning NCU Profiling (TRT-LLM INference):

cd /usr/local/lib/python3.12/dist-packages/tensorrt_llm/runtime/

vim model_runner.py

Add imports: 

import torch
import torch.cuda.nvtx as nvtx

Add NVTX markers for profiling generation (decode):

print("===================Generate.decode was called!======================")
nvtx.range_push("Generate.decode START")
outputs = self.session.decode(
    batch_input_ids,
    input_lengths,
    sampling_config,
    stop_words_list=sampling_config.stop_words_list,
    bad_words_list=sampling_config.bad_words_list,
    output_sequence_lengths=sampling_config.output_sequence_lengths,
    output_generation_logits=output_generation_logits,
    return_dict=sampling_config.return_dict,
    streaming=streaming,
    stopping_criteria=stopping_criteria,
    logits_processor=logits_processor,
    position_ids=position_ids,
    encoder_output=encoder_input_features,
    encoder_input_lengths=encoder_output_lengths,
    cross_attention_mask=cross_attention_masks,
    **other_kwargs
)
nvtx.range_pop()

cd /app/tensorrt_llm/exmaples/llama/

run:

ncu --target-processes all --set full --export trtllm_decode_profile python3 ../run.py --engine_dir ./llama-3.1-8b-engine --max_output_len 100 --tokenizer_dir Meta-Llama-3.1-8B-Instruct --input_text "How do I count to nine in French?" --use_py_session

You should see:

==PROF== Connected to process 2671 (/usr/bin/python3.12)
[TensorRT-LLM] TensorRT-LLM version: 0.18.0.dev2025020400
[03/23/2025-17:30:17] [TRT-LLM] [I] Using Python session
[03/23/2025-17:30:24] [TRT-LLM] [W] Implicitly setting LLaMAConfig.fc_after_embed = False
[03/23/2025-17:30:24] [TRT-LLM] [W] Implicitly setting LLaMAConfig.use_input_layernorm_in_first_layer = True
[03/23/2025-17:30:24] [TRT-LLM] [W] Implicitly setting LLaMAConfig.use_last_layernorm = True
[03/23/2025-17:30:24] [TRT-LLM] [W] Implicitly setting LLaMAConfig.layer_idx_offset = 0
[03/23/2025-17:30:24] [TRT-LLM] [W] Implicitly setting LLaMAConfig.has_partial_lora_mask = False
[03/23/2025-17:30:24] [TRT-LLM] [W] Implicitly setting LLaMAConfig.tie_word_embeddings = False
[03/23/2025-17:30:24] [TRT-LLM] [I] Set dtype to bfloat16.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set bert_attention_plugin to auto.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set gpt_attention_plugin to auto.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set gemm_plugin to bfloat16.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set explicitly_disable_gemm_plugin to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set gemm_swiglu_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set fp8_rowwise_gemm_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set qserve_gemm_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set identity_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set nccl_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set lora_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set dora_plugin to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set weight_only_groupwise_quant_matmul_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set weight_only_quant_matmul_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set smooth_quant_plugins to True.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set smooth_quant_gemm_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set layernorm_quantization_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set rmsnorm_quantization_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set quantize_per_token_plugin to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set quantize_tensor_plugin to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set moe_plugin to auto.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set mamba_conv1d_plugin to auto.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set low_latency_gemm_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set low_latency_gemm_swiglu_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set gemm_allreduce_plugin to None.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set context_fmha to True.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set bert_context_fmha_fp32_acc to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set paged_kv_cache to True.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set remove_input_padding to True.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set reduce_fusion to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set user_buffer to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set tokens_per_block to 32.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set use_paged_context_fmha to True.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set use_fp8_context_fmha to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set fuse_fp4_quant to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set multiple_profiles to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set paged_state to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set streamingllm to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set manage_weights to False.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set use_fused_mlp to True.
[03/23/2025-17:30:24] [TRT-LLM] [I] Set pp_reduce_scatter to False.
[03/23/2025-17:30:24] [TRT] [I] Loaded engine size: 15388 MiB
[03/23/2025-17:30:26] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 15380 (MiB)
[03/23/2025-17:30:26] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 15380 (MiB)
[03/23/2025-17:30:26] [TRT-LLM] [W] The paged KV cache in Python runtime is experimental. For performance and correctness, please, use C++ runtime.
[03/23/2025-17:30:26] [TRT-LLM] [I] Load engine takes: 9.029225826263428 sec
==WARNING== Unable to access the following 6 metrics: ctc__rx_bytes_data_user.sum, ctc__rx_bytes_data_user.sum.pct_of_peak_sustained_elapsed, ctc__rx_bytes_data_user.sum.per_second, ctc__tx_bytes_data_user.sum, ctc__tx_bytes_data_user.sum.pct_of_peak_sustained_elapsed, ctc__tx_bytes_data_user.sum.per_second.


==PROF== Profiling "vectorized_elementwise_kernel" - 0: 0%....50%....100% - 39 passes
===================Generate.decode was called!======================
==PROF== Profiling "reduce_kernel" - 1: 0%....50%....100% - 39 passes
==PROF== Profiling "reduce_kernel" - 2: 0%....50%....100% - 39 passes
==PROF== Profiling "curandInitialize" - 3: 0%....50%....100% - 39 passes
==PROF== Profiling "setupTopKRuntimeArgs" - 4: 0%....50%....100% - 39 passes
==PROF== Profiling "setTopPRuntimeArgs" - 5: 0%....50%....100% - 39 passes
==PROF== Profiling "scatterDecodingParamsKernel" - 6: 0%....50%....100% - 39 passes
==PROF== Profiling "scatterDecodingParamsKernel" - 7: 0%....50%....100% - 39 passes
==PROF== Profiling "scatterDecodingParamsKernel" - 8: 0%....50%....100% - 39 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 9: 0%....50%....100% - 39 passes
/usr/local/lib/python3.12/dist-packages/torch/nested/__init__.py:228: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  return _nested.nested_tensor(
==PROF== Profiling "vectorized_elementwise_kernel" - 10: 0%....50%....100% - 39 passes
