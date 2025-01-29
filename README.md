# profiling

CLI for profiling using Nsight-Compute:

export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B
export DEVICE=cuda
huggingface-cli login

#Option 1:
##running Nsight Compute for gpt-fast framework (You can choose your own kernel range using --launch-skip (start point) and --launch-count):
ncu --call-stack --replay-mode kernel --set roofline --launch-skip 100 --launch-count 50 -o /home/ubuntu/projects/llama_project/parser_proj/batch_128 python generate.py --batch_size 128 --compile --compile_prefill --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is" --max_new_tokens 20 --device $DEVICE

#Option 2:
##running Nsight Compute for vllm framework:
ncu --call-stack --set roofline --launch-skip 100 --launch-count 50 -f -o <filename> python vllm_test.py

#Importing .ncu-rep file to a raw .csv file:
ncu --import file_name.ncu-rep --csv --section SpeedOfLight > raw_csv_output_path.csv

#Generating a readable format .csv file:
python organize.py

#Adding valuable metrics for utilization:
python calc_metrics.py output_test_3.csv output_test_vllm_3.csv


