pip install -r requirements.txt

pip install --upgrade --force-reinstall nvidia-cublas-cu12

pip install --upgrade --force-reinstall nvidia-cudnn-cu12


**Steps for enabling access to performance counters on NVIDIA GPUs:**

1. Check Your Current GPU Profiling Permission Status:
   
   cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
   
If it returns RmProfilingAdminOnly=1, it means profiling is restricted to admin users.

3. Enable GPU Performance Counters Permanently:
   
   echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-prof.conf

4. Update the Initramfs:
   
   sudo update-initramfs -u -k all

6. sudo reboot
   
7. Check again:
   
  cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly

You supposed to see RmProfilingAdminOnly=0, which means that you have permissions for the GPU counters.

#profiling

**CLI for profiling using Nsight-Compute:**

export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B

export DEVICE=cuda

huggingface-cli login

**#Option 1:**

##running Nsight Compute for gpt-fast framework (You can choose your own kernel range using --launch-skip (start point) and --launch-count):

ncu --call-stack --replay-mode kernel --set roofline --launch-skip 100 --launch-count 50 -o /home/ubuntu/projects/llama_project/parser_proj/batch_128 python generate.py --batch_size 128 --compile --compile_prefill --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is" --max_new_tokens 20 --device $DEVICE

**#Option 2:**

##running Nsight Compute for vllm framework:

ncu --metrics smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fp8_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fp8_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_fp8_hfma_pred_on.sum,gpu__time_duration --call-stack --launch-skip 0 --launch-count 4000 --replay-mode kernel -f --export smsp_timing python vllm_test.py

#Importing .ncu-rep file to a raw .csv file:

ncu --import file_name.ncu-rep --csv > raw_data.csv

#Generating a readable format .csv file:

python organize.py

#Adding valuable metrics for utilization:

python calc_metrics.py output.csv output1.csv


