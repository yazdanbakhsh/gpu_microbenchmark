python3 gpu_mbm.py --m 64 --n 12288 --k 49152 --iter 20 --warmup 5 >> out-gemm-64.txt
python3 gpu_mbm.py --m 512 --n 12288 --k 49152 --iter 20 --warmup 5 >> out-gemm-512.txt
python3 gpu_mbm.py --m 4096 --n 12288 --k 49152 --iter 20 --warmup 5 >> out-gemm-4096.txt
python3 gpu_mbm.py --m 36864 --n 12288 --k 49152 --iter 20 --warmup 5 >> out-gemm-36864.txt
python3 gpu_mbm.py --bmm --bsz 96 --m 1 --n 128 --k 64 --iter 20 --warmup 5 >> out-bmm-1-64.txt
python3 gpu_mbm.py --bmm --bsz 96 --m 1 --n 128 --k 256 --iter 20 --warmup 5 >> out-bmm-1-256.txt
python3 gpu_mbm.py --bmm --bsz 96 --m 1 --n 128 --k 1024 --iter 20 --warmup 5 >> out-bmm-1-1024.txt
python3 gpu_mbm.py --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 20 --warmup 5 >> out-bmm-32-64.txt
python3 gpu_mbm.py --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 20 --warmup 5 >> out-bmm-32-256.txt
python3 gpu_mbm.py --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 20 --warmup 5 >> out-bmm-32-1024.txt
