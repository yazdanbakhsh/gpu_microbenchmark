import torch
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Matrix multiplication with optional settings.")
parser.add_argument('--bmm', action='store_true', help="Use batched matrix multiplication.")
parser.add_argument('--bsz', type=int, default=5040, help="Batch size (only used if --bmm is set).")
parser.add_argument('--m', type=int, default=2048, help="Number of rows of the first matrix.")
parser.add_argument('--n', type=int, default=2048, help="Number of columns of the first matrix and rows of the second matrix.")
parser.add_argument('--k', type=int, default=2048, help="Number of columns of the second matrix.")
parser.add_argument('--iter', type=int, default=5, help="Number of iterations to repeat the process.")
parser.add_argument('--warmup', type=int, default=2, help="Number of iterations to repeat the process.")
parser.add_argument('--dtype', type=str, default='bfloat16', help="Data type of the matrix multiplication.")
args = parser.parse_args()

# Set variables based on arguments
bmm = args.bmm
bsz = args.bsz if bmm else 1
m = args.m
n = args.n
k = args.k
iterations = args.iter
warmup = args.warmup
dtype = args.dtype

data_type = torch.float16 if dtype == 'bfloat16' else torch.float16

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)   

# Repeat the time measurement process
total_duration = 0
count = 0
for i in range(iterations):
    # Generate random tensors
    if bmm:
        a = torch.ones((bsz, m, n), dtype=data_type, device='cuda')
        b = torch.ones((bsz, n, k), dtype=data_type, device='cuda')
    else:
        a = torch.rand((m, n), dtype=data_type, device='cuda')
        b = torch.rand((n, k), dtype=data_type, device='cuda')

    # Measure time
    if bmm:
        start_event.record()
        c = torch.bmm(a, b)
        end_event.record()
    else:
        start_event.record()
        c = torch.matmul(a, b)
        end_event.record()
    torch.cuda.synchronize()  # Wait for all operations to finish
    duration = start_event.elapsed_time(end_event)  # Time in milliseconds
    # Accumulate duration
    duration /= 1000
    if i > warmup - 1:
        total_duration += duration

        print(f"Iteration {i - warmup}: output data type: {c.dtype}, Duration: {duration:.6f} seconds")
    del a, b
    c = c + 1
    torch.cuda.empty_cache()

# Calculate average duration and GFLOPS
average_duration = total_duration / (iterations-warmup)
n_comp = 2 * bsz * m * n * k if bmm else 2 * m * n * k
gflops = n_comp / average_duration / 10**9

print(f"Average Duration: {average_duration:.6f} seconds")
print(f"Throughput: {gflops} (GFLOPS)")
