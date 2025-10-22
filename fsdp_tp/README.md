# 1D Device Mesh Example (Toy Model)
mpiexec -n 8 --ppn 4 --cpu-bind none python tensor_parallel_example.py 

# 2D Parallelism (Tensor Parallelism + FSDP) Example (LLAMA2)
mpiexec -n 8 --ppn 4 --cpu-bind none python fsdp_tp_example.py

# To learn more: 
https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
