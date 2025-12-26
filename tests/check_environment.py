#!/usr/bin/env python3
"""
Comprehensive environment check for distributed PyTorch training on Derecho.

Supports multiple launchers:
    - torchrun: torchrun --standalone --nproc_per_node=4 check_environment.py
    - mpiexec:  mpiexec -n 8 --ppn 4 --cpu-bind none python check_environment.py
    - single:   python check_environment.py
"""

import os
import sys
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════════
# Rank Detection (supports torchrun, mpirun, cray-mpich, mpi4py)
# ═══════════════════════════════════════════════════════════════════════════

def get_rank_info():
    """
    Detect LOCAL_RANK, WORLD_SIZE, WORLD_RANK from various launchers.
    
    Returns:
        tuple: (local_rank, world_size, world_rank, method)
    """
    # Method 1: torchrun / torch.distributed.launch (check FIRST!)
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        return (
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ["RANK"]),
            "torchrun"
        )
    
    # Method 2: OpenMPI mpirun
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        
        # Set MASTER_ADDR for NCCL if not already set
        _set_master_addr_from_mpi(world_rank)
        
        return local_rank, world_size, world_rank, "openmpi"
    
    # Method 3: Cray MPICH (PMI)
    if "PMI_RANK" in os.environ:
        local_rank = int(os.environ.get("PMI_LOCAL_RANK", 0))
        world_size = int(os.environ["PMI_SIZE"])
        world_rank = int(os.environ["PMI_RANK"])
        
        # Set MASTER_ADDR for NCCL if not already set
        _set_master_addr_from_mpi(world_rank)
        
        return local_rank, world_size, world_rank, "cray-mpich"
    
    # Method 4: mpi4py (fallback for mpiexec without env vars)
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        
        if world_size > 1:
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_rank = shmem_comm.Get_rank()
            world_rank = comm.Get_rank()
            
            # Use MPI broadcast for MASTER_ADDR (more reliable)
            master_addr = comm.bcast(socket.gethostbyname(socket.gethostname()), root=0)
            os.environ.setdefault('MASTER_ADDR', master_addr)
            os.environ.setdefault('MASTER_PORT', '29500')
            
            return local_rank, world_size, world_rank, "mpi4py"
    except ImportError:
        pass
    except Exception:
        pass
    
    # Method 5: Single process (no distributed)
    return 0, 1, 0, "single"

def _set_master_addr_from_mpi(world_rank):
    """Set MASTER_ADDR and MASTER_PORT for MPI-based launchers."""
    if 'MASTER_ADDR' not in os.environ:
        # Try to use mpi4py for coordinated broadcast
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            if world_rank == 0:
                master_addr = socket.gethostbyname(socket.gethostname())
            else:
                master_addr = None
            master_addr = comm.bcast(master_addr, root=0)
            os.environ['MASTER_ADDR'] = master_addr
        except ImportError:
            # Fallback: just use current hostname (works for single-node)
            os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
    
    os.environ.setdefault('MASTER_PORT', '29500')

# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_nccl_library_path():
    """Find the NCCL library path."""
    try:
        import ctypes.util
        nccl_path = ctypes.util.find_library('nccl')
        if nccl_path:
            return nccl_path
        
        common_paths = [
            Path(torch.__file__).parent / "lib" / "libnccl.so",
            Path(sys.prefix) / "lib" / "libnccl.so",
        ]
        
        ld_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
        for ld_path in ld_paths:
            if ld_path:
                common_paths.append(Path(ld_path) / "libnccl.so")
        
        for path in common_paths:
            if path.exists():
                return str(path)
            for versioned in path.parent.glob("libnccl.so*"):
                if versioned.exists():
                    return str(versioned)
        
        return "Not found"
    except Exception as e:
        return f"Error: {e}"


def get_nccl_version():
    """Get NCCL version tuple."""
    try:
        return torch.cuda.nccl.version()
    except Exception:
        return None


def format_nccl_version(version):
    """Format NCCL version tuple as string."""
    if isinstance(version, tuple):
        return f"{version[0]}.{version[1]}.{version[2]}"
    return str(version) if version else "N/A"


def setup_distributed(local_rank, world_size, world_rank):
    """Initialize distributed process group."""
    if world_size <= 1:
        return False
    
    if dist.is_initialized():
        return True
    
    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=world_rank,
            device_id=device,
        )
        return True
    except TypeError:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=world_rank,
        )
        return True
    except Exception as e:
        if world_rank == 0:
            print(f"  Warning: Could not initialize distributed: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed process group gracefully."""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_hostnames(world_rank, world_size):
    """Gather hostnames from all ranks to rank 0."""
    hostname = socket.gethostname()
    
    if not dist.is_initialized():
        return [hostname]
    
    hostname_list = [None] * world_size
    dist.all_gather_object(hostname_list, hostname)
    
    return hostname_list


def print_header(title, char="=", width=70):
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_section(title):
    print(f"\n{'─' * 70}")
    print(f" {title}")
    print('─' * 70)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    
    # Get rank info
    local_rank, world_size, world_rank, launch_method = get_rank_info()

    # Setup distributed
    is_distributed = setup_distributed(local_rank, world_size, world_rank)
    
    # Gather info from all ranks
    gpus_per_node = torch.cuda.device_count()
    
    if is_distributed:
        hostnames = gather_hostnames(world_rank, world_size)
        unique_hosts = sorted(set(hostnames))
        num_nodes = len(unique_hosts)
    else:
        hostnames = [socket.gethostname()]
        unique_hosts = hostnames
        num_nodes = 1
    
    total_gpus = num_nodes * gpus_per_node
    
    # Only rank 0 prints
    if world_rank != 0:
        if is_distributed:
            dist.barrier()
            time.sleep(0.5)
            cleanup_distributed()
        return
    
    print_header("DISTRIBUTED PyTorch ENVIRONMENT CHECK")
    
    # ─────────────────────────────────────────────────────────────────
    # Distributed Configuration (consolidated)
    # ─────────────────────────────────────────────────────────────────
    print_section("Distributed Configuration")
    print(f"  Launch Method    : {launch_method}")
    print(f"  Local Rank       : {local_rank}")
    print(f"  World Rank       : {world_rank}")
    print(f"  World Size       : {world_size}")
    print(f"  Num Nodes        : {num_nodes}")
    print(f"  GPUs per Node    : {gpus_per_node}")
    print(f"  Total GPUs       : {total_gpus}")
    print(f"  Master Address   : {os.environ.get('MASTER_ADDR', 'N/A')}")
    print(f"  Master Port      : {os.environ.get('MASTER_PORT', 'N/A')}")
    
    if num_nodes > 1:
        print(f"  Nodes            : {', '.join(unique_hosts)}")
    
    # ─────────────────────────────────────────────────────────────────
    # Rank → Node Mapping
    # ─────────────────────────────────────────────────────────────────
    if is_distributed and world_size > 1:
        print_section("Rank → Node Mapping")
        for rank, host in enumerate(hostnames):
            gpu_id = rank % gpus_per_node
            print(f"  Rank {rank:2d} → {host} (GPU {gpu_id})")
    
    # ─────────────────────────────────────────────────────────────────
    # PyTorch & CUDA
    # ─────────────────────────────────────────────────────────────────
    print_section("PyTorch & CUDA")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  CUDA             : {torch.version.cuda}")
    print(f"  cuDNN            : {torch.backends.cudnn.version()}")
    
    # ─────────────────────────────────────────────────────────────────
    # NCCL Configuration
    # ─────────────────────────────────────────────────────────────────
    print_section("NCCL")
    print(f"  Version          : {format_nccl_version(get_nccl_version())}")
    print(f"  Library          : {get_nccl_library_path()}")
    
    # Only show NCCL env vars that are set
    nccl_env_vars = ['NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'NCCL_NET_GDR_LEVEL', 
                     'NCCL_CROSS_NIC', 'NCCL_TIMEOUT', 'NCCL_DEBUG']
    set_vars = {var: os.environ.get(var) for var in nccl_env_vars if os.environ.get(var)}
    if set_vars:
        print(f"  Environment:")
        for var, value in set_vars.items():
            print(f"    {var}: {value}")
    
    # ─────────────────────────────────────────────────────────────────
    # GPU Information
    # ─────────────────────────────────────────────────────────────────
    print_section("GPUs")
    if torch.cuda.device_count() == 0:
        print("  No GPUs detected")
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"  [{i}] {props.name} | {mem_gb:.0f} GB | SM {props.major}.{props.minor}")
    
    # ─────────────────────────────────────────────────────────────────
    # Network Interfaces
    # ─────────────────────────────────────────────────────────────────
    print_section("Network Interfaces (Slingshot)")
    try:
        result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
        interfaces = result.stdout
        found = [iface for iface in ['hsn0', 'hsn1', 'hsn2', 'hsn3'] if iface in interfaces]
        missing = [iface for iface in ['hsn0', 'hsn1', 'hsn2', 'hsn3'] if iface not in interfaces]
        print(f"  Found: {', '.join(found) if found else 'none'}")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
    except Exception as e:
        print(f"  Could not check: {e}")
    
    # ─────────────────────────────────────────────────────────────────
    # PyTorch Build Info (condensed)
    # ─────────────────────────────────────────────────────────────────
    print_section("PyTorch Build")
    print(torch.__config__.show())
    
    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    print("\n")
    print_header("SUMMARY", "═")
    
    checks = [
        ("PyTorch", True),
        ("CUDA", torch.cuda.is_available()),
        ("GPUs", torch.cuda.device_count() > 0),
        ("NCCL", torch.distributed.is_nccl_available()),
        ("Distributed", is_distributed or world_size == 1),
    ]
    
    all_passed = all(passed for _, passed in checks)
    
    status_line = "  " + " | ".join(f"{'✓' if p else '✗'} {n}" for n, p in checks)
    print(status_line)
    
    print()
    if all_passed:
        msg = f"✓ Ready: {total_gpus} GPUs across {num_nodes} node(s)"
        print(f"  ╔{'═' * (len(msg) + 2)}╗")
        print(f"  ║ {msg} ║")
        print(f"  ╚{'═' * (len(msg) + 2)}╝")
    else:
        print("  ╔════════════════════════════════════════╗")
        print("  ║  ✗ SOME CHECKS FAILED                  ║")
        print("  ╚════════════════════════════════════════╝")
    
    print("=" * 70)
    sys.stdout.flush()
    
    # Cleanup
    if is_distributed:
        dist.barrier()
        time.sleep(0.5)
        cleanup_distributed()


if __name__ == "__main__":
    main()