#!/usr/bin/env python3
# example of multinode multi-gpu training
# code adapted from LambdaLabsML

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import socket
import torch
import time  # Import time module for timing


try: 
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    LOCAL_RANK = shmem_comm.Get_rank()
    WORLD_SIZE = comm.Get_size()
    WORLD_RANK = comm.Get_rank()

    os.environ['MASTER_ADDR'] = comm.bcast( socket.gethostbyname( socket.gethostname() ), root=0 )
    os.environ['MASTER_PORT'] =	'1234'
except:

    print ("here!")
    if "LOCAL_RANK" in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # Environment variables set by mpirun
        LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
    else:
        print ("Error: No environment variables set for distributed training")
    #os.environ['MASTER_ADDR'] = comm.bcast( socket.gethostbyname( socket.gethostname() ), root=0 )
    os.environ['MASTER_PORT'] =	'1234'

if WORLD_RANK==0:
    print ('----------------------')
    print ('LOCAL_RANK  : ', LOCAL_RANK)
    print ('WORLD_SIZE  : ', WORLD_SIZE)
    print ('WORLD_RANK  : ', WORLD_RANK)
    print ('cuda device : ', torch.cuda.device_count())
    print ('pytorch version : ', torch.__version__)
    print ('nccl version : ', torch.cuda.nccl.version())
    print ('----------------------')   

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

def ddp_setup(backend):
    init_process_group(backend=backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = LOCAL_RANK
        self.global_rank = WORLD_RANK
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.epoch_times = []  # List to store the time for each epoch
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        start_time = time.time()  # Start timing the epoch

        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

        epoch_time = time.time() - start_time  # Calculate the epoch time
        self.epoch_times.append(epoch_time)  # Store the epoch time
        print(f"[GPU{self.global_rank}] Epoch {epoch} completed in {epoch_time:.2f} seconds.")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        
        # Summarize training time after all epochs
        if self.local_rank == 0:
            self._summarize_times()

    def _summarize_times(self):
        print("\nEpoch Training Time Summary:")
        for i, t in enumerate(self.epoch_times, 1):
            print(f"Epoch {i}: {t:.2f} seconds")
        total_time = sum(self.epoch_times)
        print(f"\nTotal Training Time: {total_time:.2f} seconds")
        print(f"Average Time per Epoch: {total_time/len(self.epoch_times):.2f} seconds")

def load_train_objs():
    train_set = MyTrainDataset(2000)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", backend: str = "nccl"):
    ddp_setup(backend)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    print (snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    
    num_epochs_default = 50
    batch_size_default = 32

    parser.add_argument(
        "--total_epochs", "--num_epochs",
        type=int,
        help="Total epochs to train the model",
        default=num_epochs_default,
        dest='total_epochs'
    )

    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"])

    parser.add_argument('--save_every',
        type=int,
        help='How often to save a snapshot',
        default=10,
        dest='save_every')

    parser.add_argument('--batch_size', 
        type=int,
        help='Input batch size on each device (default: 32)',
        default=batch_size_default,)

    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size, 'snapshot.pt', args.backend)
