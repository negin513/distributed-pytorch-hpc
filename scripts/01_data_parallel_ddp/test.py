import torch
from torch.utils.data import Dataset, DataLoader
import time


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(512, 1)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        print("loading idx {} from worker {}".format(
            idx, worker_info.id))
        x = self.data[idx]
        return x

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = MyDataset()
    loader = DataLoader(dataset, num_workers=8, batch_size=2)

    print("=" * 60)
    print("Creating iterator (preloading starts here)")
    print("=" * 60)
    loader_iter = iter(loader)

    # Give workers time to finish prefetching before we consume
    time.sleep(1.0)

    print("=" * 60)
    print("With default prefetch_factor=2, expect 2*num_workers=16 batches preloaded")
    print("=> max idx printed should be 31 (16 batches * batch_size=2 = 32 samples)")
    print("=" * 60)

    for i in range(5):
        print("\n" + "=" * 60)
        print(f"next() call {i+1} -- consumes 1 batch, triggers 1 new prefetch")
        print("=" * 60)
        data = next(loader_iter)

        # Give the worker time to refill
        time.sleep(0.5)

        print(f"  => consumed batch shape: {data.shape}")
        print(f"  => batch values (flattened): {data.flatten().tolist()}")
