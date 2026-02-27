"""
Training configuration dataclass.

Provides a single place to define and parse training hyperparameters,
loadable from command-line arguments or YAML files.

Usage:
    from utils.config import TrainingConfig

    # From command-line args:
    config = TrainingConfig.from_args()

    # From YAML:
    config = TrainingConfig.from_yaml("config.yaml")

    # Direct construction:
    config = TrainingConfig(num_epochs=20, batch_size=64)
"""

import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TrainingConfig:
    """Standard training hyperparameters for distributed PyTorch."""

    # Training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    seed: int = 42

    # Distributed
    backend: str = "nccl"

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"

    # Checkpointing
    save_every: int = 0  # 0 = don't save
    checkpoint_dir: str = "./checkpoints"

    # Profiling
    profile: bool = False
    profile_dir: str = "./profiler_output"

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path):
        """
        Load config from a YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            TrainingConfig instance.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_args(cls, args=None):
        """
        Parse config from command-line arguments.

        Any field in the dataclass becomes a CLI flag (underscores to hyphens).

        Args:
            args: Optional list of args (defaults to sys.argv).

        Returns:
            TrainingConfig instance.
        """
        parser = argparse.ArgumentParser(description="Distributed Training Config")

        parser.add_argument("--num-epochs", type=int, default=cls.num_epochs)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--learning-rate", "--lr", type=float,
                            default=cls.learning_rate)
        parser.add_argument("--weight-decay", type=float, default=cls.weight_decay)
        parser.add_argument("--momentum", type=float, default=cls.momentum)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--backend", type=str, default=cls.backend,
                            choices=["nccl", "gloo", "mpi"])
        parser.add_argument("--use-amp", action="store_true", default=cls.use_amp)
        parser.add_argument("--amp-dtype", type=str, default=cls.amp_dtype,
                            choices=["bfloat16", "float16"])
        parser.add_argument("--save-every", type=int, default=cls.save_every)
        parser.add_argument("--checkpoint-dir", type=str,
                            default=cls.checkpoint_dir)
        parser.add_argument("--profile", action="store_true", default=cls.profile)
        parser.add_argument("--profile-dir", type=str, default=cls.profile_dir)

        parsed, _ = parser.parse_known_args(args)

        return cls(
            num_epochs=parsed.num_epochs,
            batch_size=parsed.batch_size,
            learning_rate=parsed.learning_rate,
            weight_decay=parsed.weight_decay,
            momentum=parsed.momentum,
            seed=parsed.seed,
            backend=parsed.backend,
            use_amp=parsed.use_amp,
            amp_dtype=parsed.amp_dtype,
            save_every=parsed.save_every,
            checkpoint_dir=parsed.checkpoint_dir,
            profile=parsed.profile,
            profile_dir=parsed.profile_dir,
        )
