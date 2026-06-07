"""Batch sampler to reduce padding waste in transformer models by considering sequence length."""

import math
import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class BucketingBatchSampler(Sampler[list[int]]):
    """Groups dataset indices into batches of similar sequence lengths.

    Greatly reduces padding waste for transformer models.

    Parameters
    ----------
    lengths : list[int]
        Lengths of each example (e.g., len(ag_emb[i])).
    batch_size : int
        Target batch size.
    shuffle : bool
        Whether to shuffle buckets between epochs.
    bucket_size : int
        Number of examples to sort together (controls randomness).
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size: int | None = None,
    ) -> None:
        """Initialize the BucketingBatchSampler.

        Parameters
        ----------
        lengths : list[int]
            Lengths of each example (e.g., sequence length).
        batch_size : int
            Target batch size.
        shuffle : bool, optional
            Whether to shuffle buckets between epochs, by default True.
        bucket_size : int, optional
            Number of examples to sort together (controls randomness). Defaults
            to ``batch_size * 10`` if not provided.
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size or batch_size * 10

    def __iter__(self) -> Iterator[list[int]]:
        """Yield mini-batches of indices.

        Yields
        ------
        list[int]
            A list of indices for each mini-batch.
        """
        # Step 1. Make index list
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        # Step 2. Sort within each bucket
        buckets = [
            sorted(
                indices[i : i + self.bucket_size],
                key=lambda x: self.lengths[x],
                reverse=True,
            )
            for i in range(0, len(indices), self.bucket_size)
        ]

        # Step 3. Yield mini-batches from each bucket
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    def __len__(self) -> int:
        """Return the number of mini-batches per epoch.

        Returns
        -------
        int
            Number of mini-batches per epoch.
        """
        return math.ceil(len(self.lengths) / self.batch_size)
