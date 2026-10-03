"""A Random baseline that is random *per user*.

RecBole's ``Random.full_sort_predict`` draws ONE random score vector per batch and
repeats it for every user in the batch (``recbole/model/general_recommender/random.py``).
With the default ``eval_batch_size`` all 943 users are one batch, so every user would
receive the same random ranking (minus their history). That is a fixed random
permutation of the catalogue, not a random recommender, and it would make the
baseline's coverage / diversity numbers meaningless for Tasks 2 and 3.

This subclass draws an independent score vector per user from a generator seeded
with ``seed + user_id``: deterministic, independent of batching, and identical
between RecBole's own evaluation pass and our export.
"""
import torch

from recbole.model.general_recommender.random import Random


class RandomPerUser(Random):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._seed = int(config["seed"])

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        out = torch.empty(len(users), self.n_items, dtype=torch.float32)
        for row, uid in enumerate(users.tolist()):
            g = torch.Generator().manual_seed(self._seed * 1_000_003 + uid)
            out[row] = torch.rand(self.n_items, generator=g)
        return out.to(users.device).view(-1)
