"""A Pop baseline that ranks by the exact number of training interactions.

RecBole's ``Pop.calculate_loss`` counts with ``item_cnt[item, :] = item_cnt[item, :] + 1`` per
training batch (``recbole/model/general_recommender/pop.py``). Indexed assignment adds 1 only once
for an id that occurs several times in the batch, so ``item_cnt`` is the number of *batches* that
contain an item, not its number of interactions. Every popular film is in every batch, so the
counts saturate (11 distinct scores on ML-100K, 5,630 of the exported test rows tied at the
maximum) and ``torch.topk``'s tie-breaking decides which of roughly 100 films a user gets. That is
neither a popularity ranking nor stable between machines (test NDCG@10 0.100 / 0.103 in two runs,
0.142 with exact counts).

This subclass counts every interaction of the training split once, directly from the dataset the
model is built with, so the scores are exact, deterministic and independent of batching and epochs.
"""
import torch

from recbole.model.general_recommender.pop import Pop


class PopExact(Pop):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        items = dataset.inter_feat[dataset.iid_field].long()
        self.item_cnt = torch.bincount(items, minlength=self.n_items).view(-1, 1).to(self.device)
        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)  # counts are fixed at construction
