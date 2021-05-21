import random

import numpy as np
import torch
from scipy.stats import entropy
from tqdm import tqdm


class Policy:
    def __init__(self, loader, warmup_epochs, n_new, model,
                 update_freq=1, seed=431):
        assert len(loader.keys()) == 2, 'will lead to problem with indexes'
        self.loader_warmup = loader['warmup']
        self.loader_pool = loader['pool']
        self.n_used = len(self.loader_warmup)
        self.n_new = n_new
        self.model = model
        self.cur_epoch = -warmup_epochs
        self.update_freq = update_freq
        self.last_updated_epoch = -update_freq
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self._points = list(self.loader_warmup)

    def __len__(self):
        return len(self._points)

    def _update_points(self):
        raise NotImplemented

        ep = self.cur_epoch
        n_used = self.n_used
        n_new = self.n_new
        return self.loader_pool

    def __call__(self):
        if self.cur_epoch - self.last_updated_epoch >= self.update_freq:
            self.last_updated_epoch = self.cur_epoch
            self._points = self._update_points()
            print('====== updating points')
            print(len(self._points))

        self.cur_epoch += 1
        return self._points


class RandomLearner(Policy):
    """ Policy that chooses random images per update
    """

    def __init__(self, loader, warmup_epochs, n_new, model,
                 update_freq=1, seed=431, *args, **kwargs):
        super().__init__(loader, warmup_epochs, n_new, model,
                         update_freq, seed)
        self.used_ids = set()  # ignoring warmup indexes

    def update_used_points(self, idxs: set):
        new_idxs = idxs - self.used_ids
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            triplets = sum([list(zip(*batch))
                            for batch in all_points], [])

            new_points = [tr for tr in tqdm(triplets) if tr[0].item() in new_idxs]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(
                x, batch_size) for x in col_wise]
            self.batches = list(zip(*col_batches))
        self._points += self.batches

    @staticmethod
    def _create_chunks(array, size):
        step = size
        return [torch.stack(array[i:i+step]) for i in range(0, len(array), step)]

    def _update_points(self):
        training = self.model.training
        self.model.train()
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            self.n_used += self.n_new
            triplets = sum([list(zip(batch[0],
                                     batch[1],
                                     batch[2],
                                     )
                                 )
                            for batch in all_points], [])
            triplets = [(tr[0],
                         tr[1],
                         tr[2],
                         )
                        for tr in tqdm(triplets) if tr[0] not in self.used_ids
                        ]
            new_points = triplets[-self.n_new*batch_size:]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x, col_wise[0])))

            col_batches = [self._create_chunks(x, batch_size) for x in col_wise]
            self.batches = list(zip(*col_batches))
            torch.cuda.empty_cache()
        self.model.train() if training else self.model.eval()
        return self._points + self.batches


class UncertainLearnerEntropy(Policy):
    """ Chooses new datapoints taking most uncertain ones 
    """

    def __init__(self, loader, warmup_epochs, n_new, model,
                 update_freq=1, trials=10, seed=431, 
                 device=torch.device('cpu'), *args, **kwargs):
        super().__init__(loader, warmup_epochs, n_new, model,
                         update_freq, seed)
        self.used_ids = set()  # ignoring warmup indexes
        self._device = device
        self._trials = trials

    def update_used_points(self, idxs: set):
        new_idxs = idxs - self.used_ids
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            triplets = sum([list(zip(*batch))
                            for batch in all_points], [])

            new_points = [tr for tr in tqdm(triplets) if tr[0].item() in new_idxs]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(
                x, batch_size) for x in col_wise]
            self.batches = list(zip(*col_batches))
        self._points += self.batches

    @staticmethod
    def _create_chunks(array, size):
        step = size
        return [torch.stack(array[i:i+step]) for i in range(0, len(array), step)]

    def _get_entropy(self, item):
        entr = np.mean([entropy(torch.flatten(self.model(torch.unsqueeze(item, 0).to(self._device)).cpu(), 1), axis=1)
                        for _ in range(self._trials)])
        return entr

    def _update_points(self):
        self.model.eval()
        training = self.model.training
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            pool_points = list(self.loader_pool)
            self.n_used += self.n_new
            triplets = sum([list(zip(batch[0],
                                     batch[1],
                                     batch[2],
                                     )
                                 )
                            for batch in pool_points], [])
            triplets = [(tr[0],
                         tr[1],
                         tr[2],
                         self._get_entropy(tr[1])
                         )
                        for tr in tqdm(triplets) if tr[0].item() not in self.used_ids]

            new_points = sorted(
                triplets, key=lambda x: x[3])[-self.n_new*batch_size:]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(x, batch_size) for x in col_wise[:-1]]
            self.batches = list(zip(*col_batches))
            torch.cuda.empty_cache()

        self.model.train() if training else self.model.eval()
        return self._points + self.batches


class VarianceLearnerDropout(Policy):
    def __init__(self, loader, warmup_epochs, n_new, model,
                 update_freq=1, seed=431, trials=10,
                 device=torch.device('cpu'), *args, **kwargs):
        super().__init__(loader, warmup_epochs, n_new, model,
                         update_freq, seed)
        self.used_ids = set()  # ignoring warmup indexes
        self._trials = trials
        self._device = device

    def update_used_points(self, idxs: set):
        new_idxs = idxs - self.used_ids
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            triplets = sum([list(zip(*batch))
                            for batch in all_points], [])

            new_points = [tr for tr in tqdm(triplets) if tr[0].item() in new_idxs]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(
                x, batch_size) for x in col_wise]
            self.batches = list(zip(*col_batches))
        self._points += self.batches

    @staticmethod
    def _create_chunks(array, size):
        step = size
        return [torch.stack(array[i:i+step]) for i in range(0, len(array), step)]

    def _get_variance(self, item):
        std = np.std([self.model(item).cpu().flatten().numpy()
                      for _ in range(self._trials)], axis=0).mean()
        return std

    def _update_points(self):
        training = self.model.training
        self.model.train()
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            self.n_used += self.n_new
            triplets = sum([list(zip(batch[0],
                                     batch[1],
                                     batch[2],
                                     )
                                 )
                            for batch in all_points], [])
            triplets = [(tr[0],
                         tr[1],
                         tr[2],
                         self._get_variance(
                torch.unsqueeze(tr[1], 0).to(self._device))
            )
                for tr in tqdm(triplets) if tr[0].item() not in self.used_ids
            ]
            new_points = sorted(
                triplets, key=lambda x: x[3])[-self.n_new*batch_size:]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(x, batch_size) for x in col_wise[:-1]]
            self.batches = list(zip(*col_batches))
            torch.cuda.empty_cache()
        self.model.train() if training else self.model.eval()
        return self._points + self.batches


class OurLearnerDeterministic(Policy):
    def __init__(self, loader, warmup_epochs, n_new, model, policy_model, feature_gen,
                 update_freq=1, seed=432, *args, **kwargs):
        super().__init__(loader, warmup_epochs, n_new, model, update_freq, seed)
        self.used_ids = set()
        self.policy_model = policy_model
        self.feature_gen = feature_gen
        # self._trials = trials

    def update_used_points(self, idxs: set):
        new_idxs = set([idx.item() for idx in idxs]) - self.used_ids
        if len(new_idxs) == 0:
            return
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            triplets = sum([list(zip(*batch))
                            for batch in all_points], [])
            new_points = [tr for tr in tqdm(triplets) if tr[0].item() in new_idxs]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return 
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(
                x, batch_size) for x in col_wise]
            self.batches = list(zip(*col_batches))
        self._points += self.batches

    @staticmethod
    def _create_chunks(array, size):
        step = size
        return [torch.stack(array[i:i+step]) for i in range(0, len(array), step)]

    def _get_improvement(self, item):
        scaler = self.policy_model['scaler']
        model = self.policy_model['clf']

        feature_map = self.feature_gen(item.detach().cpu()).numpy()
        feature_map = scaler.transform(feature_map.reshape(1, -1))
        # score = model.predict(feature_map) # if regressor
        score = model.predict_proba(feature_map)[:, -1] # if classifier with highest class prob in last place
        return score

    def _update_points(self):
        training = self.model.training
        self.model.train()
        with torch.no_grad():
            batch_size = self.loader_pool.batch_size
            all_points = list(self.loader_pool)
            self.n_used += self.n_new
            triplets = sum([list(zip(*batch))
                            for batch in all_points], [])
            triplets = [(tr[0],
                         tr[1],
                         tr[2],
                         self._get_improvement(torch.unsqueeze(tr[1], 0))
                         )
                        for tr in tqdm(triplets) if tr[0].item() not in self.used_ids
                        ]
            # print(triplets[0][3])
            new_points = sorted(
                triplets, key=lambda x: x[3])[-self.n_new*batch_size:]
            col_wise = list(zip(*new_points))
            if not col_wise:
                return self._points
            self.used_ids.update(set(map(lambda x: x.item(), col_wise[0])))

            col_batches = [self._create_chunks(
                x, batch_size) for x in col_wise[:-1]]
            self.batches = list(zip(*col_batches))
            torch.cuda.empty_cache()
        self.model.train() if training else self.model.eval()
        return self._points + self.batches
