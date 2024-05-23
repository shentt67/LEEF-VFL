import torch

class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True,device = torch.device("cpu")):
        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).cuda()
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit_predict(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).cuda()
        init_points = x[init_row]
        self.centers = init_points
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()
        return self.labels

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().cuda()
        dists = torch.empty((0, self.n_clusters)).cuda()
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).cuda()
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        self.representative_samples = torch.argmin(self.dists, (0))