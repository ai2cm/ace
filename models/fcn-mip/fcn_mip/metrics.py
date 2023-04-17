import torch


class RMSE:
    def __init__(self, weight=None):
        self._xy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xy = self._mean((truth - pred) ** 2)
        return xy.cpu()

    def gather(self, seq):
        return torch.sqrt(sum(seq) / len(seq))


class ACC:
    def __init__(self, mean, weight=None):
        self.mean = mean
        self._xy = {}
        self._xx = {}
        self._yy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xx = self._mean((truth - self.mean) ** 2).cpu()
        yy = self._mean((pred - self.mean) ** 2).cpu()
        xy = self._mean((pred - self.mean) * (truth - self.mean)).cpu()
        return xx, yy, xy

    def gather(self, seq):
        """seq is an iterable of (xx, yy, xy) tuples"""
        # transpose seq
        xx, yy, xy = zip(*seq)

        xx = sum(xx)
        xy = sum(xy)
        yy = sum(yy)
        return xy / torch.sqrt(xx) / torch.sqrt(yy)
