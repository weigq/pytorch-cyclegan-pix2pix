
import torch.utils.data

__all__ = ['create_dataloader']


def create_dataset(opt):
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset {} was created".format(dataset.name()))

    dataset.initialize(opt)
    return dataset


class DataLoader:
    def __init__(self, opt):
        self.dataloader = None
        self.dataset = None
        self.opt = opt

    def initialize(self):
        self.dataset = create_dataset(self.opt)
        shuffle = True if self.opt.isTrain else False
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=self.opt.batchSize,
                                                      shuffle=shuffle,
                                                      num_workers=int(self.opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def create_dataloader(opt):
    data_loader = DataLoader(opt)
    data_loader.initialize()
    return data_loader
