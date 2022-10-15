from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset
from traitlets import Bool

from src.common.utils import PROJECT_ROOT

import src.pl_data.setupdata as setupdata
import pickle as pk


class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, context: str, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        #data = setupdata.get_dataset()
        if context=="train":
            with open(str(PROJECT_ROOT)+'/data/train/train_dataset.pickle', 'rb') as pfile:
                data = pk.load(pfile)
            self.data = data["train"]
        elif context=="test":
            with open(str(PROJECT_ROOT)+'/data/test/test_dataset.pickle', 'rb') as pfile:
                data = pk.load(pfile)
            self.data = data["test"]
        elif context=="val":
            with open(str(PROJECT_ROOT)+'/data/val/val_dataset.pickle', 'rb') as pfile:
                data = pk.load(pfile)
            self.data = data["val"]


    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
