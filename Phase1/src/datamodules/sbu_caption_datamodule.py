from ..datasets import SBUDataset
from .datamodule_base import BaseDataModule


class SBUDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SBUDataset

    @property
    def dataset_name(self):
        return "sbu"