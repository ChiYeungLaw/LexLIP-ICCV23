from glob import glob
from .base_dataset import CsvDataset
import io
from PIL import Image


class SBUDataset(CsvDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        
        if split == "train":
            input_filename = f"sbu_out.tsv"
        else:
            input_filename = None
        
        img_key = "filepath"
        caption_key = "title"

        super().__init__(
            *args,
            **kwargs,
            input_filename=input_filename,
            img_key=img_key,
            caption_key=caption_key,
            dataset_name="sbu",
        )


    def __getitem__(self, index):
        return self.get_suite(index)