from .base_dataset import CsvDataset
import io
from PIL import Image

class F30KCaptionKarpathyDataset(CsvDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            input_filename = "f30k_train.tsv"
        elif split == "val":
            input_filename = "f30k_val.tsv"
        elif split == "test":
            input_filename = "f30k_test.tsv"

        img_key = "filepath"
        caption_key = "title"
        img_id_key = "image_id"

        super().__init__(
            *args,
            **kwargs,
            input_filename=input_filename,
            img_key=img_key,
            caption_key=caption_key,
            img_id_key=img_id_key,
            dataset_name="F30k",
        )


    def __getitem__(self, index):
        suite = self.get_suite(index)

        return suite
