import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.utils.data.distributed import DistributedSampler

NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(data_dirs, recursive: bool = True):
    """
    Collect image files under data_dirs.

    - Accepts common extensions (case-insensitive).
    - Searches recursively by default to match different dataset layouts.
    """
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [str(data_dirs)]

    missing_dirs = []
    image_paths = []

    for directory in data_dirs:
        directory = str(directory)
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
            continue
        if recursive:
            for root, _dirs, files in os.walk(directory):
                for name in files:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in _IMAGE_EXTS:
                        image_paths.append(os.path.join(root, name))
        else:
            for name in os.listdir(directory):
                full = os.path.join(directory, name)
                if not os.path.isfile(full):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in _IMAGE_EXTS:
                    image_paths.append(full)

    image_paths.sort()
    if len(image_paths) == 0:
        msg = [
            "No images found for dataset.",
            f"Scanned dirs: {list(data_dirs)}",
        ]
        if missing_dirs:
            msg.append(f"Missing dirs: {missing_dirs}")
        msg.append(f"Accepted extensions: {sorted(_IMAGE_EXTS)}")
        raise RuntimeError("\n".join(msg))
    return image_paths


class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = _collect_images(data_dir, recursive=True)
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.RandomCrop((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert("RGB")
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir, crop_size: int | None = None):
        self.data_dir = data_dir
        self.imgs = _collect_images(self.data_dir, recursive=True)
        self.crop_size = crop_size

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        name = os.path.basename(image_ori)
        image = Image.open(image_ori).convert("RGB")
        if self.crop_size is not None:
            crop = int(self.crop_size)
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop((crop, crop)),
                    transforms.ToTensor(),
                ]
            )
            img = self.transform(image)
        else:
            # No crop: return full resolution tensor. Any padding needed for the network
            # is handled in main.py eval helpers (tiled/direct full-res).
            self.transform = transforms.ToTensor()
            img = self.transform(image)
        return img, name

    def __len__(self):
        return len(self.imgs)


def get_loader(args, config):
    if args.trainset != "DIV2K":
        raise ValueError("Only DIV2K is supported; set --trainset DIV2K.")

    train_dataset = HR_image(config, config.train_data_dir)
    val_dir = getattr(config, "val_data_dir", config.test_data_dir)
    val_dataset = Datasets(val_dir, crop_size=getattr(config, "eval_crop_size", None))
    test_dataset = Datasets(config.test_data_dir, crop_size=getattr(config, "eval_crop_size", None))

    def worker_init_fn_seed(worker_id):
        seed = 10
        # Make dataloader workers different across ranks in DDP.
        try:
            seed += int(os.environ.get("RANK", "0")) * 1000
        except Exception:
            pass
        seed += worker_id
        np.random.seed(seed)

    sampler = None
    shuffle = True
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=NUM_DATASET_WORKERS,
        pin_memory=True,
        batch_size=config.batch_size,
        worker_init_fn=worker_init_fn_seed,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        persistent_workers=bool(NUM_DATASET_WORKERS) and NUM_DATASET_WORKERS > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False
    )

    return train_loader, val_loader, test_loader
