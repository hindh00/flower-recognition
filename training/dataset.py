"""Oxford 102 Flowers: download a stratified split, and
transforms"""

import json
import ssl
import tarfile
import urllib.request
from pathlib import Path

import _sympy_fix  # noqa: F401
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Flowers102

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
NUM_CLASSES = 102
SEGMENTATION_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz"


"""SSL workaround for robots.ox.ac.uk downloads"""
def _use_unverified_ssl() -> None:
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    )


"""Download with SSL fallback."""
def _download_with_ssl_fallback(root: str) -> None:
    try:
        Flowers102(root=root, split="train", download=True)
    except Exception as exc:
        print(
            f"Standard download failed ({exc!r}); retrying with an unverified "
            "SSL context (known robots.ox.ac.uk flakiness)."
        )
        _use_unverified_ssl()
        Flowers102(root=root, split="train", download=True)


"""Downloads and extracts the Oxford 102 segmentation trimaps"""
def download_segmentations(root: str) -> Path:
    root = Path(root)
    base = root / "flowers-102"
    seg_dir = base / "segmim"
    if seg_dir.exists() and any(seg_dir.iterdir()):
        return seg_dir

    archive_path = base / "102segmentations.tgz"
    base.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SEGMENTATION_URL, archive_path)
    except Exception as exc:
        print(
            f"Standard download failed ({exc!r}); retrying with an unverified "
            "SSL context (known robots.ox.ac.uk flakiness)."
        )
        _use_unverified_ssl()
        urllib.request.urlretrieve(SEGMENTATION_URL, archive_path)

    with tarfile.open(archive_path) as tf:
        tf.extractall(base)
    archive_path.unlink()
    return seg_dir


"""Download and merge all three official splits into one pool of (image_path, label) pairs"""
def load_all_samples(root: str):
    _download_with_ssl_fallback(root)
    samples = []
    for split in ("train", "val", "test"):
        ds = Flowers102(root=root, split=split, download=False)
        if hasattr(ds, "_image_files") and hasattr(ds, "_labels"):
            samples.extend(zip(ds._image_files, ds._labels))
        else:
            samples.extend((ds._image_files[i], ds[i][1]) for i in range(len(ds)))
    return samples


def stratified_split(samples, val_size=0.15, test_size=0.15, seed=42):
    paths, labels = zip(*samples)
    paths, labels = list(paths), list(labels)

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=(val_size + test_size), stratify=labels, random_state=seed
    )
    rest_test_fraction = test_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, test_size=rest_test_fraction, stratify=rest_labels, random_state=seed
    )
    return {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }


class FlowersDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # Simulates occlusion from overlapping leaves/other blooms.
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def get_dataloaders(root, batch_size=32, num_workers=2, seed=42):
    samples = load_all_samples(root)
    splits = stratified_split(samples, seed=seed)
    train_tf, eval_tf = get_transforms()

    datasets = {
        "train": FlowersDataset(splits["train"], transform=train_tf),
        "val": FlowersDataset(splits["val"], transform=eval_tf),
        "test": FlowersDataset(splits["test"], transform=eval_tf),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, splits


"""Load class names from the cat_to_name.json file"""
def load_class_names(cat_to_name_path="cat_to_name.json"):
    with open(cat_to_name_path) as f:
        cat_to_name = json.load(f)
    return [cat_to_name[str(i + 1)] for i in range(NUM_CLASSES)]
