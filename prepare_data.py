import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# train: 0.8, val: 0.2
def prepare_data(
        raw_train_dir: str = "data/raw/train",
        test_dir: str = "data/raw/test",
        processed_dir: str = "data/processed",
        val_ratio: float = 0.2,
        seed: int = 42
):
    random.seed(seed)
    raw_train_path = Path(raw_train_dir)
    test_path = Path(test_dir)
    processed_path = Path(processed_dir)

    # create folder
    for split in ['train', 'val']:
        for category in ['cats', 'dogs']:
            path = processed_path / split / category
            path.mkdir(parents=True, exist_ok=True)
    (processed_path / 'test').mkdir(parents=True, exist_ok=True)

    cat_img = [f for f in os.listdir(raw_train_dir) if f.startswith('cat.')]
    dog_img = [f for f in os.listdir(raw_train_dir) if f.startswith('dog.')]
    test_img = [f for f in os.listdir(test_dir)]

    # split dataset
    val_cat_size = int(len(cat_img) * val_ratio)
    val_dog_size = int(len(dog_img) * val_ratio)

    random.shuffle(cat_img)
    random.shuffle(dog_img)

    train_cats = cat_img[val_cat_size:]
    val_cats = cat_img[:val_cat_size]
    train_dogs = dog_img[val_dog_size:]
    val_dogs = dog_img[:val_dog_size]

    copy_files(train_cats, raw_train_path, processed_path / 'train' / 'cats', 'Train Cats')
    copy_files(val_cats, raw_train_path, processed_path / 'val' / 'cats', 'Val Cats')
    copy_files(train_dogs, raw_train_path, processed_path / 'train' / 'dogs', 'Train Dogs')
    copy_files(val_dogs, raw_train_path, processed_path / 'val' / 'dogs', 'Val Dogs')
    copy_files(test_img, test_path, processed_path / 'test', 'Test Images')

    tqdm.write(f"🎉 Data preparation complete!")
    tqdm.write(f"Training set: {len(train_cats)} cats and {len(train_dogs)} dogs")
    tqdm.write(f"Validation set: {len(val_cats)} cats and {len(val_dogs)} dogs")
    tqdm.write(f"Test set: {len(test_img)} images")


def copy_files(file_list, src_dir, dst_dir, desc='Copying'):
    for file in tqdm(file_list, desc=f"🚀 {desc}"):
        shutil.copy2(src_dir / file, dst_dir / file)


if __name__ == "__main__":
    prepare_data()
