from datasets import load_dataset, load_from_disk

from settings import NUM_CLASSES, input_state_shape, EPOCHS, PATCH_SIZE, DIRECTORY_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, \
    VALIDATION_FOLDER
from settings import TRAIN_FOLDER, TEST_FOLDER, BALANCED_PATH, RGB_BALANCED, RGB_PREPROCESSED_BALANCED, MINIMAL_RGB
from PIL import Image
from settings import BATCH_SIZE, checkpoint_filepath, RGB_DIRECTORY_PATH, RGB_CLAHE_PATH, RANDOM_SEED
from settings import RGB_DIRECTORY_PATH
from huggingface_hub import HfFolder
from transformers import DefaultDataCollator
from transformers import ViTFeatureExtractor
from torchvision import transforms
import datasets
from torch import Tensor
import torchvision
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

from settings import RGB_DIRECTORY_PATH

model_id = "google/vit-base-patch16-224-in21k"
train_batch_size = BATCH_SIZE

import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(RANDOM_SEED)

data_collator = DefaultDataCollator(return_tensors="pt")


class PGUImageDataSet(Dataset):
    def __init__(self, hug_dataset, one_hot=False, num_classes=2):
        self.hug_dataset = hug_dataset
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __len__(self):
        return len(self.hug_dataset)

    def __getitem__(self, idx):
        if self.one_hot:
            labels = torch.nn.functional.one_hot(torch.as_tensor(self.hug_dataset[idx]['label']),
                                                 num_classes=self.num_classes)
            return {'pixel_values': Tensor(self.hug_dataset[idx]['pixel_values']), 'labels': labels}
        else:
            return {'pixel_values': Tensor(self.hug_dataset[idx]['pixel_values']),
                    'labels': self.hug_dataset[idx]['label']}


class BaseClassLoader:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def process(self, examples):
        # print('dddddddddddddddddddddddddd', ))
        examples.update(self.feature_extractor(examples['image']))
        return examples

    def load_data_pytorch(self, root_path, folder, num_classes=2, one_hot=False):
        data_dir = root_path + folder + '/'

        transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()
                                        ])  # TODO: compose transforms here
        dataset = torchvision.datasets.ImageFolder(data_dir,
                                                   transform=self.feature_extractor)  # TODO: create the ImageFolder
        return dataset
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    def load_data(self, root_path, folder, num_classes=2, one_hot=False):
        # cat = 'train' if folder=='train' else 'test'
        print('dataset path = ', root_path + folder + '/')
        dataset = load_dataset("imagefolder", data_dir=root_path + folder, )
        cat = list(dataset.keys())[0]
        dataset = dataset[cat]
        print('data set len=', len(dataset))
        processed_data_set = dataset.map(self.process, batched=True)
        processed_data_set = PGUImageDataSet(processed_data_set, num_classes=num_classes, one_hot=one_hot)
        return processed_data_set

    def get_data_loader(self, path, folder, batch_size=1, shuffle=False, num_classes=2, one_hot=False):
        # ds = self.load_data_pytorch(root_path=path, folder=folder, num_classes=num_classes, one_hot = one_hot)
        ds = self.load_data(root_path=path, folder=folder, num_classes=num_classes, one_hot=one_hot)
        pt_ds = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, )
        # print(pt_ds.dataset)
        return pt_ds


if __name__ == "__main__":
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)
    base_loader = BaseClassLoader(feature_extractor=feature_extractor)

    train_dataset = base_loader.load_data(MINIMAL_RGB, folder='test', num_classes=2, one_hot=True)
    print('dataset len', len(train_dataset))
    pt_train_dataset = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True, )
    # collate_fn=data_collator)

    # print(type(pt_train_dataset.dataset[0]))
    # print('features', next(iter(pt_train_dataset)))
    train_features, train_labels = next(iter(pt_train_dataset))
    # print(f"Feature batch shape: {train_features[0]}")
    # print(f"Labels batch shape: {train_labels[0]}")
