from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.celeba import CelebA
from torchvision import transforms
import torchvision.transforms.functional as F
import os
import pandas as pd


class CropCelebA128(object):
    def __call__(self, img):
        new_img = F.crop(img, 57, 25, 128, 128)
        return new_img

class CELEBA128(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = self.config["data_path"]
        self.image_channel = self.config["image_channel"]
        self.image_size = self.config["image_size"]
        self.split = self.config["split"]
        self.augmentation = self.config["augmentation"]

        assert self.image_size == 128

        # load celeba images using torchvision dataset
        self.celeba_train = CelebA(self.data_path, split='train', target_type='attr', download=True)
        self.celeba_valid = CelebA(self.data_path, split='valid', target_type='attr', download=True)
        self.celeba_test = CelebA(self.data_path, split='test', target_type='attr', download=True)
        assert len(self.celeba_train) == 162_770 and len(self.celeba_valid) == 19_867 and len(self.celeba_test) == (19_962)

        if self.augmentation:
            self.transform = transforms.Compose([
                CropCelebA128(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])
            
            if "ssl_aug" in self.config.keys():
                ssl_aug = self.config["ssl_aug"]
            else:
                ssl_aug = "simclr_waug"

            if ssl_aug == "simsiam":
                self.ssl_transform = transforms.Compose([
                    CropCelebA128(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,), inplace=True)
                ])
            elif ssl_aug == "simclr":
                s = 1.0
                self.ssl_transform = transforms.Compose([
                    CropCelebA128(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,), inplace=True)
                ])
            elif ssl_aug == "simclr_waug":
                s = 1.0
                self.ssl_transform = transforms.Compose([
                    CropCelebA128(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,), inplace=True)
                ])
            elif ssl_aug == "simclr_waug2":
                s = 1.0
                self.ssl_transform = transforms.Compose([
                    CropCelebA128(),
                    transforms.RandomApply([transforms.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,), inplace=True)
                ])
            elif ssl_aug == "mae":
                self.ssl_transform = transforms.Compose([
                    CropCelebA128(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,), inplace=True)
                ])
            else:
                assert False, 'Specified SSL aug not found!'
        else:
            self.transform = transforms.Compose([
                CropCelebA128(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])

            self.ssl_transform = transforms.Compose([
                CropCelebA128(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])

        with open(os.path.join(self.data_path, "celeba", "list_attr_celeba.txt")) as f:
            f.readline() # discard the top line
            self.df = pd.read_csv(f, delim_whitespace=True)

        self.id_to_label = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
        self.num_attributes = len(self.id_to_label)
        self.label_to_id = {v: k for k, v in enumerate(self.id_to_label)}

    # train: 0~162,769  162,770
    # valid: 162,770~182,636   19,867
    # test: 182,637~202,599   19,963
    def __len__(self):
        if self.split == "train":
            return len(self.celeba_train)
        if self.split == "valid":
            return len(self.celeba_valid)
        if self.split == "test" or self.split == "eval":
            return len(self.celeba_test)
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        if self.split == "train":
            celeba_dataset = self.celeba_train
        elif self.split == "valid":
            celeba_dataset = self.celeba_valid
        elif self.split == "test" or self.split == "eval":
            celeba_dataset = self.celeba_test
        else:
            raise NotImplementedError()

        image, attr = celeba_dataset[index]
        assert attr.shape == (40,)

        image_dm = self.transform(image)
        image_ssl1 = self.ssl_transform(image)
        image_ssl2 = self.ssl_transform(image)
        gt = image_dm.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return {
            "x_0": image_dm,
            "x_ssl1": image_ssl1,
            "x_ssl2": image_ssl2,
            "gt": gt,
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
            "label": attr # should be [0, 1, 1, 0, 1, ...], not sure why original repo had -1 here in comment
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        x_0=[]
        x_ssl1=[]
        x_ssl2=[]
        gts = []
        x_T=[]
        label = []
        for i in range(batch_size):
            x_0.append(batch[i]["x_0"])
            x_ssl1.append(batch[i]["x_ssl1"])
            x_ssl2.append(batch[i]["x_ssl2"])
            gts.append(batch[i]["gt"])
            x_T.append(batch[i]["x_T"])
            label.append(batch[i]["label"])

        x_0 = torch.stack(x_0, dim=0)
        x_ssl1 = torch.stack(x_ssl1, dim=0)
        x_ssl2 = torch.stack(x_ssl2, dim=0)
        x_T = torch.stack(x_T, dim=0)
        label = torch.stack(label, dim=0)

        return {
            "net_input": {
                "x_0": x_0,
                "x_ssl1": x_ssl1,
                "x_ssl2": x_ssl2,
                "x_T": x_T,
                "label": label,
            },
            "gts": np.asarray(gts),
        }
