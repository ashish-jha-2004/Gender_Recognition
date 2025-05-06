import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

IMG_SIZE = 64
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      # important for CNN to work well
])


class HandDataset(torch.utils.data.Dataset):
    # 1, 2, 3, 4, 5, 6, 7, 8 -> these are ordered as specified in AIML_project.pdf
    CLASSES = ('female-dorsal-right', 'female-dorsal-left', 'female-palmar-left', 'female-palmar-right',
               'male-dorsal-right', 'male-dorsal-left', 'male-palmar-left', 'male-palmar-right')

    def __init__(self, root_dir, csv_file, transform=TRANSFORM):
        self.ground_truth = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.files = []

        for i, row in self.ground_truth.iterrows():
            sample = {}
            sample['img_path'] = self.root_dir + '/' + row[7]
            sample['img_class'] = self.get_class_idx(row[2], row[6])
            self.files.append(sample)

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        img_class = self.files[idx]['img_class']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # one_hot_vector = np.zeros((8,), dtype=np.float32)
        # one_hot_vector[img_class] = 1
        # sample = {'image': image, 'class': torch.from_numpy(one_hot_vector)}

        sample = {'image': image, 'class': img_class}
        return sample

    def get_class_idx(self, gender, hand_aspect):
        split_aspects = hand_aspect.split()
        split_aspects.insert(0, gender)
        class_name = '-'.join(split_aspects)
        return self.CLASSES.index(class_name)
