import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

class imsitu_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        '''
        baseline crf uses below 3 also for training transform and only first 2 for dev transform
        tv.transforms.Scale(224),
        tv.transforms.RandomCrop(224),
        tv.transforms.RandomHorizontalFlip(),
        '''
        self.transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, roles, labels = self.encoder.encode(ann)

        return img, verb, roles, labels

    def __len__(self):
        return len(self.annotations)
