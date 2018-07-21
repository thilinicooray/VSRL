import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import cv2
from scipy.misc import imread
import torch

from models.faster_rcnn.utils.blob import im_list_to_blob
from models.faster_rcnn.utils.config import cfg

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
        print('get item ', index)
        _id = self.ids[index]
        ann = self.annotations[_id]
        '''img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)'''

        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)


        #im_in = np.array(Image.open(os.path.join(self.img_dir, _id)).convert('RGB'))
        im_file = os.path.join(self.img_dir, _id)
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:,:,np.newaxis]
            im_in = np.concatenate((im_in,im_in,im_in), axis=2)

        im = im_in[:,:,::-1]

        print('read img ', index)

        blobs, im_scales = self._get_image_blob(im)
        print('got blob ', index)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        print('add image ', index)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()


        verb, roles, labels = self.encoder.encode(ann)
        print('CAME HERE')

        return im_data, im_info, gt_boxes, num_boxes, verb, roles, labels

    def __len__(self):
        return len(self.annotations)

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        print('image shape:', im_shape)
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []
        print('came for _get_image_blob ', im_size_min)

        for target_size in cfg.TEST.SCALES:
            print('came for _get_image_blob', target_size)
            im_scale = float(target_size) / float(im_size_min)
            print('got scale', im_scale)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            print('round', target_size)
            '''im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)'''
            im = self.transform(im_orig)
            print('resize', target_size)
            im_scale_factors.append(im_scale)
            print('scale factor', target_size)
            processed_ims.append(im)
            print('done', target_size)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)
        print('leaving _get_image_blob', len(np.array(im_scale_factors)))
        return blob, np.array(im_scale_factors)
