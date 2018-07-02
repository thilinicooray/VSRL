import torch
from imsitu_encoder import imsitu_encoder
from imsitu_loader import imsitu_loader
import json

def main():
    train_set = json.load(open("imsitu_data/train.json"))
    encoder = imsitu_encoder(train_set)

    train_set = imsitu_loader('of500_images_resized', train_set, encoder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=3, shuffle=True, num_workers=1)

    for i, (img, verb, roles, labels) in enumerate(train_loader):
        print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        break

if __name__ == "__main__":
    main()






