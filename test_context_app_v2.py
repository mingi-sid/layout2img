import argparse
from collections import OrderedDict
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_app_v2 import *
from utils.util import *
import imageio
from tqdm import tqdm
from skimage import img_as_ubyte


def get_dataloader(dataset='coco', img_size=128):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/images/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        # with open("./datasets/vg/vocab.json", "r") as read_file:
        #     vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab_json='./data/tmp/vocab.json',
                                      h5_path='./data/tmp/preprocess_vg/val.h5',
                                      image_dir='./datasets/vg/',
                                      image_size=(img_size, img_size), left_right_flip=False, max_objects=10)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    device = torch.device('cuda')
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 11

    dataloader = get_dataloader(args.dataset, args.image_size)

    if args.image_size == 128:
        netG = ResnetGenerator128_context(num_classes=num_classes, output_dim=3).to(device)
    else:
        netG = ResnetGenerator64_context(num_classes=num_classes, output_dim=3).to(device)

    # args.model_path = args.model_path.format(args.dataset, args.load_epoch)
    # args.sample_path = args.sample_path.format(args.dataset, args.load_epoch)

    if not os.path.isfile(args.model_path):
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres = 2.0
    with tqdm(total=dataloader.__len__() * args.num_img) as pbar:
        for idx, data in enumerate(dataloader):
            real_images, label, bbox = data
            real_images, label, bbox = real_images.cuda(), label.long().unsqueeze(-1).cuda(), bbox.float()
            print('bbox', bbox, bbox.shape)
            print('label', label, label.shape)
            print('real_images', real_images, real_images.shape)

            for j in range(args.num_img):

                z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
                # z_obj = torch.randn((1, num_o, 128)).float().cuda()
                print('z_obj', z_obj, z_obj.shape)
                z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
                # z_im = torch.randn((1, 1, 128)).float().cuda()
                print('z_im', z_im, z_im.shape)
                fake_images = netG(z=z_obj, bbox=bbox, z_im=z_im, y=label.squeeze(dim=-1))
                print('fake_images', fake_images.shape)
                print('fake_images.data', fake_images.cpu().data)
                fake_images_uint = img_as_ubyte(fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                # imageio.imwrite("{save_path}/sample_{idx}_numb_{numb}.jpg".format(save_path=args.sample_path, idx=idx, numb=j), fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                imageio.imwrite("{save_path}/sample_{idx}_numb_{numb}.jpg".format(save_path=args.sample_path, idx=idx, numb=j), fake_images_uint)
                pbar.update(1)
            img_orig = imagenet_deprocess_orig(real_images)
            imageio.imwrite("{save_path}/sample_{idx}.jpg".format(save_path=args.sample_path, idx=idx), img_orig[0].cpu().detach().numpy().transpose(1, 2, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg',
                        help='training dataset')
    parser.add_argument('--load_epoch', type=int, default=200,
                        help='which checkpoint to load')
    parser.add_argument('--model_path', type=str, default='/home/liao/work_code/LostGANs/outputs/tmp/app2/coco/128/model/G_170.pth', help='which epoch to load')
    parser.add_argument('--num_img', type=int, default=5, help="number of image to be generated for each layout")
    parser.add_argument('--sample_path', type=str, default='samples/tmp/app2/coco/G170/128_5',
                        help='path to save generated images')
    parser.add_argument('--image_size', type=int, default=128,
                        help="size of the input&output image")
    args = parser.parse_args()
    main(args)
