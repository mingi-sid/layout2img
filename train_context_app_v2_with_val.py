import argparse
import os
import pickle
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from collections import OrderedDict

from utils.util import *
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_app_v2 import *
from model.rcnn_discriminator_app import *
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger
from tqdm import tqdm

from skimage import img_as_ubyte
import imageio


def get_dataset(dataset, img_size):
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir='./datasets/coco/images/val2017/',
                                     instances_json='./datasets/coco/annotations/instances_val2017.json',
                                     stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                     stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == "coco_train":
        data = CocoSceneGraphDataset(image_dir='./datasets/coco/images/train2017/',
                                     instances_json='./datasets/coco/annotations/instances_train2017.json',
                                     stuff_json='./datasets/coco/annotations/stuff_train2017.json',
                                     stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        data = VgSceneGraphDataset(vocab_json='./data/tmp/vocab.json', h5_path='./data/tmp/preprocess_vg/val.h5',
                                   image_dir='./datasets/vg/',
                                   image_size=(img_size, img_size), max_objects=10, left_right_flip=False)
    return data


def main(args):
    # parameters
    img_size = args.image_size
    z_dim = 128
    lamb_obj = 1.0
    lamb_app = 1.0
    lamb_img = 0.1
    num_classes = 184 if args.dataset in ['coco', 'coco_train'] else 179
    num_obj = 8 if args.dataset in ['coco', 'coco_train'] else 31

    args.out_path = os.path.join(args.out_path, args.dataset, str(img_size))

    num_gpus = torch.cuda.device_count()
    num_workers = 4 * num_gpus
    if num_gpus > 1:
        parallel = True
        args.batch_size = args.batch_size * num_gpus
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    # data loader
    train_data = get_dataset(args.dataset, img_size)

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=False, shuffle=False, num_workers=num_workers)

    # Load model
    device = torch.device('cuda')
    if img_size == 128:
        netG = ResnetGenerator128_context(num_classes=num_classes, output_dim=3).to(device)
    elif img_size == 64:
        netG = ResnetGenerator64_context(num_classes=num_classes, output_dim=3).to(device)
    netD = CombineDiscriminator128_app(num_classes=num_classes).to(device)

    if (args.checkpoint_epoch is not None) and (args.checkpoint_epoch != 0):
        load_G = args.out_path + '/model/G_{}.pth'.format(args.checkpoint_epoch)
        load_D = args.out_path + '/model/D_{}.pth'.format(args.checkpoint_epoch)
        if not os.path.isfile(load_G) or not os.path.isfile(load_D):
            return
        
        # load generator
        state_dict = torch.load(load_G)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netG.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netG.load_state_dict(model_dict)
        netG.cuda()

        # load discriminator
        state_dict = torch.load(load_D)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`nvidia
            new_state_dict[name] = v

        model_dict = netD.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netD.load_state_dict(model_dict)
        netD.cuda()
    else:
        args.checkpoint_epoch = 0

    # parallel = False
    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = nn.DataParallel(netD)

    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    # make dirs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.makedirs(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()
    vgg_loss = nn.DataParallel(vgg_loss)
    l1_loss = nn.DataParallel(nn.L1Loss())
    for epoch in range(args.checkpoint_epoch, args.total_epoch):
        netG.train()
        netD.train()
        print("Epoch {}/{}".format(epoch, args.total_epoch))
        for idx, data in enumerate(tqdm(dataloader)):
            real_images, label, bbox = data
            # print(real_images.shape)
            # print(label.shape)
            # print(bbox.shape)
            real_images, label, bbox = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float()
            #print('bbox', bbox, bbox.shape)
            #print('label', label, label.shape)
            if idx == 0:
                print('real_images', real_images, real_images.shape)

            # update D network
            netD.zero_grad()
            real_images, label = real_images.to(device), label.long().to(device)
            d_out_real, d_out_robj, d_out_robj_app = netD(real_images, bbox, label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_robj_app = torch.nn.ReLU()(1.0 - d_out_robj_app).mean()
            # print(d_loss_robj)
            # print(d_loss_robj_app)

            for j in range(5):
                z = torch.randn(real_images.size(0), num_obj, z_dim).to(device)
                z_im = None#torch.randn(real_images.size(0), 1, z_dim).to(device)
                fake_images = netG(z, bbox, z_im, y=label.squeeze(dim=-1))
                #print('z', z, z.shape)
                if idx == 0:
                    print('fake_images', fake_images.shape)
                    print('fake_images.data', fake_images.cpu().data)
                for img_idx in range(args.batch_size * idx, min(args.batch_size * (idx + 1), len(train_data))):
                    #fake_images_orig = imagenet_deprocess_orig(fake_images)
                    fake_images_uint = img_as_ubyte(fake_images[img_idx % args.batch_size].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                    #fake_images_uint = fake_images_orig[img_idx % args.batch_size].cpu().detach().numpy().transpose(1, 2, 0)

                    # imageio.imwrite("{save_path}/sample_{idx}_numb_{numb}.jpg".format(save_path=args.sample_path, idx=idx, numb=j), fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                    imageio.imwrite("{save_path}/sample_{idx}_numb_{numb}.jpg".format(save_path=args.sample_path+'/coco/G_100/64_5', idx=img_idx, numb=j), fake_images_uint)

            for img_idx in range(args.batch_size * idx, min(args.batch_size * (idx + 1), len(train_data))):
                img_orig = imagenet_deprocess_orig(real_images)
                imageio.imwrite("{save_path}/sample_{idx}.jpg".format(save_path=args.sample_path+'/coco/G_100/64_5', idx=img_idx), img_orig[img_idx % args.batch_size].cpu().detach().numpy().transpose(1, 2, 0))
                img_orig2 = img_as_ubyte(real_images[img_idx % args.batch_size].cpu().detach().numpy().transpose(1,2,0)*0.5+0.5)
                imageio.imwrite("{save_path}/sample_{idx}_.jpg".format(save_path=args.sample_path+'/coco/G_100/64_5', idx=img_idx), img_orig2)


            '''d_out_fake, d_out_fobj, d_out_fobj_app = netD(fake_images.detach(), bbox, label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fobj_app = torch.nn.ReLU()(1.0 + d_out_fobj_app).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake) + lamb_app * (d_loss_robj_app + d_loss_fobj_app)
            #d_loss.backward()
            #d_optimizer.step()
            '''

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                '''
                g_out_fake, g_out_obj, g_out_obj_app = netD(fake_images, bbox, label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                g_loss_obj_app = - g_out_obj_app.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss + lamb_app * g_loss_obj_app
                #g_loss.backward()
                #g_optimizer.step()'''

            if (idx + 1) % 500 == 0 or True:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                '''
                logger.info("Step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                              idx + 1,
                                                                                                              d_loss_real.item(),
                                                                                                              d_loss_fake.item(),
                                                                                                              g_loss_fake.item()))
                logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                    d_loss_robj.item(),
                    d_loss_fobj.item(),
                    g_loss_obj.item()))
                logger.info("             d_obj_real_app: {:.4f}, d_obj_fake_app: {:.4f}, g_obj_fake_app: {:.4f} ".format(
                    d_loss_robj_app.item(),
                    d_loss_fobj_app.item(),
                    g_loss_obj_app.item()))

                logger.info("             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))
                '''

                writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)
                writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)

        '''
        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))
            torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch + 1)))'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size of training data. Default: 32')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/app2',
                        help='path to output files')
    parser.add_argument('--checkpoint_epoch', type=int, 
                        help='checkpoint epoch')
    parser.add_argument('--image-size', type=int, default=128,
                        help='size of the input & output image')
    parser.add_argument('--sample_path', type=str, default='./samples/tmp/app_train',
                        help='path to output files')
    args = parser.parse_args()
    main(args)
