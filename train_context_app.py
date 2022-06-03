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

from utils.util import *
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_context import *
from model.rcnn_discriminator_app import *
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger
from tqdm import tqdm


def get_dataset(dataset, img_size):
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir='./datasets/coco/images/train2017/',
                                     instances_json='./datasets/coco/annotations/instances_train2017.json',
                                     stuff_json='./datasets/coco/annotations/stuff_train2017.json',
                                     stuff_only=True, image_size=(img_size, img_size), left_right_flip=True)
    elif dataset == 'vg':
        data = VgSceneGraphDataset(vocab_json='./data/tmp/vocab.json', h5_path='./data/tmp/preprocess_vg/train.h5',
                                   image_dir='./datasets/vg/',
                                   image_size=(img_size, img_size), max_objects=30, left_right_flip=True)
    return data


def main(args):
    # parameters
    img_size = args.image_size
    z_dim = 128
    lamb_obj = 1.0
    lamb_app = 1.0
    lamb_img = 0.1
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 31

    args.out_path = os.path.join(args.out_path, args.dataset + '_1gpu', str(img_size))

    # data loader
    train_data = get_dataset(args.dataset, img_size)

    num_gpus = torch.cuda.device_count()
    num_workers = 4 * num_gpus
    if num_gpus > 1:
        parallel = True
        args.batch_size = args.batch_size * num_gpus
        num_workers = num_workers * num_gpus
    else:
        parallel = False

    print("{} GPUs, {} workers are used".format(num_gpus, num_workers))
    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=True, shuffle=True, num_workers=num_workers)

    # Load model
    device = torch.device('cuda')
    if img_size == 128:
        netG = context_aware_generator(num_classes=num_classes, output_dim=3).to(device)
    else:
        netG = context_aware_generator64(num_classes=num_classes, output_dim=3).to(device)
    netD = CombineDiscriminator128_app(num_classes=num_classes).to(device)

    # if os.path.isfile(args.checkpoint):
    #     state_dict = torch.load(args.checkpoint)

    # parallel = True
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
    # writer = None
    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()
    vgg_loss = nn.DataParallel(vgg_loss)
    l1_loss = nn.DataParallel(nn.L1Loss())
    for epoch in range(args.total_epoch):
        netG.train()
        netD.train()
        print("Epoch {}/{}".format(epoch, args.total_epoch))
        for idx, data in enumerate(tqdm(dataloader)):
            real_images, label, bbox = data
            real_images, label, bbox = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float()

            # update D network
            netD.zero_grad()
            real_images, label = real_images.to(device), label.long().to(device)
            d_out_real, d_out_robj, d_out_robj_app = netD(real_images, bbox, label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_robj_app = torch.nn.ReLU()(1.0 - d_out_robj_app).mean()
            # print(d_loss_robj)
            # print(d_loss_robj_app)

            z = torch.randn(real_images.size(0), num_obj, z_dim).to(device)
            fake_images = netG(z, bbox, y=label.squeeze(dim=-1))
            d_out_fake, d_out_fobj, d_out_fobj_app = netD(fake_images.detach(), bbox, label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fobj_app = torch.nn.ReLU()(1.0 + d_out_fobj_app).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake) + lamb_app * (d_loss_robj_app + d_loss_fobj_app)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj, g_out_obj_app = netD(fake_images, bbox, label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()
                g_loss_obj_app = - g_out_obj_app.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + pixel_loss + feat_loss + lamb_app * g_loss_obj_app
                g_loss.backward()
                g_optimizer.step()
            # print("d_loss_real={:.3f}, d_loss_robj={:.3f}, d_loss_robj_app={:.3f}".format(d_loss_real.item(), d_loss_robj.item(), d_loss_robj_app.item()))
            # print("d_loss_fake={:.3f}, d_loss_fobj={:.3f}, d_loss_fobj_app={:.3f}".format(d_loss_fake.item(), d_loss_fobj.item(), d_loss_fobj_app.item()))
            # print("g_loss_fake={:.3f}, g_loss_obj={:.3f}, g_loss_obj_app={:.3f}".format(g_loss_fake.item(), g_loss_obj.item(), g_loss_obj_app.item()))
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
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
                if writer is not None:
                    writer.add_image("real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)
                    writer.add_image("fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4), epoch * len(dataloader) + idx + 1)

                    writer.add_scalars("D_loss_real", {"real": d_loss_real.item(),
                                                       "robj": d_loss_robj.item(),
                                                       "robj_app": d_loss_robj_app.item(),
                                                       "loss": d_loss.item()})
                    writer.add_scalars("D_loss_fake", {"fake": d_loss_fake.item(),
                                                       "fobj": d_loss_fobj.item(),
                                                       "fobj_app": d_loss_fobj_app.item()})
                    writer.add_scalars("G_loss", {"fake": g_loss_fake.item(),
                                                  "obj_app": g_loss_obj_app.item(),
                                                  "obj": g_loss_obj.item(),
                                                  "loss": g_loss.item()})

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))
            torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='mini-batch size of training data. Default: 32')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs/tmp/context_app',
                        help='path to output files')
    parser.add_argument('--image_size', type=int, default=128,
                        help='size of the input&output image')
    # parser.add_argument('--checkpoint', type=str, default='./outputs/tmp/app/model/G_10.pth',
    #                     help='path of checkpoint')
    args = parser.parse_args()
    main(args)
