import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, \
    add_weight_decay, OTE_detection
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch-size', default=56, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--data-set', default='coco', type=str)
parser.add_argument('--gpus', default=1, type=int)
parser.add_argument('--output-dir', default="./", type=str)

# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def main():
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    if args.gpus > 1:
        device_ids = [i for i in range(args.gpus)]
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=0)
        print("Data Parallel enabled")

    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(0)
    # model = torch.nn.DataParallel(model,device_ids=[0])

    print('done')

    val_transform = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        # normalize, # no need, toTensor does normalization
                        ])
    train_transform = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        CutoutPIL(cutout_factor=0.5),
                        RandAugment(),
                        transforms.ToTensor(),
                        # normalize,
                        ])
    # Data loading
    instances_path_train = os.path.join(args.data, 'train.json')
    instances_path_val = os.path.join(args.data, 'val.json')
    val_dataset = OTE_detection(args.data,
                                instances_path_val,
                                val_transform)
    train_dataset = OTE_detection(args.data,
                                instances_path_train,
                                train_transform)

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr, args.output_dir)


def train_multi_label_coco(model, train_loader, val_loader, lr, output_dir):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    # validate before training
    for epoch in range(Epochs):
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0 or (i == (len(train_loader) - 1)):
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        ema_state_dict = ema.module.state_dict()
        model_state_dict = model.state_dict()
        save_model(model_state_dict, epoch, optimizer, scheduler, output_dir, 'main')
        save_model(ema_state_dict, epoch, optimizer, scheduler, output_dir, 'ema')

        model.eval()

        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.pth'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for _, (input, target) in enumerate(val_loader):
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)

def save_model(state_dict, epoch, optimizer, scheduler, output_dir, name):
    checkpoint = {
            'state_dict': state_dict,
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }

    torch.save(checkpoint, os.path.join(
            output_dir, 'model-{}-{}.ckpt'.format(name, epoch + 1)))

if __name__ == '__main__':
    main()
