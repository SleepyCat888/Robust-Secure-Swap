from argparse import ArgumentParser
from datetime import datetime
import os
import torch.nn as nn
import torch
import numpy as np
import random
from data_loader_Swapping import GetLoader
from torchvision import transforms
from datetime import timedelta
import torch.nn.functional as F
import time
import torchvision

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(224),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)

def denorm(x):
    x = x * std + mean
    return x 



def arcface_trans(img_id, netArc,no_grad):
    if no_grad:
        with torch.no_grad():
            img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')  
            latent_id       = netArc(img_id_112)
            latent_id       = F.normalize(latent_id, p=2, dim=1)
    else:
        img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')  
        latent_id       = netArc(img_id_112)
        latent_id       = F.normalize(latent_id, p=2, dim=1)      
    return latent_id

def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'


def load_configs_initialize_training():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--netArc_ckpt", type=str, default='')
    parser.add_argument("--pretrained_ckpt", type=str, default='')
    parser.add_argument("--stu_ckpt", type=str, default='')
    parser.add_argument("--data_path", type=str, default='')

    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)


    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--image_steps", type=int, default=1000)
    parser.add_argument("--same_steps", type=int, default=3)


    parser.add_argument("--lamda_img", type=float, default=10.0)
    parser.add_argument("--lamda_mark",type=float, default=3.0)
    parser.add_argument("--lamda_id", type=float, default=1.0)
    parser.add_argument("--lamda_recon", type=float, default=1.0)
    parser.add_argument("--lamda_cyc", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0003)

    parser.add_argument("--if_use_poi", type=float, default=1.0)
    parser.add_argument("--num_id", type=int, default=512)
    parser.add_argument("--train_parts", type=float, default=0.8)

    args = parser.parse_args()
    return args


class NormalizeToNegOnePosOne:
    def __call__(self, tensor):
        # Scale [0, 1] to [-1, 1]
        return tensor * 2 - 1


if __name__ == "__main__":


    now = datetime.now()
    dt_string = f'{now.strftime("%y%m%d%H%M%S")}'

    exp_path = os.path.join('runs',dt_string)
    model_save_dir = os.path.join('runs',dt_string,'models')
    sample_dir = os.path.join('runs',dt_string,'samples')
    data_dir = os.path.join('runs',dt_string,'data')
    os.makedirs(exp_path,exist_ok=True)
    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(sample_dir,exist_ok=True)
    os.makedirs(data_dir,exist_ok=True)

    args = load_configs_initialize_training()

    with open(os.path.join(exp_path,'options.txt'), 'a') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Id network
    netArc = torch.load(args.netArc_ckpt, map_location=torch.device("cpu"))
    netArc.to(device)
    netArc.eval()
    
    from models.fs_networks import Generator_Adain_Upsample
    teacher = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
    teacher.load_state_dict(torch.load(args.pretrained_ckpt))
    teacher.eval()
    teacher.to(device)

    from models.fs_networks_stu import Generator_Adain_Upsample_stu
    student = Generator_Adain_Upsample_stu(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
    student.load_state_dict(torch.load(args.stu_ckpt),strict=False)
    student.train()
    student.to(device)

    # Loads hidden decoder
    from efficientnet_pytorch import EfficientNet
    wm_decoder = EfficientNet.from_pretrained('efficientnet-b0')
    feature = wm_decoder._fc.in_features
    wm_decoder._fc = nn.Linear(in_features=feature, out_features=args.bits, bias=True)
    wm_decoder.to(device)
    try:
        pretrained_weights = torch.load(args.stu_ckpt.replace('student', 'decoder'), map_location=torch.device("cpu"))
        wm_decoder.load_state_dict(pretrained_weights,strict=True)
        wm_decoder.to(device)
        print('Loaded!')
    except:
        pass

    optimizer = torch.optim.Adam(list(student.parameters()) + list(wm_decoder.parameters()), lr=args.lr)
    for param in [*teacher.parameters()]:
        param.requires_grad = False
    for param in [*student.parameters(), *wm_decoder.parameters()]:
        param.requires_grad = True


    train_loader    = GetLoader(args.data_path, args.batch, 8, args.seed)
    
    lpips_norm = NormalizeToNegOnePosOne()

    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    def cos_loss(x1, x2):
        return F.cosine_similarity(x1, x2).mean()

    randindex = [i for i in range(args.batch)]
    random.shuffle(randindex)
    start_time = time.time()

    from torch.utils import data
    from data_loader_Swapping import CustomImageDataset, CelebAImageDataset
    ipr_set = CelebAImageDataset(ID_num=args.num_id, train_num=args.train_parts)
    ipr_loader = data.DataLoader(dataset=ipr_set, drop_last=True, batch_size=args.batch, shuffle=True, num_workers=8)
    ipr_iter = iter(ipr_loader)

    ######################### train ######################
    for global_step in range(1, args.train_steps + 1):
        src_image1, src_image2  = train_loader.next()
        wm = torch.zeros((args.batch, args.bits), dtype=torch.float).random_(0, 2).to(device)
        visual_results = None

        if global_step % args.same_steps == 0:
            latent_id = arcface_trans(src_image2, netArc, True)
            with torch.no_grad():
                teacher_output = teacher(src_image1, latent_id)
            student_output = student(src_image1, latent_id, wm)

            loss_lpips = args.lamda_img * loss_fn_vgg.forward(lpips_norm(teacher_output), lpips_norm(student_output)).mean()
            loss_recon = args.lamda_recon * mse_loss(src_image1, student_output)

            swaped_id = arcface_trans(student_output, netArc, False)
            loss_id = args.lamda_id * (1 - cos_loss(latent_id, swaped_id))

            student_output_cyc = student(src_image1, swaped_id, wm)
            loss_cyc = args.lamda_cyc * mse_loss(src_image1, student_output_cyc)

            prd_wm = wm_decoder(student_output)
            loss_wm = bce_loss(prd_wm.float(), wm.float())

            total_loss = loss_lpips + loss_recon + loss_id + loss_cyc + loss_wm
            visual_results = student_output

        else:
            src_image2 = src_image2[randindex]
            latent_id = arcface_trans(src_image2, netArc, True)
            with torch.no_grad():
                teacher_output = teacher(src_image1, latent_id)
            student_output = student(src_image1, latent_id, wm)

            loss_lpips = args.lamda_img * loss_fn_vgg.forward(lpips_norm(teacher_output), lpips_norm(student_output)).mean()
            
            swaped_id = arcface_trans(student_output, netArc, False)
            loss_id = args.lamda_id * (1 - cos_loss(latent_id, swaped_id))

            prd_wm = wm_decoder(student_output)
            loss_wm = bce_loss(prd_wm.float(), wm.float())

            total_loss = loss_lpips + loss_id + loss_wm
            visual_results = student_output

        if args.if_use_poi:
            try:
                pro_ids = next(ipr_iter).to(device)

            except:
                ipr_iter = iter(ipr_loader)
                pro_ids = next(ipr_iter).to(device)

            poi_latent = arcface_trans(pro_ids, netArc, False)
            POI_student  = student(src_image1, poi_latent, wm)

            target = torch.zeros_like(src_image1).to(device)
            loss_poi = mse_loss(POI_student, target)
            total_loss += loss_poi

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if global_step % args.log_step == 0:
            last_time = time.time() - start_time

            wm_predicted = (prd_wm > 0.0).float()
            bitwise_acc = 100 * (1.0 - torch.mean(torch.abs(wm - wm_predicted)))
            
            if args.if_use_poi:
                log = f'SimSwap {dt_string} {global_step} Lpips-Loss:{loss_lpips.item():.5f} ID-Loss:{loss_id.item():.5f} Recon-Loss:{loss_recon.item():.5f} Cyc-Loss:{loss_cyc.item():.5f} POI-Loss:{loss_poi.item():.5f} WM-Loss:{loss_wm.item():.5f} WM-Acc:{bitwise_acc.item():.5f} Time:[{format_time(last_time)}]'
            else:
                log = f'SimSwap {dt_string} {global_step} Lpips-Loss:{loss_lpips.item():.5f} ID-Loss:{loss_id.item():.5f} Recon-Loss:{loss_recon.item():.5f} Cyc-Loss:{loss_cyc.item():.5f} WM-Loss:{loss_wm.item():.5f} WM-Acc:{bitwise_acc.item():.5f} Time:[{format_time(last_time)}]'
            print(log)
            with open(os.path.join(exp_path, 'wm_logs.txt'), 'a', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')

        if global_step % args.save_steps == 0:
            torch.save(student.state_dict(), os.path.join(model_save_dir, f'{global_step}-student_weights.pth'))
            torch.save(wm_decoder.state_dict(), os.path.join(model_save_dir, f'{global_step}-decoder_weights.pth'))

        if global_step % args.image_steps == 0:
            validation_image = []
            for (idx, x) in enumerate(visual_results):
                validation_image = validation_image + [transforms.Resize(224)(src_image1)[idx], transforms.Resize(224)(denorm(src_image2))[idx], x]
            validation_image = torchvision.utils.make_grid(validation_image, nrow=3)
            torchvision.utils.save_image(validation_image,os.path.join(sample_dir, f"{global_step}.jpg"))






