import os
import datetime

import torch
import torch.optim as optim
import pickle
import time
import sys
import numpy as np


from utils.split_test_model import kernel_test_model_3d_VSR_swin_batch_softpadding
from utils.loss import batch_PSNR_linear_transform
from utils.read_utils import save_nz_tiff, save_tiff, is_specify_file, load_tif_img_origin, load_3d_mrc_dataset_origin
from model import get_model_with_opt
from utils.SSIM import ssim3D

import config.SRFormer_pMoE_train_config as opt

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(name):
    r"""
    A dirty hack to load a module from a string input

    Returns:
        A pointer to the loaded module
    """
    strCmd = "from utils.dataset import " + name + " as Dataset"
    exec(strCmd)
    return eval('Dataset')

def save_checkpoint(epoch, iter, state_dict, optimizer, path):
    torch.save({'epoch': epoch,
                'iter': iter,
                'state_dict': state_dict,
                'optimizer': optimizer}, path)


def test_and_save(model, device, iter, opt, save_name='default'): #TODO
    with torch.no_grad():
        if is_specify_file(opt.save_img, ".mrc"):
            load_function = load_3d_mrc_dataset_origin
        else:
            load_function = load_tif_img_origin
        input_t = load_function(os.path.join(opt.test_dir, opt.input_dir[-1], opt.save_img))

        if opt.is3D:
            if opt.img_frame != 0:
                input_t = input_t[:opt.img_frame]
        if not opt.input_img_neg:
            input_t = np.where(input_t < 0, 0, input_t)

        if opt.normal:
            input_t /= np.max(np.abs(input_t))

        # test split input img
        model.eval()

        if opt.split_test > 1:
            re = kernel_test_model_3d_VSR_swin_batch_softpadding(model, device, input_t, batch=1,kernel=(opt.noise_crop, opt.noise_crop, opt.input_frame),
                                               padding=(opt.noise_crop//5, opt.noise_crop//5, opt.input_frame//5),
                                               output_head=opt.output_frame,upfactor=opt.upfactor, zupfactor=opt.zupfactor,
                                                                 half=opt.half)  # TODO

        else:
            input_t = torch.tensor(input_t[:, np.newaxis])
            if opt.half:
                input_t=input_t.half()
            input_t = input_t.to(device)
            re = model(input_t)
            re = re.squeeze().cpu().numpy()

            input_t = input_t.squeeze().cpu().numpy()

        if len(re.shape) == 3:
            save_function = save_nz_tiff
        else:
            save_function = save_tiff

        filename = 'Res_it' + str(iter + 1).zfill(6) + '.tif'
        if save_name != 'default':
            filename = save_name + '.tif'
        save_function(os.path.join(img_dir, filename), re)

        if iter + 1 == opt.save_checkpoint:
            target_t = load_function(os.path.join(opt.test_dir, opt.gt_dir[-1], opt.save_img))
            save_function(os.path.join(img_dir, 'val_gt.tif'), target_t)
            save_function(os.path.join(img_dir, 'val_input.tif'), input_t)



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_num

    device = torch.device("cuda:"+opt.main_device_num)

    log_dir = os.path.join("/disk1/LASIM/log", opt.env_name)
    print(opt.env_name)

    if os.path.isfile(os.path.join(log_dir, 'model_it010000.pth') and not opt.resume):
        in_str = input(opt.env_name + " model is trained.\noverwrite it? (y/n)")
        if in_str != "y":
            sys.exit(1)

    mkdir(log_dir)
    img_dir = os.path.join(log_dir, "img")
    mkdir(img_dir)

    log_name = os.path.join(log_dir, datetime.datetime.now().strftime('%Y-%m-%dT%H') + "log.txt")
    with open(log_name, "w") as f:
        f.write(str(vars(opt)))

    #####1 Init Model
    if opt.resume:
        opt_resume_model = opt
        opt_resume_model.input_frame = opt.resume_input_frame
        opt_resume_model.output_frame = opt.resume_output_frame
        opt_resume_model.noise_crop = opt.resume_noise_crop
        opt_resume_model.clean_crop = opt_resume_model.noise_crop * opt_resume_model.upfactor
        opt_resume_model.crop_depth = opt_resume_model.input_frame
        model = get_model_with_opt(opt.arch, opt_resume_model)
    else:
        model = get_model_with_opt(opt.arch, opt)
    if opt.half:
        model.half()
    # model
    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    model = torch.nn.DataParallel(model, device_ids=opt.device_ids,
                                  output_device=opt.main_device_id)

    ####Load weights
    if opt.resume:
        pretrained = torch.load(opt.resume_path, map_location='cpu')
        model.load_state_dict(pretrained['state_dict'])

    model = model.cuda()
    # model.to(device)
    # init_weights(model, opt.init_type, opt.init_bn_type, opt.init_gain)

    # optimizer = optim.AdamW(UNet.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), eps=1e-8, weight_decay=0)
    start_epoch = 0
    iter = 0
    pklProfile = {"loss": [], "perceptual_loss":[],"loss_l1": [],"aux_loss": [], "psnr": [], "lr": [], "val_loss": [],
                  "val_perceptual_loss":[],"val_loss_l1": [],"val_psnr": [],"val_aux_loss": [],
                  "val_ssim":[], "best_epoch": 0, "best_iter": 0,"best_psnr": 0}

    ####Load weights
    if opt.resume:
        start_epoch = pretrained['epoch']
        optimizer.load_state_dict(pretrained['optimizer'])
        for p in optimizer.param_groups: lr = p['lr']
        iter = pretrained['iter']
        # pklProfile = np.load(os.path.join(log_dir, "model_loss.pkl"))
        print("loaded: " + opt.resume_path)

    if opt.scheduler is not None and opt.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.opti_decay, patience=opt.patience,
                                                               min_lr=opt.min_lr)
    elif opt.scheduler == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_step, last_epoch=-1,
                                                    gamma=opt.opti_decay)

    #####2 Load Data
    Dataset = load_dataset(opt.dataset)
    train_dataset = Dataset(opt.train_dir, opt)
    train_length = train_dataset.get_len()

    Dataset_val = load_dataset(opt.dataset_val)
    opt.augment_size = opt.val_augment_size
    val_dataset = Dataset_val(opt.val_dir, opt)
    val_length = val_dataset.get_len()
    print("env: %s train: %s val: %s" % (opt.env_name, opt.train_dir, opt.val_dir))
    print("start train...")
    with open(log_name, 'a') as f:
        f.write("env: %s train: %s val: %s" % (opt.env_name, opt.train_dir, opt.val_dir))
        f.write("start train...")



    loss_function = torch.nn.functional.l1_loss

    if opt.perceptual_loss:
        from utils.loss import LPIPSLoss
        perceptual_loss_function = LPIPSLoss(opt.perceptual_loss_weight,opt.perceptual_loss_xz,opt.perceptual_loss_yz)
    perceptual_loss = torch.zeros(1)
    aux_loss = torch.zeros(1)
    loss_l1_function = torch.nn.functional.l1_loss

    #####3 Training...
    print("using device: " + opt.device + " " +opt.device_num)
    with open(log_name, 'a') as f:
        f.write("using device: " + opt.device + " " +opt.device_num)
    for epoch in range(opt.nepoch):
        start_time = time.time()
        model.train()
        for iter_ in range(train_length):
            optimizer.zero_grad()
            target, input_ = train_dataset.get_data(iter_)
            if opt.half:
                target = target.half()
                input_ = input_.half()
            target = target.to(device)
            input_ = input_.to(device)
            if opt.moe_ffn:
                if opt.aux_loss:
                    restored, aux_loss = model(input_,train=True)
                else:
                    restored = model(input_, train=True)
            else:
                restored = model(input_)
            loss = loss_function(restored, target).mean()
            if opt.perceptual_loss:
                perceptual_loss = perceptual_loss_function(restored, target)
            loss_l1 = loss_l1_function(restored, target)

            pklProfile["lr"].append(optimizer.state_dict()['param_groups'][0]['lr'])
            pklProfile["loss_l1"].append(loss_l1.item())
            pklProfile["aux_loss"].append(aux_loss.item())
            pklProfile["perceptual_loss"].append(perceptual_loss.item())
            pklProfile["loss"].append(loss.item())

            if opt.perceptual_loss:
                loss += perceptual_loss.to(device)
            if opt.moe_ffn and opt.aux_loss:
                loss += aux_loss.to(device) * opt.aux_loss_weight
            loss.backward()
            optimizer.step()
            if opt.scheduler is not None and opt.scheduler == 'ReduceLROnPlateau':
                scheduler.step(loss)
            else:
                scheduler.step()
            iter += 1

            if (iter + 1) % opt.val_checkpoint == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_loss_l1 = 0
                    val_psnr = 0
                    val_ssim = 0
                    val_perceptual_loss =0
                    for j in range(val_length):
                        target_t, input_t = val_dataset.get_data(j)
                        if opt.half:
                            target_t = target_t.half()
                            input_t = input_t.half()
                        target_t = target_t.to(device)
                        input_t = input_t.to(device)
                        if opt.moe_ffn:
                            re = model(input_t, train=False).to(device)
                        else:
                            re = model(input_t).to(device)
                        val_loss += loss_function(re, target_t).mean().item()

                        if opt.perceptual_loss:
                            val_perceptual_loss += perceptual_loss_function(restored, target)
                        val_loss_l1 += loss_l1_function(restored, target)
                        val_psnr += batch_PSNR_linear_transform(re, target_t, device).item()
                        val_ssim += ssim3D(re.unsqueeze(1), target_t.unsqueeze(1)).item()

                    val_psnr /= val_dataset.get_full_len()
                    val_ssim /= val_length

                    pklProfile["val_loss"].append(val_loss)
                    pklProfile["val_perceptual_loss"].append(val_perceptual_loss)
                    pklProfile["val_loss_l1"].append(val_loss_l1)
                    pklProfile["val_psnr"].append(val_psnr)
                    pklProfile["val_ssim"].append(val_ssim)


                    if val_psnr > pklProfile["best_psnr"]:
                        pklProfile["best_epoch"] = epoch + 1
                        pklProfile["best_iter"] = iter + 1
                        pklProfile["best_psnr"] = val_psnr

                    print("Ep %d it %d  \tPSNR: %.4f\tSSIM: %.4f\tVal Loss: %.4f\tVal Loss P: %.4f\tVal Loss l1: %.4f\t\tbest_ep %d\tbest_it %d\tbest_psnr %.4f\n" %
                            (
                                epoch + 1, iter + 1, val_psnr, val_ssim, val_loss,val_perceptual_loss, val_loss_l1,
                                pklProfile["best_epoch"],
                                pklProfile["best_iter"],
                                pklProfile["best_psnr"]))
                    with open(log_name, 'a') as f:
                        f.write("Ep %d it %d  \tPSNR: %.4f\tSSIM: %.4f\tVal Loss: %.4f\tVal Loss P: %.4f\tVal Loss l1: %.4f\t\tbest_ep %d\tbest_it %d\tbest_psnr %.4f\n" %
                            (
                                epoch + 1, iter + 1, val_psnr, val_ssim, val_loss,val_perceptual_loss, val_loss_l1,
                                pklProfile["best_epoch"],
                                pklProfile["best_iter"],
                                pklProfile["best_psnr"]))

            # TODO
            if (iter + 1) % opt.save_checkpoint == 0:
                test_and_save(model, device, iter, opt)
                save_checkpoint(epoch + 1, iter + 1, model.state_dict(), optimizer.state_dict(),
                                os.path.join(log_dir, "model_it"+str(iter + 1).zfill(6)+".pth"))


        # if iter > opt.stages[2]:

        print("-----------------")
        print("Epoch: %d\tTime: %.4f\tLearningRate: %.7f" % (
            epoch + 1, time.time() - start_time, optimizer.state_dict()['param_groups'][0]['lr']))
        print("-----------------")
        with open(log_name, 'a') as f:
            f.write("-----------------\n")
            f.write("Epoch: %d\tTime: %.4f\tLearningRate: %.7f\n" % (
                epoch + 1, time.time() - start_time, optimizer.state_dict()['param_groups'][0]['lr']))
            f.write("-----------------\n")
        with open(os.path.join(log_dir, "model_loss.pkl"), 'wb') as handle:
            pickle.dump(pklProfile, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_checkpoint(opt.nepoch, 0, model.state_dict(), optimizer.state_dict(), os.path.join(log_dir, "model_latest.pth"))
