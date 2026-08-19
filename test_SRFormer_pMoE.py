import os
import torch
import numpy as np
from utils.read_mrc import read_mrc_with_hd, write_mrc
import tifffile
from utils.split_test_model import kernel_test_model_3d_VSR_swin_batch_softpadding
from contextlib import contextmanager
import glob
import time
from model import get_model_with_opt

@contextmanager
def timeblock(label, debug=1):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        if debug:
            print('{} : {}'.format(label, end - start))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cuda_number = '7'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_number

    from config.SRFormer_pMoE_train_config import _C as opt
    arch = "getSRFormer_pMoE"
    batch = 2
    pretrained_model = './pretrained_model/pretrained_SRFormer_pMoE.pth'
    
    filelist = sorted(glob.glob('./data/*LLSM.tif'))

    print('using cuda:'+str(cuda_number))
    with torch.no_grad():
        Net = get_model_with_opt(arch, opt)

        Net = torch.nn.DataParallel(Net,device_ids=[0],output_device=0).cuda()

        weights = torch.load(pretrained_model, map_location='cuda')


        Net.load_state_dict(weights['state_dict'], strict=True)
        Net.cuda()
        Net.eval()

        for f in filelist:
            save_name = f[0:-4] + "_SRFormer_pMoE" + f[-4:]

            with timeblock("****************\n time for predicting %s : " % f):
                # load mrc
                if f.endswith('.mrc'):
                    img, header = read_mrc_with_hd(f)
                    img = np.float32(img)
                # load tif
                elif f.endswith('.tif'):
                    img = tifffile.imread(f)
                    img = np.rot90(img, 3, axes=(1, 2)).astype(np.float32)
                else:
                    continue

                img /= np.max(img)
                restored_t = kernel_test_model_3d_VSR_swin_batch_softpadding(Net, "cuda", img, batch=batch,
                                                                             kernel=(opt.noise_crop, opt.noise_crop,
                                                                                     opt.input_frame),
                                                                             padding=(
                                                                             opt.noise_crop // 5, opt.noise_crop // 5,
                                                                             opt.input_frame // 5),
                                                                             output_head=opt.output_frame,
                                                                             upfactor=opt.upfactor,
                                                                             zupfactor=opt.zupfactor)  #
                restored_t = restored_t.numpy().astype(np.float32)

                if f.endswith('.mrc'):
                    write_mrc(save_name, restored_t, header)
                else:
                    restored_t=np.rot90(restored_t, 1, axes=(1, 2))
                    tifffile.imwrite(save_name, restored_t)

