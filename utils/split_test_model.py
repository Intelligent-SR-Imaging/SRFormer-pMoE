import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import tifffile
import os


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def getmask3d(kernel, padding, factor):
    mask1d = []
    for l, p, f in zip(kernel, padding, factor):
        l *= f
        p *= f
        tt = torch.ones(l)
        tt[0:p // 2] = 0
        tt[-p // 2:] = 0
        tt[p // 2:p // 2 + p] = torch.arange(0, 1, 1 / p)
        tt[l - p // 2 - p:l - p // 2] = torch.arange(1, 0, -1 / p)
        mask1d.append(tt)
    z, x, y = torch.meshgrid(mask1d[0], mask1d[1], mask1d[2])
    mask = z * x * y
    return mask

def add_kernel(in_,out, mask,pos, kx,ky,kz,zupfactor,upfactor):
    out[pos[0] * zupfactor:(pos[0] + kz) * zupfactor,
    pos[1] * upfactor:(pos[1] + kx) * upfactor,
    pos[2] * upfactor:(pos[2] + ky) * upfactor] += in_ * mask  # differ
    return out

def kernel_test_model_3d_VSR_swin_batch_softpadding(model, device, img, batch, kernel, padding, output_head=3, upfactor=1,
                                        zupfactor=1,half=False):

    nz, nx, ny = img.shape
    # if nz == 1:
    #     return
    kx, ky, kz = kernel
    px, py, padding_output = padding
    pz = (kz - output_head) // 2 + padding_output
    # if padding == 0:
    #     padding = kernel // 8
    embed_x = kx - 2 * px
    embed_y = ky - 2 * py
    embed_z = kz - 2 * pz
    # assert embed_z == output_head

    split_x = math.ceil(nx / embed_x)
    split_y = math.ceil(ny / embed_y)
    split_z = math.ceil(nz / embed_z)

    margin_x = split_x * embed_x + 2 * px
    margin_y = split_y * embed_y + 2 * py
    margin_z = split_z * embed_z + 2 * pz
    img_margin = np.zeros((margin_z, margin_x, margin_y))

    # fill in frame with zero
    for f_i in range(pz):
        img_margin[f_i, px:px + nx, py:py + ny] = img[0]
    for f_i in range(pz + nz, margin_z):
        img_margin[f_i, px:px + nx, py:py + ny] = img[nz - 1]
    img_margin[pz:pz + nz, px:px + nx, py:py + ny] = img

    output = torch.zeros((np.ceil(margin_z * zupfactor).astype(np.int16), margin_x * upfactor, margin_y * upfactor))

    mask = getmask3d((kz, kx, ky), (pz, px, py), (zupfactor, upfactor, upfactor))
    mask_out = torch.zeros(output.shape)

    pos = []
    for i in range(split_x):
        for j in range(split_y):
            for k in range(split_z):
                pos.append((k * embed_z, i * embed_x, j * embed_y))
    print("predict tile:%d" % (len(pos)))
    test_len = np.int16(math.ceil(len(pos) / batch))
    for i in range(test_len):
        pos_i = pos[i * batch:i * batch + batch]

        input_ = torch.zeros(len(pos_i), kz, kx, ky)
        for j in range(len(pos_i)):
            input_[j] = torch.FloatTensor(img_margin[pos_i[j][0]:pos_i[j][0] + kz
                                          , pos_i[j][1]:pos_i[j][1] + kx
                                          , pos_i[j][2]:pos_i[j][2] + ky])
        if half:
            input_ = input_.half()
        with torch.no_grad():
            out = model(input_.cuda()).cpu()


        assert out[0].shape == mask.shape
        for j in range(len(pos_i)):
            output = add_kernel(out[j],output,mask,pos_i[j],kx, ky, kz, zupfactor, upfactor)
            mask_out = add_kernel(1, mask_out, mask, pos_i[j], kx, ky, kz, zupfactor, upfactor)

    output = output[pz * zupfactor:(pz + nz) * zupfactor, px * upfactor:(px + nx) * upfactor,
           py * upfactor:(py + ny) * upfactor]
    mask_out = mask_out[pz * zupfactor:(pz + nz) * zupfactor, px * upfactor:(px + nx) * upfactor,
           py * upfactor:(py + ny) * upfactor]
    output = torch.div(output,mask_out)
    output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)

    return output
