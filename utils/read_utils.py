import cv2
import numpy as np
import os
from libtiff import TIFF
import tifffile
from skimage import io
# from PIL import Image
from read_utils.read_mrc import read_mrc_with_hd, read_mrc


def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])


def is_specify_file(filename, ext):
    return any(filename.endswith(extension) for extension in [ext])


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def load_tif_img(filepath, with_neg=True):
    img = io.imread(filepath)
    # img = cv2.imread(filepath, -1)
    img = img.astype(np.float32)
    # if img.ndim == 3:
    #     img = img.transpose([2, 0, 1])
    if with_neg:
        img = load_recon_both(img)
    else:
        img = load_recon_no_negative(img)
    return img


def load_tif_img_origin(filepath):
    img = tifffile.imread(filepath)
    # img = cv2.imread(filepath, -1)
    img = img.astype(np.float32)
    # if not is3D and img.ndim == 3:
    #     img = img[0]
    return img

def load_tif_2d_img_origin(filepath):
    img = tifffile.imread(filepath)
    # img = cv2.imread(filepath, -1)
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    return img


def normalize(img, max, min):
    img_max = np.max(img)
    img_min = np.min(img)
    img = (img - img_min) / (img_max - img_min) * (max - min) + min
    return img


def input3d_guass_blur(img):
    nz, _, _ = img.shape
    for i in range(nz):
        img[i] = cv2.GaussianBlur(img[i], (3, 3), 0.6, 0.6)
    return img


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_tiff(path, img):
    # cv2.imwrite(path, img)

    tif = TIFF.open(path, 'w')
    tif.write_image(img, compression=None)
    tif.close()


def save_nz_tiff(path, img):
    nz, _, _ = img.shape
    save_path = os.path.splitext(path)[0] + '.tif'
    tif = TIFF.open(save_path, 'w')
    for i in range(nz):
        tif.write_image(img[i], compression=None)
    tif.close()


### load diff needs for test img input

def load_mrc_dataset(filename, with_neg):
    img, _ = read_mrc_with_hd(filename)
    img = np.float32(img)
    if with_neg:
        img = load_recon_both(img)
    else:
        img = load_recon_no_negative(img)
    return img[0]


def load_mrc_dataset_origin(filename):
    img, _ = read_mrc_with_hd(filename)
    img = np.float32(img)
    return img[0]


def load_3d_mrc_dataset(filename, with_neg):
    img, _ = read_mrc_with_hd(filename)
    img = np.float32(img)
    # nz, _, _ = img.shape
    # for i in range(nz):
    if with_neg:
        img = load_recon_both(img)
    else:
        img = load_recon_no_negative(img)
    return img


def load_3d_mrc_dataset_origin(filename):
    img, _ = read_mrc_with_hd(filename)
    img = np.float32(img)
    # nz, _, _ = img.shape
    # for i in range(nz):
    return img


def load_3d_mrc_dataset_normal_frame(filename, with_neg):
    img, _ = read_mrc_with_hd(filename)
    img = np.float32(img)
    nz, _, _ = img.shape
    for i in range(nz):
        if with_neg:
            img[i] = load_recon_both(img[i])
        else:
            img[i] = load_recon_no_negative(img[i])
    return img

def load_origin_img(path):
    if is_specify_file(path, '.mrc'):
        img = read_mrc(path)
    elif is_specify_file(path, '.tif'):
        img = tifffile.imread(path)
    else:
        img = io.imread(path)
    return img

def normalize_recon(img):
    img_max = np.max(np.abs(img))
    img /= img_max
    return img


def load_recon_both(img):
    img_max = np.max(np.abs(img))
    img /= img_max
    return img


def load_recon_no_negative(img):
    img = np.where(img < 0, 0, img)
    img_max = np.max(np.abs(img))
    img /= img_max
    return img


def load_recon_with_normal(img):
    img = np.where(img < 0, 0, img)
    img = normalize(img, 1, 0)
    return img


def load_recon_no_negative_median_intensity(img, clamp_low):
    img = normalize(np.where(img < 0, 0, img), 1, 0)
    length = img.shape[0] * img.shape[1] * img.shape[2]
    i = np.reshape(img, length)
    maxi = np.max(i)
    clampi = 0
    while (i[clampi] <= clamp_low * maxi):
        clampi += 1
    img /= np.median(i[clampi:length - 1]) * 6
    return img


def stretch(img, min, max):
    img = np.where(img < min, min, img)
    img = np.where(img > max, max, img)
    img = (img - min) / (max - min)
    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    root = '../../data/SIM/45nm_bead_20220409'
    img_name = 'TIRF560_cam2_step1_001_L.tif'
    f = '560-45nm-1x_20220409_112844'

    name_list = os.listdir(root)
    clamp = [15, 20, 25, 30]  # 20
    for c in clamp:
        img = load_recon_no_negative(load_tif_img(os.path.join(root, f, img_name)))
        median = np.median(img)
        mean = np.mean(img)
        print("median:{} mean:{}".format(median, mean))
        img = stretch(img, 0, mean * c)

        plt.figure()
        plt.imshow(img, 'gray')
    plt.show()
