# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:31:36 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

rec_header_dtd = \
    [
        ("nx", "i4"),  # Number of columns
        ("ny", "i4"),  # Number of rows
        ("nz", "i4"),  # Number of sections

        ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),  # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),  # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),  # map column  1=x,2=y,3=z.
        ("mapr", "i4"),  # map row     1=x,2=y,3=z.
        ("maps", "i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),  # Minimum pixel value
        ("amax", "f4"),  # Maximum pixel value
        ("amean", "f4"),  # Mean pixel value

        ("ispg", "i4"),  # space group number (ignored by IMOD)
        ("next", "i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", "V20"),  # extra data (not used)
        ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", "i2"),
        ("norientation", "i2"),
        ("vd1", "i2"),  # vd1 = 100. * tilt increment
        ("vd2", "i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles1", "f4",2),  # 0,1,2 = original:  3,4,5 = current
        ('nt', 'i2'),
        ('ImgSequence', 'i2'),
        ("triangles2", "f4",3),


        ("xorg", "f4"),  # Origin of image
        ("yorg", "f4"),  # Origin of image
        ("zorg", "f4"),

        ("cmap", "S4"),  # Contains "MAP "
        ("stamp", "u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),  # RMS deviation of densities from mean density

        ("nlabl", "i4"),  # Number of labels with useful data
        ("labels", "S80", 10)  # 10 labels of 80 charactors
    ]

rec_header_dtd_b = \
    [
        ("nx", ">i4"),  # Number of columns
        ("ny", ">i4"),  # Number of rows
        ("nz", ">i4"),  # Number of sections

        ("mode", ">i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", ">i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", ">i4"),
        ("nzstart", ">i4"),

        ("mx", ">i4"),  # Grid size in X, Y and Z
        ("my", ">i4"),
        ("mz", ">i4"),

        ("xlen", ">f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", ">f4"),
        ("zlen", ">f4"),

        ("alpha", ">f4"),  # Cell angles - ignored by IMOD
        ("beta", ">f4"),
        ("gamma", ">f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", ">i4"),  # map column  1=x,2=y,3=z.
        ("mapr", ">i4"),  # map row     1=x,2=y,3=z.
        ("maps", ">i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", ">f4"),  # Minimum pixel value
        ("amax", ">f4"),  # Maximum pixel value
        ("amean", ">f4"),  # Mean pixel value

        ("ispg", ">i4"),  # space group number (ignored by IMOD)
        ("next", ">i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", ">i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", ">V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", ">i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", ">i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", ">V20"),  # extra data (not used)
        ("imodStamp", ">i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", ">i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", ">i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", ">i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", ">i2"),
        ("norientation", ">i2"),
        ("vd1", ">i2"),  # vd1 = 100. * tilt increment
        ("vd2", ">i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles1", ">f4",2),  # 0,1,2 = original:  3,4,5 = current
        ('nt', '>i2'),
        ('ImgSequence', '>i2'),
        ("triangles2", ">f4",3),

        ("xorg", ">f4"),  # Origin of image
        ("yorg", ">f4"), #nt i2
        ("zorg", ">f4"),

        ("cmap", ">S4"),  # Contains "MAP "
        ("stamp", ">u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", ">f4"),  # RMS deviation of densities from mean density

        ("nlabl", ">i4"),  # Number of labels with useful data
        ("labels", ">S80", 10)  # 10 labels of 80 charactors
    ]

def read_mrc(filename):
    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)
    bigFlag = False
    if header[0][3]>6:
        fd.seek(0)
        header = np.fromfile(fd, dtype=rec_header_dtd_b, count=1)
        bigFlag = True
    nx, ny, nz = np.int64(header['nx'][0]), np.int64(header['ny'][0]), np.int64(header['nz'][0])

    if header[0][3] == 1:
        data_type, read_type = 'int16', 'i2'
    elif header[0][3] == 2:
        data_type, read_type = 'float32', 'f4'
    elif header[0][3] == 6:
        data_type, read_type = 'uint16', 'u2'

    if bigFlag: read_type = '>' + read_type

    imgrawdata = np.fromfile(fd, read_type)
    img_data = np.ndarray(shape=(nz, nx, ny), dtype=data_type)
    fd.close()

    for iz in range(nz):
        img_data_2d=imgrawdata[nx*ny*iz:nx*ny*(iz+1)]
        img_data[iz, :, :]=img_data_2d.reshape(nx, ny, order='F')

    return img_data


def read_mrc_with_hd(filename):
    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)
    bigFlag = False
    if header[0][3]>6:
        fd.seek(0)
        header = np.fromfile(fd, dtype=rec_header_dtd_b, count=1)
        bigFlag = True
    nx, ny, nz = np.int64(header['nx'][0]), np.int64(header['ny'][0]), np.int64(header['nz'][0])

    if header[0][3] == 1:
        data_type, read_type = 'int16', 'i2'
    elif header[0][3] == 2:
        data_type, read_type = 'float32', 'f4'
    elif header[0][3] == 6:
        data_type, read_type = 'uint16', 'u2'

    if bigFlag: read_type = '>' + read_type

    imgrawdata = np.fromfile(fd, read_type)
    img_data = np.ndarray(shape=(nz, nx, ny), dtype=data_type)
    fd.close()

    for iz in range(nz):
        img_data_2d=imgrawdata[nx*ny*iz:nx*ny*(iz+1)]
        img_data[iz, :, :]=img_data_2d.reshape(nx, ny, order='F')

    return img_data, header


def write_mrc(filename, img_data, header):
    if img_data.dtype == 'int16':
        header[0][3] = 1
    elif img_data.dtype == 'float32':
        header[0][3] = 2
    elif img_data.dtype == 'uint16':
        header[0][3] = 6

    nz, nx, ny = img_data.shape
    header['nx'][0] = nx
    header['ny'][0] = ny
    header['nz'][0] = nz
    nx = np.int64(nx)
    ny = np.int64(ny)
    nz = np.int64(nz)

    fd = open(filename, 'wb')
    for i in range(len(rec_header_dtd)):
        header[rec_header_dtd[i][0]].tofile(fd)

    imgrawdata = np.ndarray(shape=(nx*ny*nz), dtype=img_data.dtype)
    for iz in range(nz):
        imgrawdata[nx*ny*iz:nx*ny*(iz+1)] = img_data[iz,:,:].reshape(nx*ny, order='F')
    imgrawdata.tofile(fd)

    fd.close()
    return


if __name__ == '__main__':
    filename = '/mnt/TLdata/isoSIM/230815_MT_Z_fix_objZ/Cell3_L_Z_405_65%_5ms_488_mW_0.23mW_0.8mW_0.8mW_25ms_20ms_20ms_20230815_135659/isoSIM_Z_3Ang6Ph_405_488_Cyc1_Ch1_St3_ROI_piPhaseShift_rmCamPat_add4_SIM_DCsup0.3.mrc'
    img_data, header = read_mrc_with_hd(filename)

    # plt.figure(figsize=(16, 8))
    # plt.imshow(img_data[:, :, 0], plt.cm.gray)
    # plt.axis('off')
    # plt.show()

    savename = '/home/li-lab/liukan/Datasets/DL_denoising/2_IBP_20181208_beads/tmp/TIRF488_Cyc2_Ch2_St1.mrc'
    write_mrc(savename, img_data, header)