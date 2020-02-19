
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm


import os 
import cv2 as cv
import numpy as np


# path of the generated images directory
dataset = ["draw_lines", "draw_polygon", "draw_multiple_polygons", "draw_ellipses", "draw_star", "draw_checkerboard", "draw_stripes", "draw_cube"]
# dataset = ["draw_lines", "draw_polygon", "draw_multiple_polygons", "draw_ellipses", "draw_star", "draw_checkerboard", "draw_stripes"]
dtype = ["training", "test", "validation"]
dnum = [2000, 250, 100]



for t, data in tqdm(enumerate(dataset), leave=True):
    for data_type, num in zip(dtype, dnum):
        for idx in range(num):

            # print(data, data_type, idx)

            pt = Path("datasets", data, "points", data_type, "{:04d}".format(idx))
            img = Path("datasets", data, "images", data_type, "{:04d}".format(idx))
            linear = Path("datasets", data, "linear", data_type, "{:04d}".format(idx))
            
            # if len(os.listdir(pt)) != 6:
            #     print(pt, len(os.listdir(pt)))

            # if len(os.listdir(img)) != 8:
            #     print(img, len(os.listdir(img)))

            if len(os.listdir(linear)) != 5:
                print(linear, len(os.listdir(linear)))


print("Done")


# for t, data in tqdm(enumerate(dataset), leave=True):
# # for data in dataset:
#     for data_type, num in zip(dtype, dnum):

#         pt_folder = Path("repack/", data, "points", data_type)
#         ev_folder = Path("repack/", data, "events", data_type)
#         cm_folder = Path("repack/", data, "combined", data_type)

        
#         cm_folder.mkdir(parents=True, exist_ok=True)


#         for idx in range(num):

#             for i in range(10):

#                 pt_pth = Path(pt_folder, "{:04d}.npy".format(10*idx+i))
#                 ev_pth = Path(ev_folder, "{}.png".format(10*idx+i))
#                 cm_pth = Path(cm_folder, "{}.png".format(10*idx+i))

#                 # print("copy from ", point_source, " to ", point_target)
#                 if os.path.isfile(ev_pth):
#                     print(ev_pth)
#                     ev  = cv.imread(str(ev_pth), flags=cv.IMREAD_COLOR)
#                     print(ev.shape)
#                     pts = np.load(str(pt_pth))

#                     for pt in pts:
#                         cv.circle(ev, (int(pt[1]), int(pt[0])), 1, (0, 255, 0), -1)

                
#                 cv.imwrite(str(cm_pth), ev)
#                 # print(10 * idx + i)


        
#         # print(cm_folder)
