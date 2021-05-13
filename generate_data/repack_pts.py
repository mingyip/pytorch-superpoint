
import os, sys
import shutil
# import rosbag
# import rospy


import cv2 as cv
import numpy as np
# import time as timer


from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

dataset = ["draw_lines", "draw_polygon", "draw_multiple_polygons", "draw_ellipses", "draw_star", "draw_checkerboard", "draw_stripes", "draw_cube"]


def repack_pts(frames, dataset_):

    if dataset_:
        dataset = [dataset_]

    for d in dataset:
        arr = os.listdir(f"/media/gen/data/{d}/raw_points/")
        # arr.sort()
        
        idx = 0
        pnt_path = Path(f"/media/gen/data/{d}/points")
        chk_path = Path(f"/media/gen/data/{d}/events_check")
        pnt_path.mkdir(parents=True, exist_ok=True)
        chk_path.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(arr)), desc=d):

            if (i + 1) % frames != 0:

                shutil.move("/media/gen/data/{}/raw_points/pnt_{:05d}.npy".format(d, i), 
                            "/media/gen/data/{}/points/pnt_{:05d}.npy".format(d, idx))

                if idx % 53 == 0:
                    img = cv.imread("/media/gen/data/{}/events/{:05d}.png".format(d, idx))
                    pts = np.load("/media/gen/data/{}/points/pnt_{:05d}.npy".format(d, idx))
                    for pt in pts:
                        cv.circle(img, (int(pt[0]), int(pt[1])), 3, [0, 255, 0], -1)
                    chk_ = str(Path(chk_path, "chk_{:05d}.png".format(idx)))
                    cv.imwrite(chk_, img)

                idx += 1

        os.remove(f"/media/gen/data/{d}/{d}.bag")


if __name__ == "__main__":
    
    img_size = (160, 120)
    iteration = 550
    frames = 20



    parser = ArgumentParser()
    parser.add_argument("-f", "--frames", dest="frames", default=6, type=int,
                        help="Frames per batch")
    parser.add_argument("-d", "--draw_function", dest="draw_function",
                        help="Select a draw function")
    args = parser.parse_args()


    # print("======================================")
    # print("Compose Events Data")
    # print("Frames: ", frames)
    # print("======================================")

    repack_pts(args.frames, args.draw_function)
    

