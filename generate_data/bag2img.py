
import os, sys
import shutil
import rosbag
import rospy


import cv2 as cv
import numpy as np
# import time as timer


from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

# path:        /media/pytorch-superpoint/datasets/draw_lines/images/training/0010/0010.bag
# version:     2.0
# duration:    0.0s
# start:       Jan 01 1970 00:00:00.00 (0.00)
# end:         Jan 01 1970 00:00:00.01 (0.01)
# size:        122.3 KB
# messages:    12
# compression: none [1/1 chunks]
# types:       dvs_msgs/EventArray    [5e8beee5a6c107e504c2e78903c224b8]
#              sensor_msgs/CameraInfo [c9a58c1b0b154e0e6da7578cb991d214]
#              sensor_msgs/Image      [060021388200f6f0f447d0fcd9c64743]
# topics:      /cam0/camera_info    1 msg            : sensor_msgs/CameraInfo
#              /cam0/events        10 msgs @ 1.2 kHz : dvs_msgs/EventArray   
#              /cam0/image_raw      1 msg            : sensor_msgs/Image



# topics = bag.get_type_and_topic_info()[1].keys()
# types = []

# for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
#     types.append(bag.get_type_and_topic_info()[1].values()[i][0])
# print(types)

dataset = ["draw_lines", "draw_polygon", "draw_multiple_polygons", "draw_ellipses", "draw_star", "draw_checkerboard", "draw_stripes", "draw_cube"]




def read_bag(H, W, iteration, frames):

    for d in dataset:

        idx = 0

        bag = rosbag.Bag(f"/media/gen/data/{d}/{d}.bag", 'r')
        evt_path = Path(f"/media/gen/data/{d}/events")
        pnt_path = Path(f"/media/gen/data/{d}/points")
        chk_path = Path(f"/media/gen/data/{d}/events_check")

        evt_path.mkdir(parents=True, exist_ok=True)
        pnt_path.mkdir(parents=True, exist_ok=True)
        chk_path.mkdir(parents=True, exist_ok=True)

        for i, (topic, msg, time) in enumerate(tqdm(bag.read_messages(topics=['/cam0/events']), desc=d, total=iteration*frames)):

            img = np.zeros((H, W, 3))


            for j, e in enumerate(msg.events):
                img[e.y, e.x, e.polarity * 2] = 255


            if (i + 1) % frames != 0:

                shutil.move("/media/gen/data/{}/raw_points/pnt_{:05d}.npy".format(d, i), 
                            "/media/gen/data/{}/points/pnt_{:05d}.npy".format(d, idx))

                cv.imwrite("/media/gen/data/{}/events/{:05d}.png".format(d, idx), img)
                

                if idx % 53 == 0:
                    pts = np.load("/media/gen/data/{}/points/pnt_{:05d}.npy".format(d, idx))
                    for pt in pts:
                        cv.circle(img, (int(pt[0]), int(pt[1])), 3, [0, 255, 0], -1)
                    chk_ = str(Path(chk_path, "chk_{:05d}.png".format(idx)))
                    cv.imwrite(chk_, img)


                idx += 1
        bag.close()





if __name__ == "__main__":
    
    img_size = (160, 120)
    iteration = 550
    frames = 20



    parser = ArgumentParser()
    parser.add_argument("-H", "--height", dest="height",
                        help="Original Height of a generated image")
    parser.add_argument("-W", "--width", dest="width",
                        help="Original Width of a generated image")
    parser.add_argument("-i", "--iteration", dest="iteration",
                        help="How many iterations for each shape batch")
    parser.add_argument("-f", "--frames", dest="frames",
                        help="Frames per batch")
    args = parser.parse_args()



    H, W = int(args.height), int(args.width)
    iteration = int(args.iteration)
    frames = int(args.frames)



    print("======================================")
    print("Compose Events Data")
    print("======================================")
    print("Original image size: ", H, W)
    print("Iteration: ", iteration)
    print("Frames: ", frames)
    print("Total images: ", frames*iteration)
    print("======================================")



    read_bag(H, W, iteration, frames)
    

