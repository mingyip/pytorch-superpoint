import rosbag
import rospy

import cv2 as cv
import numpy as np

from pathlib import Path

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
dtype = ["training", "test", "validation"]
dnum = [2000, 250, 100]



# bag = rosbag.Bag("datasets/draw_lines/images/training/0135/0135.bag")

# for i, (topic, msg, time) in enumerate(bag.read_messages(topics=['/cam0/events'])):
#     print(i)

# raise


for data in dataset:
    for data_type, num in zip(dtype, dnum):
        for idx in range(num):


            print(data, data_type, idx)


            bag_folder = Path("datasets/", data, "images", data_type, "{:04d}".format(idx), "{:04d}.bag".format(idx))
            ev_folder = Path("datasets/", data, "events", data_type, "{:04d}".format(idx))
            cm_folder = Path("datasets/", data, "combined", data_type, "{:04d}".format(idx))
            gray_folder = Path("datasets/", data, "gray", data_type, "{:04d}".format(idx))
            linear_decay_folder = Path("datasets/", data, "linear", data_type, "{:04d}".format(idx))
            exponential_decay_folder = Path("datasets/", data, "expo", data_type, "{:04d}".format(idx))

            ev_folder.mkdir(parents=True, exist_ok=True)
            cm_folder.mkdir(parents=True, exist_ok=True)
            gray_folder.mkdir(parents=True, exist_ok=True)
            linear_decay_folder.mkdir(parents=True, exist_ok=True)
            exponential_decay_folder.mkdir(parents=True, exist_ok=True)

            bag = rosbag.Bag(str(bag_folder))


            start_nsec = 0
            for i, (topic, msg, time) in enumerate(bag.read_messages(topics=['/cam0/events'])):

                # print(data, " ", data_type, " ", idx, " ", i)
                img_path = Path("datasets/", data, "images", data_type, "{:04d}".format(idx), "{:02d}.png".format(i))
                pts_path = Path("datasets/", data, "points", data_type, "{:04d}".format(idx), "{:02d}.npy".format(i))

                # print(img_path)
                img = cv.imread(str(img_path))
                bg = np.zeros_like(img)
                linear = np.zeros_like(img)
                exponential = np.zeros_like(img)
                pts = np.load(str(pts_path))
     

                total_time = time.to_nsec() - start_nsec
                decay_constant = -0.000001


                for j, e in enumerate(msg.events):
                    t = e.ts.to_nsec() - start_nsec
                    linear_decay = (t / total_time) * 255
                    exponential_decay = np.exp(decay_constant * t) * 255

                    if e.polarity:
                        img[e.y, e.x] = [0, 0, 255]
                        bg[e.y, e.x] = [0, 0, 255]
                        linear[e.y, e.x] = [0, 0, linear_decay]
                        exponential[e.y, e.x] = [0, 0, exponential_decay]
                    else:
                        img[e.y, e.x] = [255, 0, 0]
                        bg[e.y, e.x] = [255, 0, 0]
                        linear[e.y, e.x] = [linear_decay * 255, 0, 0]
                        exponential[e.y, e.x] = [exponential_decay * 255, 0, 0]


                for pt in pts:
                    cv.circle(img, (int(pt[1]), int(pt[0])), 1, (0, 255, 0), -1)

                gray = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)
                linear = cv.cvtColor(linear, cv.COLOR_BGR2GRAY)
                exponential = cv.cvtColor(exponential, cv.COLOR_BGR2GRAY)

                cv.imwrite(str(Path(ev_folder, "{}.png".format(i))), bg)
                cv.imwrite(str(Path(cm_folder, "{}.png".format(i))), img)
                cv.imwrite(str(Path(gray_folder, "{}.png".format(i))), gray)
                cv.imwrite(str(Path(linear_decay_folder, "{}.png".format(i))), linear)
                cv.imwrite(str(Path(exponential_decay_folder, "{}.png".format(i))), exponential)

                start_nsec = time.to_nsec()




