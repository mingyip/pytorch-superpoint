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

dataset = ["draw_lines", "draw_polygon", "draw_multiple_polygons", "draw_ellipses", "draw_star"]
dtype = ["training", "test", "validation"]
dnum = [1000, 50, 20]

for data in dataset:
    for data_type, num in zip(dtype, dnum):

        
        ev_folder = Path("pytorch-superpoint/datasets/", data, "events", data_type)
        cm_folder = Path("pytorch-superpoint/datasets/", data, "combined", data_type)
        ev_folder.mkdir(parents=True, exist_ok=True)
        cm_folder.mkdir(parents=True, exist_ok=True)

        for idx in range(num):

            bag_folder = Path("pytorch-superpoint/datasets/", data, "images", data_type, "{:04d}".format(idx), "{:04d}.bag".format(idx))
            ev_idx_folder = Path(ev_folder, "{:04d}".format(idx))
            cm_idx_folder = Path(cm_folder, "{:04d}".format(idx))
            ev_idx_folder.mkdir(parents=True, exist_ok=True)
            cm_idx_folder.mkdir(parents=True, exist_ok=True)

            bag = rosbag.Bag(str(bag_folder))

            for i, (topic, msg, time) in enumerate(bag.read_messages(topics=['/cam0/events'])):

                print(data, " ", data_type, " ", idx, " ", i)
                img_path = Path("pytorch-superpoint/datasets/", data, "images", data_type, "{:04d}".format(idx), "{}.png".format(i))
                pts_path = Path("pytorch-superpoint/datasets/", data, "points", data_type, "{:04d}".format(idx), "{}.npy".format(i))

                print(img_path)
                img = cv.imread(str(img_path))
                bg = np.zeros_like(img)
                pts = np.load(str(pts_path))
        

                for e in msg.events:
                    if e.polarity:
                        img[e.y, e.x] = [0, 0, 255]
                        bg[e.y, e.x] = [0, 0, 255]
                    else:
                        img[e.y, e.x] = [255, 0, 0]
                        bg[e.y, e.x] = [255, 0, 0]


                for pt in pts:
                    cv.circle(img, (int(pt[1]), int(pt[0])), 1, (0, 255, 0), -1)

                cv.imwrite(str(Path(ev_idx_folder, "{}.png".format(i))), bg)
                cv.imwrite(str(Path(cm_idx_folder, "{}.png".format(i))), img)


