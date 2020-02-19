
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm


import os 




# path of the generated images directory
# dataset = ["draw_lines", "draw_polygon", "draw_multiple_polygons", "draw_ellipses", "draw_star", "draw_checkerboard", "draw_stripes", "draw_cube"]
dataset = ["draw_cube"]
dtype = ["training", "test", "validation"]
dnum = [2000, 250, 100]
nFrames = 5



for t, data in tqdm(enumerate(dataset), leave=True):
# for data in dataset:
    for data_type, num in zip(dtype, dnum):


        source_pt_folder = Path("datasets/", data, "points", data_type)
        source_ev_folder = Path("datasets/", data, "linear", data_type)


        target_pt_folder = Path("repack/", data, "points", data_type)
        target_ev_folder = Path("repack/", data, "images", data_type)

        
        target_pt_folder.mkdir(parents=True, exist_ok=True)
        target_ev_folder.mkdir(parents=True, exist_ok=True)


        for idx in range(num):

            pt_img_folder = Path(source_pt_folder, "{:04d}".format(idx))
            ev_img_folder = Path(source_ev_folder, "{:04d}".format(idx))

            for i in range(nFrames):

                point_source = Path(pt_img_folder, "{:02d}.npy".format(i))
                point_target = Path(target_pt_folder, "{:04d}.npy".format(nFrames*idx+i))

                event_source = Path(ev_img_folder, "{}.png".format(i))
                event_target = Path(target_ev_folder, "{}.png".format(nFrames*idx+i)) 


                # print("copy from ", point_source, " to ", point_target)
                # if os.path.isfile(event_source):
                copyfile(event_source, event_target)
                copyfile(point_source, point_target)

                # print(10 * idx + i)


        
        # print(cm_folder)
