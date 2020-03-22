"""
Adapted from https://github.com/rpautrat/SuperPoint/blob/master/superpoint/datasets/synthetic_dataset.py

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import torch.utils.data as data
import torch
import numpy as np
from imageio import imread

# from os import path as Path
import tensorflow as tf
from pathlib import Path
import tarfile

# import os
import random
import logging

from utils.tools import dict_update

from datasets import synthetic_dataset

# from models.homographies import sample_homography

from tqdm import tqdm
import cv2
import shutil
from settings import DEBUG as debug
from settings import DATA_PATH
from settings import SYN_TMPDIR

# DATA_PATH = '.'
import multiprocessing

TMPDIR = SYN_TMPDIR  # './datasets/' # you can define your tmp dir


def load_as_float(path):
    return imread(path).astype(np.float32) / 255


class SyntheticDataset_gaussian(data.Dataset):
    """
    """

    default_config = {
        "primitives": "all",
        "truncate": {},
        "validation_size": -1,
        "test_size": -1,
        "on-the-fly": False,
        "cache_in_memory": False,
        "suffix": None,
        "add_augmentation_to_test_set": False,
        "num_parallel_calls": 10,
        "generation": {
            "split_sizes": {"training": 2000, "validation": 100, "test": 250},
            # "image_size": [960, 1280],
            "image_size": [120, 160],
            "random_seed": 0,
            "num_frames": 5,
            "params": {
                "generate_background": {
                    "min_kernel_size": 150,
                    "max_kernel_size": 500,
                    "min_rad_ratio": 0.02,
                    "max_rad_ratio": 0.031,
                },
                "draw_stripes": {"transform_params": (0.1, 0.1)},
                "draw_multiple_polygons": {"kernel_boundaries": (50, 100)},
            },
        },
        "preprocessing": {"resize": [240, 320], "blur_size": 11,},
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0,},
        },
    }

    # debug = True

    if debug == True:
        drawing_primitives = [
            "draw_checkerboard",
        ]
    else:
        drawing_primitives = [
            "draw_lines",
            "draw_polygon",
            "draw_multiple_polygons",
            "draw_ellipses",
            "draw_star",
            "draw_checkerboard",
            "draw_stripes",
            "draw_cube",
            # "gaussian_noise",
        ]
    print(drawing_primitives)

    """
    def dump_primitive_data(self, primitive, tar_path, config):
        pass
    """

    def dump_primitive_data(self, primitive, tar_path, config):
        # temp_dir = Path(os.environ['TMPDIR'], primitive)
        temp_dir = Path(TMPDIR, primitive)

        tf.logging.info("Generating tarfile for primitive {}.".format(primitive))
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )


        image_size = config["generation"]["image_size"]
        bg_config = config["generation"]["params"]["generate_background"]
        frames = config["generation"]["num_frames"]
        primitive_config = config["generation"]["params"].get(primitive, {})


        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir, chk_dir = [Path(temp_dir, i, split) for i in ["images", "points", "check"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)
            chk_dir.mkdir(parents=True, exist_ok=True)



            count = 0
            for i in tqdm(range(size), desc=split, leave=False):

                pts_list, img_list = getattr(synthetic_dataset, primitive)(image_size,
                                                                frames,
                                                                bg_config,
                                                                **primitive_config
                                                            )

                # old_pts = None

                for j, (pts, img) in enumerate(zip(pts_list, img_list)):
                    if j > frames:
                        break

                    check_img = np.copy(img)
                    for pt in pts:
                        cv2.circle(check_img, (pt[0], pt[1]), 20, (0, 255, 0), -1)

                    pts = np.flip(pts, 1)  # reverse convention with opencv

                    
                    b = config["preprocessing"]["blur_size"]
                    img = cv2.GaussianBlur(img, (b, b), 0)

                    pts = (
                        pts
                        * np.array(config["preprocessing"]["resize"], np.float)
                        / np.array(config["generation"]["image_size"], np.float)
                    )
                    img = cv2.resize(
                        img,
                        tuple(config["preprocessing"]["resize"][::-1]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    im_folder = Path(im_dir, "{:04d}".format(i))
                    pts_folder = Path(pts_dir, "{:04d}".format(i))
                    chk_folder = Path(chk_dir, "{:04d}".format(i))
                    im_folder.mkdir(parents=True, exist_ok=True)
                    pts_folder.mkdir(parents=True, exist_ok=True)
                    chk_folder.mkdir(parents=True, exist_ok=True)

                    check_img = cv2.putText(check_img, str(len(pts)), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 

                    cv2.imwrite(str(Path(im_folder, "{:02d}.png".format(j))), img)
                    cv2.imwrite(str(Path(chk_folder, "{:02d}.png".format(j))), check_img)
                    np.save(Path(pts_folder, "{:02d}.npy".format(j)), pts)

                    # old_pts = pts


        # Pack into a tar file
        tar = tarfile.open(tar_path, mode="w:gz")
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        # shutil.rmtree(temp_dir)
        tf.logging.info("Tarfile dumped to {}.".format(tar_path))


    def parse_primitives(self, names, all_primitives):
        p = (
            all_primitives
            if (names == "all")
            else (names if isinstance(names, list) else [names])
        )
        assert set(p) <= set(all_primitives)
        return p

    def __init__(
        self,
        seed=None,
        task="train",
        sequence_length=3,
        transform=None,
        target_transform=None,
        getPts=False,
        warp_input=False,
        **config,
    ):
        from utils.homographies import sample_homography_np as sample_homography
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import compute_valid_mask
        from utils.utils import inv_warp_image, warp_points

        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, dict(config))

        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform

        ######
        self.enable_photo_train = self.config["augmentation"]["photometric"]["enable"]
        self.enable_homo_train = self.config["augmentation"]["homographic"]["enable"]
        self.enable_homo_val = False
        self.enable_photo_val = False
        ######


        if task == "train":
            self.action = "training"
            self.num_data = self.config["generation"]["split_sizes"]["training"]
        else:
            self.action = "validation"
            self.num_data = self.config["generation"]["split_sizes"]["validation"]
        

        self.cell_size = 8
        self.getPts = getPts

        self.gaussian_label = False
        if self.config["gaussian_label"]["enable"]:
            # self.params_transform = {'crop_size_y': 120, 'crop_size_x': 160, 'stride': 1, 'sigma': self.config['gaussian_label']['sigma']}
            self.gaussian_label = True

        self.pool = multiprocessing.Pool(6)

        # Parse drawing primitives
        primitives = self.parse_primitives(
            config["primitives"], self.drawing_primitives
        )

        basepath = Path(
            DATA_PATH,
            "synthetic_shapes"
            + ("_{}".format(config["suffix"]) if config["suffix"] is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {"images": [], "points": []} for s in [self.action]}

        # for primitive in primitives:
        #     tar_path = Path(basepath, "{}.tar.gz".format(primitive))
        #     if not tar_path.exists():
        #         self.dump_primitive_data(primitive, tar_path, self.config)

            # # Untar locally
            # logging.info("Extracting archive for primitive {}.".format(primitive))
            # logging.info(f"tar_path: {tar_path}")
            # tar = tarfile.open(tar_path)
            # temp_dir = Path(os.environ['TMPDIR'])
            # tar.extractall(path=temp_dir)
            # tar.close()


        # # Gather filenames in all splits, optionally truncate
        # for primitive in primitives:
        #     temp_dir = Path(TMPDIR)
        #     truncate = self.config["truncate"].get(primitive, 1)
        #     path = Path(temp_dir, primitive)
        #     for s in splits:
        #         e = [str(p) for p in Path(path, "images", s).iterdir()]
        #         f = [p.replace("images", "points") for p in e]
        #         f = [p.replace(".png", ".npy") for p in f]
        #         splits[s]["images"].extend(e[: int(truncate * len(e))])
        #         splits[s]["points"].extend(f[: int(truncate * len(f))])

        # # Shuffle
        # for s in splits:
        #     perm = np.random.RandomState(0).permutation(len(splits[s]["images"]))
        #     for obj in ["images", "points"]:
        #         splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        # self.crawl_folders(splits)

    def crawl_folders(self, splits):
        sequence_set = []
        for (img, pnts) in zip(
            splits[self.action]["images"], splits[self.action]["points"]
        ):
            sample = {"image": img, "points": pnts}
            sequence_set.append(sample)
        self.samples = sequence_set


    # def putGaussianMaps_par(self, center):
    #     crop_size_y = self.params_transform['crop_size_y']
    #     crop_size_x = self.params_transform['crop_size_x']
    #     stride = self.params_transform['stride']
    #     sigma = self.params_transform['sigma']

    #     grid_y = crop_size_y / stride
    #     grid_x = crop_size_x / stride
    #     start = stride / 2.0 - 0.5
    #     xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)))
    #     xx = xx * stride + start
    #     yy = yy * stride + start
    #     d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    #     exponent = d2 / 2.0 / sigma / sigma
    #     mask = exponent <= sigma
    #     cofid_map = np.exp(-exponent)
    #     cofid_map = np.multiply(mask, cofid_map)
    #     return cofid_map

    def putGaussianMaps(self, center, accumulate_confid_map):
        crop_size_y = self.params_transform["crop_size_y"]
        crop_size_x = self.params_transform["crop_size_x"]
        stride = self.params_transform["stride"]
        sigma = self.params_transform["sigma"]

        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)))
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= sigma
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
        return accumulate_confid_map

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

        def imgPhotometric(img):
            """
            :param img:
                numpy (H, W)
            :return:
            """
            augmentation = self.ImgAugTransform(**self.config["augmentation"])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config["augmentation"])
            return img


        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy

        sample = {}
        H, W = self.config["generation"]["image_size"]
        imgs, pnts, evts = synthetic_dataset.generate_random_shape((H, W), 5, None)
        idx = np.random.randint(1000)


        # Only take the last set of points
        pnts = torch.tensor(pnts[-1]).float()
        pnts = torch.stack((pnts[:, 1], pnts[:, 0]), dim=1)  # (x, y)
        pnts = filter_points(pnts, torch.tensor([H, W]))
        pnts_long = pnts.round().long()


        labels = torch.zeros(H, W)
        labels[pnts_long[:, 0], pnts_long[:, 1]] = 1
        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))


        for i, evt in enumerate(evts):
            evts[i, 0] = imgPhotometric(evt[0, :, :]).squeeze()
            evts[i, 1] = imgPhotometric(evt[1, :, :]).squeeze()
        evts = torch.from_numpy(evts.astype(np.float32))


        # sample.update({"images": imgs})
        sample.update({"valid_mask": valid_mask})
        sample.update({"labels_2D": labels.unsqueeze(0)})
        # sample.update({"points": pnts})
        sample.update({"events": evts})


        return sample

        # print("valid_mask: ", valid_mask.dtype, valid_mask.shape)
        # print("labels: ", labels.dtype, labels.shape)
        # print("points: ", pnts.dtype, pnts.shape)
        # print("events: ", evts.dtype, evts.shape)
        
        # # Output event images
        # for i, evt in enumerate(evts):
        #     tmp = np.zeros((H, W, 3))
        #     tmp[:, :, 0] = evt[0, :, :] * 255
        #     tmp[:, :, 2] = evt[1, :, :] * 255
        #     for pt in pnts:
        #         cv2.circle(tmp, (int(pt[1].numpy()), int(pt[0].numpy())), 3, (0, 255, 0), -1)
        #     cv2.imwrite("temp/img{}_{}.png".format(idx, i), tmp)


        # # Output label images
        # tmp = labels.numpy()
        # print(tmp.shape, tmp.dtype)
        # for pt in pnts:
        #     cv2.circle(tmp, (int(pt[0].numpy()), int(pt[0].numpy())), 3, (0, 255, 0), -1)
        # cv2.imwrite("temp/labels_{}.png".format(idx), tmp * 255)

        # raise



        

    def __len__(self):
        return self.num_data

    ## util functions
    def gaussian_blur(self, image):
        """
        image: np [H, W]
        return:
            blurred_image: np [H, W]
        """
        aug_par = {"photometric": {}}
        aug_par["photometric"]["enable"] = True
        aug_par["photometric"]["params"] = self.config["gaussian_label"]["params"]
        augmentation = self.ImgAugTransform(**aug_par)
        # get label_2D
        # labels = points_to_2D(pnts, H, W)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()

