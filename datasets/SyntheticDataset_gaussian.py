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
# import time
import random
import logging

from utils.tools import dict_update

from datasets import synthetic_dataset
from utils.utils import writeimage


# from models.homographies import sample_homography

from tqdm import tqdm
import cv2
import shutil
from settings import DEBUG as debug
from settings import DATA_PATH
from settings import SYN_TMPDIR

import multiprocessing

TMPDIR = SYN_TMPDIR  # './datasets/' # you can define your tmp dir



class SyntheticDataset_gaussian(data.Dataset):
    """
    """

    default_config = {
        "primitives": "all",
        "truncate": {},
        "on_the_fly": False,
        "cache_in_memory": False,
        "suffix": None,
        "add_augmentation_to_test_set": False,
        "generation": {
            "split_sizes": {"training": 100, "validation": 2, "test": 5},
            "image_size": [960, 1280],
            "random_seed": 0,
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
            "gaussian_noise",
        ]

    """
    def dump_primitive_data(self, primitive, tar_path, config):
        pass
    """

    def dump_primitive_data(self, primitive, tar_path, config):
        temp_dir = Path(TMPDIR, primitive)

        tf.logging.info("Generating tarfile for primitive {}.".format(primitive))
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config["generation"]["image_size"],
                    **config["generation"]["params"]["generate_background"],
                )
                points = np.array(
                    getattr(synthetic_dataset, primitive)(
                        image, **config["generation"]["params"].get(primitive, {})
                    )
                )
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                    points
                    * np.array(config["preprocessing"]["resize"], np.float)
                    / np.array(config["generation"]["image_size"], np.float)
                )
                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode="w:gz")
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        tf.logging.info("Tarfile dumped to {}.".format(tar_path))

    def parse_primitives(self, names, all_primitives):
        p = (
            all_primitives
            if (names == "all")
            else (names if isinstance(names, list) else [names])
        )
        assert set(p) <= set(all_primitives)
        return p

    def crawl_folders(self, splits):
        sequence_set = []
        for (img, pnts) in zip(
            splits[self.action]["images"], splits[self.action]["points"]
        ):
            sample = {"image": img, "points": pnts}
            sequence_set.append(sample)
        self.samples = sequence_set

    def __init__(
        self,
        seed=None,
        task="train",
        sequence_length=3,
        getPts=False, # TODO: remove this param???
        warp_input=False,
        **config,
    ):
        from utils.homographies import sample_homography_np as sample_homography
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import compute_valid_mask
        from utils.utils import inv_warp_image, warp_points
        import torchvision.transforms as transforms

        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, dict(config))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform


        self.action = "training" if task == "train" else "validation"

        if self.action == "training":
            self.doPhotometric = self.config["augmentation"]["photometric"]["enable_train"]
            self.doHomographic = self.config["augmentation"]["homographic"]["enable_train"]

        elif self.action == "validation":
            self.doPhotometric = self.config["augmentation"]["photometric"]["enable_val"]
            self.doHomographic = self.config["augmentation"]["homographic"]["enable_val"]

        self.homography_params = self.config["augmentation"]["homographic"]["params"]
        self.erosion_radius = self.config["augmentation"]["homographic"]["valid_border_margin"]

        ######
        self.enable_photo_train = self.config["augmentation"]["photometric"]["enable"]
        self.enable_homo_train = self.config["augmentation"]["homographic"]["enable"]
        self.enable_homo_val = False
        self.enable_photo_val = False
        ######

        # self.warp_input = warp_input

        self.cell_size = 8
        self.getPts = getPts

        self.gaussian_label = False
        if self.config["gaussian_label"]["enable"]:
            self.gaussian_label = True

        self.pool = multiprocessing.Pool(16)

        # Parse drawing primitives
        primitives = self.parse_primitives(
            config["primitives"], self.drawing_primitives
        )

        basepath = Path(
            DATA_PATH,
            "Data",
            "synthetic_shapes"
            + ("_{}".format(config["suffix"]) if config["suffix"] is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {"images": [], "points": []} for s in [self.action]}
        for primitive in primitives:
            tar_path = Path(basepath, "{}.tar.gz".format(primitive))
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, self.config)

            # Untar locally
            logging.info("Extracting archive for primitive {}.".format(primitive))
            logging.info(f"tar_path: {tar_path}")
            tar = tarfile.open(tar_path)
            temp_dir = Path(TMPDIR)
            tar.extractall(path=temp_dir)
            tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = self.config["truncate"].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, "images", s).iterdir()]
                f = [p.replace("images", "points") for p in e]
                f = [p.replace(".png", ".npy") for p in f]
                splits[s]["images"].extend(e[: int(truncate * len(e))])
                splits[s]["points"].extend(f[: int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]["images"]))
            for obj in ["images", "points"]:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)



    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

        def checkSat(img, name=""):
            if img.max() > 1:
                print(name, img.max())
            elif img.min() < 0:
                print(name, img.min())


        def load_data_to_tensor(sample):
            img = imread(sample["image"]).astype(np.float32) / 255
            # img = torch.from_numpy(img)
            H, W = img.shape[0], img.shape[1]

            pnts = np.load(sample["points"])  # (y, x)
            pnts = torch.tensor(pnts).float()
            pnts = torch.stack((pnts[:, 1], pnts[:, 0]), dim=1)  # (x, y)
            pnts = filter_points(pnts, torch.tensor([W, H]))

            return img, pnts


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

        def get_labels(pnts, H, W):
            labels = torch.zeros(H, W)

            pnts_int = torch.min(
                pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long()
            )
            labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
            return labels

        def get_label_res(H, W, pnts):
            quan = lambda x: x.round().long()
            labels_res = torch.zeros(H, W, 2)

            labels_res[quan(pnts)[:, 1], quan(pnts)[:, 0], :] = pnts - pnts.round()
            labels_res = labels_res.transpose(1, 2).transpose(0, 1)
            return labels_res

        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy

        # assert Hc == round(Hc) and Wc == round(Wc), "Input image size not fit in the block size"
        idx = np.random.randint(1000) 
        

        # Load Dataset
        img, pnts = load_data_to_tensor(self.samples[index])    # tensor/numpy, tensor (H, W)
        H, W = img.shape[0], img.shape[1]
        # writeimage(img, pnts, "img", idx)


        # Do Phototmetric
        if self.doPhotometric:
            img = imgPhotometric(img)   # numpy (H, W, 1)
            # writeimage(img, pnts, "img_ph", idx)


        # Do Homographic
        if not self.doHomographic:

            img = self.transform(img)   # tensor (1, H, W)
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]), 
                inv_homography=torch.eye(3)
            )

        else:

            from utils.utils import homography_scaling_torch as homography_scaling
            from numpy.linalg import inv

            # Generate a Random Warp Matrix
            homography = self.sample_homography(
                np.array([2, 2]), 
                shift=-1, 
                **self.homography_params
            )
            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()


            # Perform Homographic Transformation
            img = torch.from_numpy(img) # tensor (H, W, 1)
            img = self.inv_warp_image(img, inv_homography, mode="bilinear")  # tensor (H, W)
            img = img.unsqueeze(0)    # tensor (1, H, W)


            pnts = self.warp_points(pnts, homography_scaling(homography, H, W))
            pnts = filter_points(pnts, torch.tensor([W, H]))


            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.erosion_radius
            )  # can set to other value


            # writeimage(img.cpu().numpy().squeeze(), pnts.cpu().numpy(), "warped", idx)


        


        if self.gaussian_label:
            from datasets.data_tools import get_labels_bi

            labels_2D_bi = get_labels_bi(pnts, H, W)
            labels_gaussian = self.gaussian_blur(squeezeToNumpy(labels_2D_bi))
            labels_gaussian = np_to_tensor(labels_gaussian, H, W)

            # cv2.imwrite(f"temp/{idx}_gaussian_label.png", labels_gaussian.cpu().numpy().squeeze() * 255)




        labels_res = get_label_res(H, W, pnts)
        labels_2D = get_labels(pnts, H, W)


        sample = {}
        sample["image"] = img
        # sample["points"] = warped_pnts
        sample.update({"labels_res": labels_res})
        sample.update({"valid_mask": valid_mask})
        sample.update({"labels_2D": labels_2D.unsqueeze(0)})
        sample.update({"homographies": homography})
        sample.update({"inv_homographies": inv_homography})

        if self.getPts:
            sample.update({"pts": pnts})

        if self.gaussian_label:
            sample["labels_2D_gaussian"] = labels_gaussian

        return sample

    def __len__(self):
        return len(self.samples)

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
