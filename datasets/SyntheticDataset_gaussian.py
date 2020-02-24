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
import time

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
# import multiprocessing

TMPDIR = SYN_TMPDIR  # './datasets/' # you can define your tmp dir


def load_as_float(path):
    if path.endswith(".png"):
        return imread(path).astype(np.float32) / 255
    elif path.endswith(".npy"):
        return (np.load(path)).astype(np.float32) / 255
    else:
        raise UnknownDataFormat() 


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
            "image_size": [960, 1280],
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
            # "draw_polygon",
            # "draw_multiple_polygons",
            # "draw_ellipses",
            # "draw_star",
            # "draw_checkerboard",
            # "draw_stripes",
            # "draw_cube",
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

        self.action = "training" if task == "train" else "validation"
        # self.warp_input = warp_input

        self.cell_size = 8
        self.getPts = getPts

        self.gaussian_label = self.config["gaussian_label"]["enable"] == True
        # if self.config["gaussian_label"]["enable"]:
        #     # self.params_transform = {'crop_size_y': 120, 'crop_size_x': 160, 'stride': 1, 'sigma': self.config['gaussian_label']['sigma']}
        #     self.gaussian_label = True

        # self.pool = multiprocessing.Pool(6)

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

        splits = {s: {"images": [], "events": [], "points": []} for s in [self.action]}

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


        # Gather filenames in all splits, optionally truncate
        for primitive in primitives:
            temp_dir = Path(TMPDIR)
            truncate = self.config["truncate"].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                i = [str(p) for p in Path(path, "images", s).iterdir()]
                e = [p.replace("images", "ev_expo").replace(".png", ".npy") for p in i]
                f = [p.replace("images", "points").replace(".png", ".npy") for p in i]
                splits[s]["images"].extend(i[: int(truncate * len(i))])
                splits[s]["events"].extend(e[: int(truncate * len(e))])
                splits[s]["points"].extend(f[: int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]["events"]))
            for obj in ["images", "events", "points"]:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.crawl_folders(splits)


    def crawl_folders(self, splits):
        sequence_set = []
        for (imgs, evts, pnts) in zip(
            splits[self.action]["images"], splits[self.action]["events"], splits[self.action]["points"]
        ):
            sample = {"images": imgs, "events": evts, "points": pnts}
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

        def checkSat(img, name=""):
            if img.max() > 1:
                print(name, img.max())
            elif img.min() < 0:
                print(name, img.min())

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

        # def imgHomographic(img):
        #     pass

        def get_labels(pnts, H, W):
            labels = torch.zeros(H, W)
            # print('--2', pnts, pnts.size())
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())
            pnts_int = torch.min(
                pnts.round().long(), torch.tensor([[W - 1, H - 1]]).long()
            )

            # print('--3', pnts_int, pnts_int.size())
            labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
            return labels

        def get_label_res(H, W, pnts):
            quan = lambda x: x.round().long()
            labels_res = torch.zeros(H, W, 2)
            # pnts_int = torch.min(pnts.round().long(), torch.tensor([[H-1, W-1]]).long())

            labels_res[quan(pnts)[:, 1], quan(pnts)[:, 0], :] = pnts - pnts.round()
            # print("pnts max: ", quan(pnts).max(dim=0))
            # print("labels_res: ", labels_res.shape)
            labels_res = labels_res.transpose(1, 2).transpose(0, 1)
            return labels_res

        def output_test_img(img, pnts=[], idx=0):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for pt in pnts:
                cv2.circle(img, (pt[0], pt[1]), 2, (0, 255, 0), -1)
            cv2.imwrite(str(Path("temp", "{:02d}.png".format(idx))), img * 255)


        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy


        sample = self.samples[index]
        img = load_as_float(sample["images"])
        evt = load_as_float(sample["events"])
        H, W, C = evt.shape

        pnts = np.load(sample["points"])  # (y, x)
        pnts = torch.tensor(pnts).float()
        pnts = torch.stack((pnts[:, 1], pnts[:, 0]), dim=1)  # (x, y)
        pnts = filter_points(pnts, torch.tensor([W, H]))
        sample = {}

        # labels_2D = get_labels(pnts, H, W)
        # sample.update({"labels_2D": labels_2D.unsqueeze(0)})

        if self.action == "training":
            do_Photometrix = self.config["augmentation"]["photometric"]["enable_train"]
            do_Homographic = self.config["augmentation"]["homographic"]["enable_train"]
        elif self.action == "validation":
            do_Photometrix = self.config["augmentation"]["photometric"]["enable_val"]
            do_Homographic = self.config["augmentation"]["homographic"]["enable_val"]
        else:
            do_Photometrix = False
            do_Homographic = False


        # TODO: remove the sanity check part
        # output_test_img(img, pnts, 0)
        
        # assert Hc == round(Hc) and Wc == round(Wc), "Input image size not fit in the block size"
        if do_Photometrix:
            img = imgPhotometric(img)
        # output_test_img(img, pnts, 1)

        if do_Homographic:
            print('<<< Homograpy aug enabled for %s.'%self.action)
            from utils.utils import homography_scaling_torch as homography_scaling

            img = torch.from_numpy(img)
            erosion_radius=self.config["augmentation"]["homographic"]["valid_border_margin"]

            homography = self.sample_homography(
                np.array([2, 2]),
                shift=-1,
                **self.config["augmentation"]["homographic"]["params"],
            )
            homography = np.linalg.inv(homography) # use inverse from the sample homography
            homography = torch.tensor(homography).float()
            inv_homography = homography.inverse()

            warped_pnts = self.warp_points(pnts, homography_scaling(homography, H, W))
            warped_pnts = filter_points(warped_pnts, torch.tensor([W, H]))

            warped_img = self.inv_warp_image(img.squeeze(), inv_homography, mode="bilinear")
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, np.newaxis]

            # output_test_img(warped_img.squeeze(), warped_pnts.numpy().squeeze(), 2)
            warped_img = self.transform(warped_img) if self.transform else warped_img
            # output_test_img(warped_img.numpy().squeeze(), warped_pnts.numpy().squeeze(), 3)
            
        else:
            print('<<< Homograpy aug disabled for %s.'%self.action)
            img = img[:, :, np.newaxis]
            warped_img = self.transform(img) if self.transform else img
            warped_pnts = pnts
            erosion_radius = 0


        valid_mask = self.compute_valid_mask(
            torch.tensor([H, W]),
            inv_homography=inv_homography,
            erosion_radius=erosion_radius,
        )


        # TODO: remove the sanity check part
        # output_test_img(warped_img.numpy().squeeze(), warped_pnts.numpy().squeeze(), 4)
        # output_test_img(valid_mask.numpy().squeeze(), warped_pnts.numpy().squeeze(), 5)
        
        labels_2D = get_labels(warped_pnts, H, W)
        labels_res = get_label_res(H, W, warped_pnts)
        sample["image"] = warped_img
        sample.update({"valid_mask": valid_mask})
        sample.update({"labels_2D": labels_2D.unsqueeze(0)})
        sample.update({"labels_res": labels_res})

        # TODO: remove the sanity check part
        temp = cv2.cvtColor(warped_img.numpy().squeeze(), cv2.COLOR_GRAY2RGB)
        temp[:, :, 1] = labels_2D.numpy().squeeze() * 255
        # output_test_img(temp, [], 6)

        
        if self.gaussian_label:
            from datasets.data_tools import get_labels_bi

            labels_2D_bi = get_labels_bi(warped_pnts, H, W)

            labels_gaussian = self.gaussian_blur(squeezeToNumpy(labels_2D_bi))
            labels_gaussian = np_to_tensor(labels_gaussian, H, W)
            sample["labels_2D_gaussian"] = labels_gaussian

            # output_test_img(labels_gaussian.numpy().squeeze(), [], 7)


        ### code for warped image
        # if self.config["warped_pair"]["enable"]:
        if True:
            from datasets.data_tools import warpLabels

            homography = self.sample_homography(
                np.array([2, 2]), 
                shift=-1, 
                **self.config["warped_pair"]["params"]
            )

            ##### use inverse from the sample homography
            homography = np.linalg.inv(homography)
            #####
            inv_homography = np.linalg.inv(homography)

            homography = torch.tensor(homography).type(torch.FloatTensor)
            inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

            # photometric augmentation from original image

            # warp original image
            warped_img = img.type(torch.FloatTensor)
            warped_img = self.inv_warp_image(warped_img.squeeze(), inv_homography, mode="bilinear").unsqueeze(0)

            if (self.enable_photo_train and self.action == "train") or \
               (self.enable_photo_val and self.action == "val"):
                warped_img = imgPhotometric(warped_img.numpy().squeeze())  # numpy array (H, W, 1)
                warped_img = torch.tensor(warped_img, dtype=torch.float32)

            warped_img = warped_img.view(-1, H, W)

            # warped_labels = warpLabels(pnts, H, W, homography)
            warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
            warped_labels = warped_set["labels"]
            warped_res = warped_set["res"]
            warped_res = warped_res.transpose(1, 2).transpose(0, 1)

            if self.gaussian_label:
                warped_labels_bi = warped_set["labels_bi"]
                warped_labels_gaussian = self.gaussian_blur(
                    squeezeToNumpy(warped_labels_bi)
                )
                warped_labels_gaussian = np_to_tensor(warped_labels_gaussian, H, W)
                sample["warped_labels_gaussian"] = warped_labels_gaussian
                sample.update({"warped_labels_bi": warped_labels_bi})

            sample.update(
                {
                    "warped_img": warped_img,
                    "warped_labels": warped_labels,
                    "warped_res": warped_res,
                }
            )

            # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
            valid_mask = self.compute_valid_mask(
                torch.tensor([H, W]),
                inv_homography=inv_homography,
                erosion_radius=self.config["warped_pair"]["valid_border_margin"],
            )  # can set to other value
            sample.update({"warped_valid_mask": valid_mask})
            sample.update(
                {"homographies": homography, "inv_homographies": inv_homography}
            )


# Image
# Valid_Mask
# Labels_2D
# Labels_Res
# Warped_Img
# Warped_Labels
# Warped_Res
# Warped_Valid_Mask
# Homographies
# Inv_Homographies

        print(sample)
        for x in sample:
            print(x.strip().title())
        raise

        if self.getPts:
            sample.update({"pts": pnts})

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
