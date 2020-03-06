
import cv2 as cv
import numpy as np
import time as timer

from pathlib import Path






class Event_simulator():
    
    default_config = {
        "contrast_threshold_pos": 0.15,  # Contrast threshold (positive)
        "contrast_threshold_neg": 0.15,  # Contrast threshold  (negative))
        "contrast_threshold_sigma_pos": 0.021,  # Standard deviation of contrast threshold (positive)
        "contrast_threshold_sigma_neg": 0.021,  # Standard deviation of contrast threshold  (negative))
        "refractory_period_ns": 0,  # Refractory period (time during which a pixel cannot fire events just after it fired one), in nanoseconds
        "use_log_image": True,      # Whether to convert images to log images in the preprocessing step.
        "log_eps": 0.1,   # Epsilon value used to convert images to log: L = log(eps + I / 255.0).
        "random_seed": 0,   # Random seed used to generate the trajectories. If set to 0 the current time(0) is taken as seed.
        "frame_rate": 1200, # Specifies the input video framerate (e.g. 1200 fps)

        # "use_event_frame": False,   # 
        # "events_per_frame": 10000,  # 
    }

    def __init__(self, img, time, **config):
        self.config = self.default_config
        # TODO: use utils.tools import dict_update
        # self.config = dict_update(self.config, dict(config))

        assert len(img.shape) == 2, 'Event Simulator takes only gray image'


        if self.config["use_log_image"]:
            img = cv.log(self.config["log_eps"] + img)

        self.last_img = img.copy()
        self.ref_values = img.copy()
        self.last_event_timestamp = np.zeros_like(img)
        self.current_time = time
        self.H, self.W = img.shape


    def simulate(self, img, time):

        assert len(img.shape) == 2, "Event simulator only takes gray images"

        # For each pixel, check if new events need to be generated 
        # since the last image sample
        tolerance = 1e-6
        img = np.array(img)
        H, W = self.H, self.W
        cp = self.config["contrast_threshold_pos"]
        cm = self.config["contrast_threshold_neg"] 
        sigma_cp = self.config["contrast_threshold_sigma_pos"]
        sigma_cm = self.config["contrast_threshold_sigma_neg"]
        refractory_period = self.config["refractory_period_ns"]
        minimum_contrast_threshold = 0.01
        delta_t = (time - self.current_time)


        assert delta_t > 0, "A duration time needs to be greater than zero."
        assert img.shape == (H, W), "Image size of two images do not match."

        if self.config["use_log_image"]:
            img = cv.log(self.config["log_eps"] + img)


        # Init arrays for later calculation
        img_diff = img - self.last_img
        abs_diff = abs(img_diff)
        update_mask = abs_diff > tolerance
        poliarity = img_diff >= 0


        # Sample Positive Events
        stepsize_pos = np.full_like(img, cp) + np.random.normal(0, sigma_cp, [self.H, self.W])
        stepsize_pos = np.maximum(minimum_contrast_threshold, stepsize_pos)
        time_stepsize_pos = stepsize_pos * delta_t / abs_diff

        max_num_steps = int(np.max(np.floor(np.divide(abs_diff, stepsize_pos))))
        max_num_steps = np.array([i+1 for i in range(max_num_steps)])

        grid_pos = np.multiply(max_num_steps[None,None,:], stepsize_pos[:,:,None])
        time_grid_pos = np.multiply(max_num_steps[None,None,:], time_stepsize_pos[:,:,None])

        events_mask = grid_pos + self.last_img[:,:,None] < img[:,:,None]
        indices = np.where(events_mask)
        timestamp = time + time_grid_pos[events_mask]
        y = np.array(indices[0], dtype=float)
        x = np.array(indices[1], dtype=float)
        p = np.ones(len(timestamp), dtype=float)

        events_pos = np.dstack((x, y, timestamp, p)).squeeze()
        
       
        # Sample Negative Events
        stepsize_neg = np.full_like(img, cm) + np.random.normal(0, sigma_cm, [self.H, self.W])
        stepsize_neg = np.maximum(minimum_contrast_threshold, stepsize_neg)
        time_stepsize_neg = stepsize_neg * delta_t / abs_diff

        max_num_steps = int(np.max(np.floor(np.divide(abs_diff, stepsize_neg))))
        max_num_steps = np.array([i+1 for i in range(max_num_steps)])

        grid_neg = np.multiply(max_num_steps[None,None,:], stepsize_neg[:,:,None])
        time_grid_neg = np.multiply(max_num_steps[None,None,:], time_stepsize_neg[:,:,None])

        events_mask =  self.last_img[:,:,None] - grid_neg > img[:,:,None]
        indices = np.where(events_mask)
        timestamp = time + time_grid_neg[events_mask]
        y = np.array(indices[0], dtype=float)
        x = np.array(indices[1], dtype=float)
        p = np.zeros(len(timestamp), dtype=float)

        events_neg = np.dstack((x, y, timestamp, p)).squeeze()


        # Set time and image for next image
        self.current_time = time
        self.last_img = img
        events = np.concatenate((events_pos, events_neg))
        # events = events[events[:,2].argsort()]

        return events




if __name__ == "__main__":

    np.random.seed(0)

    current_time = 0
    frame_rate = 1200
    e = [str(p) for p in Path('boxes_6dof/images').iterdir()]
    e.sort()
    

    img = cv.imread(e[0], cv.IMREAD_GRAYSCALE)
    event_sim = Event_simulator(img, current_time)

    # print(img)
    # raise

    start = timer.time()
    write_count = 0
    synthese_count = 0
    for i, img_path in enumerate(e[1:]):
        print(img_path)


        synthese_start = timer.time()
        current_time += 1/frame_rate 
        print(img_path, current_time)

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        events = event_sim.simulate(img, current_time)
        synthese_count += timer.time() - synthese_start


        write_start = timer.time()
        raw = np.zeros((img.shape[0], img.shape[1], 3))

        pos = events[:,3] == 1.0
        y = events[pos][:,0].astype(int)
        x = events[pos][:,1].astype(int)
        raw[x, y, 2] = 255


        neg = events[:,3] == 0.0
        y = events[neg][:,0].astype(int)
        x = events[neg][:,1].astype(int)
        raw[x, y, 0] = 255

        cv.imwrite("{}.png".format(i), raw)
        write_count += timer.time() - write_start

        # if i > 4:
        #     raise

    print(timer.time() - start)
    print(write_count)
    print(synthese_count)