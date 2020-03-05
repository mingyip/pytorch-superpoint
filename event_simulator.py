
import cv2 as cv
import numpy as np

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

        events = []
        for y in range(H):
            for x in range(W):

                itdt = img[y, x]
                it = self.last_img[y, x]
                prev_cross = self.ref_values[y, x]


                if abs(it - itdt) > tolerance:

                    if itdt >= it:
                        pol = 1.0
                        c = cp
                        sigma_c = sigma_cp
                    else:
                        pol = -1.0
                        c = cm
                        sigma_c = sigma_cm

                    if sigma_c > 0:
                        c += np.random.normal(0, sigma_c, 1).squeeze()
                        c = np.maximum(minimum_contrast_threshold, c)

                    curr_cross = prev_cross
                    all_crossings = False

                    while not all_crossings:
                        curr_cross += pol * c

                        if (pol > 0 and curr_cross > it and curr_cross <= itdt) \
                            or (pol < 0 and curr_cross < it and curr_cross >= itdt):

                            edt = (curr_cross - it) * delta_t / (itdt -it)
                            t = self.current_time + edt

                            # # check that pixel (x,y) is not currently in a "refractory" state
                            # # i.e. |t-that last_timestamp(x,y)| >= refractory_period
                            last_stamp_at_xy = self.last_event_timestamp[y, x]

                            assert t > last_stamp_at_xy
                            dt = t - last_stamp_at_xy
                            if last_stamp_at_xy == 0 or dt >= refractory_period:
                                events.append([x, y, t, int(pol > 0)])
                                self.last_event_timestamp[y, x] = t

                            self.ref_values[y, x] = curr_cross
                            
                        else:
                            all_crossings = True


        self.current_time = time
        self.last_img = img
        events = np.array(events)
        events = events[events[:,2].argsort()]

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

    for i, img_path in enumerate(e[1:]):
        # t = np.random.uniform(0, 0.001, 1).squeeze()
        current_time += 1/frame_rate 
        print(img_path, current_time)

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        events = event_sim.simulate(img, current_time)

        raw = np.zeros((img.shape[0], img.shape[1], 3))

        for evt in events:
            if evt[3]:
                raw[int(evt[1]), int(evt[0]), 2] = 255
            else:
                raw[int(evt[1]), int(evt[0]), 0] = 255

        cv.imwrite("{}.png".format(i), raw)

        # if i > 4:
        #     raise
