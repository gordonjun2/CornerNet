import numpy as np
from db.base import BASE

class DETECTION(BASE):
    def __init__(self):
        super(DETECTION, self).__init__()

        self._configs["categories"]      = 80
        self._configs["kp_categories"]   = 1
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2

        self._configs["input_size"]      = [511]
        self._configs["output_sizes"]    = [[128, 128]]

        self._configs["nms_threshold"]   = 0.5
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 100
        self._configs["ae_threshold"]    = 0.5
        self._configs["nms_kernel"]      = 3

        self._configs["nms_algorithm"]   = "exp_soft_nms"
        self._configs["weight_exp"]      = 8
        self._configs["merge_bbox"]      = False
        
        self._configs["data_aug"]        = True
        self._configs["lighting"]        = True

        self._configs["border"]          = 128
        self._configs["gaussian_bump"]   = True
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1
        self._configs["rand_crop"]       = False
        self._configs["rand_color"]      = False
        self._configs["rand_pushes"]     = False
        self._configs["rand_samples"]    = False
        self._configs["special_crop"]    = False

        self._configs["test_scales"]     = [1]

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(
                self._configs["rand_scale_min"], 
                self._configs["rand_scale_max"],
                self._configs["rand_scale_step"]
            )

    @property
    def categories(self):
        return self._configs["categories"]

    @property
    def kp_categories(self):
        return self._configs["kp_categories"]

    @property
    def rand_scales(self):
        return self._configs["rand_scales"]

    @property
    def rand_scale_min(self):
        return self._configs["rand_scale_min"]

    @property
    def rand_scale_max(self):
        return self._configs["rand_scale_max"]

    @property
    def rand_scale_step(self):
        return self._configs["rand_scale_step"]

    @property
    def input_size(self):
        return self._configs["input_size"]

    @property
    def output_sizes(self):
        return self._configs["output_sizes"]

    @property
    def nms_threshold(self):
        return self._configs["nms_threshold"]

    @property
    def max_per_image(self):
        return self._configs["max_per_image"]

    @property
    def top_k(self):
        return self._configs["top_k"]

    @property
    def ae_threshold(self):
        return self._configs["ae_threshold"]

    @property
    def nms_kernel(self):
        return self._configs["nms_kernel"]

    @property
    def nms_algorithm(self):
        return self._configs["nms_algorithm"]
   
    @property
    def weight_exp(self):
        return self._configs["weight_exp"]

    @property
    def merge_bbox(self):
        return self._configs["merge_bbox"]

    @property
    def data_aug(self):
        return self._configs["data_aug"]

    @property
    def lighting(self):
        return self._configs["lighting"]

    @property
    def border(self):
        return self._configs["border"]

    @property
    def gaussian_bump(self):
        return self._configs["gaussian_bump"]

    @property
    def gaussian_iou(self):
        return self._configs["gaussian_iou"]

    @property
    def gaussian_radius(self):
        return self._configs["gaussian_radius"]

    @property
    def rand_crop(self):
        return self._configs["rand_crop"]

    @property
    def rand_color(self):
        return self._configs["rand_color"]

    @property
    def rand_pushes(self):
        return self._configs["rand_pushes"]

    @property
    def rand_samples(self):
        return self._configs["rand_samples"]

    @property
    def special_crop(self):
        return self._configs["special_crop"]

    @property
    def test_scales(self):
        return self._configs["test_scales"]

    @property
    def full(self):
        return self._configs

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

db_configs = DETECTION()
