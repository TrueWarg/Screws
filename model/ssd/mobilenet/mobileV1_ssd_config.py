import numpy as np

from model.ssd.config import Config
from model.ssd.prior_generators import SsdBoxGenParams, generate_ssd_priors

config = Config(
    image_size=300,
    image_mean=np.array([127, 127, 127]),
    image_std=128.0,
    iou_threshold=0.45,
    center_variance=0.1,
    size_variance=0.2,
)

specs = [
    SsdBoxGenParams(feature_map_size=19, shrinkage=16, average_box_relates_size=0.05, rotation_step=30,
                    aspect_ratios=[4]),
    SsdBoxGenParams(feature_map_size=10, shrinkage=32, average_box_relates_size=0.1, rotation_step=30,
                    aspect_ratios=[4]),
    SsdBoxGenParams(feature_map_size=5, shrinkage=64, average_box_relates_size=0.3, rotation_step=30,
                    aspect_ratios=[4]),
    SsdBoxGenParams(feature_map_size=3, shrinkage=100, average_box_relates_size=0.3, rotation_step=30,
                    aspect_ratios=[4]),
    SsdBoxGenParams(feature_map_size=2, shrinkage=150, average_box_relates_size=0.6, rotation_step=30,
                    aspect_ratios=[4]),
    SsdBoxGenParams(feature_map_size=1, shrinkage=300, average_box_relates_size=0.65, rotation_step=30,
                    aspect_ratios=[4]),
]

priors = generate_ssd_priors(specs, config.image_size)
