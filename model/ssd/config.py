from dataclasses import dataclass


@dataclass()
class Config:
    center_variance: float
    size_variance: float
