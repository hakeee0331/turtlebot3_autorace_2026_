from dataclasses import dataclass

@dataclass
class PerspectiveTransform:
    top_left_x: int
    top_left_y: int 
    top_right_x: int 
    top_right_y: int 

    bottom_left_x: int
    bottom_left_y: int 
    bottom_right_x: int 
    bottom_right_y: int
    

@dataclass
class LaneTrackingConfig:
    roi_start_x: int
    roi_end_x: int
    target_x: int
    error_scale: float
    normal_angular_divisor: float
    intersection_angular_divisor: float
    debug_window: str
    debug_color: tuple  # (0, 0, 255): red