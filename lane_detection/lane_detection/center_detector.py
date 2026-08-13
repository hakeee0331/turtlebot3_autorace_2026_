import cv2
import numpy as np
from .image_processor import ImageProcessor as ip

class CenterDetector:
    def __init__(self):
        pass

    def findLaneCenter(self, src: np.ndarray, detect_mode='sobel'):
        if detect_mode == 'sobel':
            return self._detectCenterSobel(src)

        raise ValueError(
            f'Unsupported detect mode: {detect_mode}'
        )

    def _detectCenterSobel(self, src: np.ndarray) -> tuple[np.ndarray, int|None]:
        src = cv2.GaussianBlur(src, (7,7), sigmaX=0, sigmaY=0)
        src = ip.sobel_xy(src)
        src = ip.morphology(src)
        src = ip.componentsWithStatsFilter(src)

        contours, _ = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        sums = 0
        count = 0

        if np.count_nonzero(src) < len(src)*len(src[0])//4:
            for contour in contours:
                if cv2.contourArea(contour) < 100: continue
                m = cv2.moments(contour)
                sums += int(m['m10']/m['m00'])
                count += 1

        if count == 0: return src, None     # center 검출 실패
        center = sums // count

        return src, center      # @src: binary image of detected lane, sums: x value of moment

