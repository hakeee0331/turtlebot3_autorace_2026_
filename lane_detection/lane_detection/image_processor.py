import cv2
import numpy as np

class ImageProcessor():
    def __init__(self):
        pass

    ## 일단 src가 640 x 480 영상인 경우
    @staticmethod
    def perspectiveTransformation(src, src_ptr: np.ndarray|None=None, dst_ptr: np.ndarray|None=None) -> np.ndarray:

        if src_ptr is None:
            src_ptr = np.array([[90,360],[550,360],[640,480],[0,480]], dtype=np.float32)
        if dst_ptr is None:
            dst_ptr = np.array([[20,0],[620,0],[640,120],[0,120]], dtype=np.float32)

        mtrx = cv2.getPerspectiveTransform(src_ptr, dst_ptr)
        src = cv2.warpPerspective(src, mtrx, (640, 120))
        return src    

    ## 흰색 과검출 검사, 바이너리 영상에 적용 중인지 체크할 것.
    @staticmethod
    def many_white(src: np.ndarray):
        return np.sum(src==255)/(src.shape[0]*src.shape[1])
        # return 0        #    일단 보류

    @staticmethod
    def sobel_xy(src: np.ndarray) -> np.ndarray:
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(src_gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = np.absolute(sobel_x)

        if np.max(abs_sobel_x) == 0:
            return np.zeros_like(src_gray, dtype=np.uint8)

        scaled_sobel = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= 60) & (scaled_sobel <= 255)] = 255
        return binary

    @staticmethod
    def morphology(src: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        src = cv2.erode(src, kernel)
        src = cv2.dilate(src, kernel2)
        return src

    @staticmethod
    def componentsWithStatsFilter(src: np.ndarray) -> np.ndarray:
        min_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        mask = np.isin(labels, valid_labels)
        filtered = (mask * 255).astype(np.uint8)
        return filtered