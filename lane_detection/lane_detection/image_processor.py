import cv2
import numpy as np

class ImageProcessor():
    def __init__(self):
        ## 영상 사이즈 등의 세팅값 넣을 예정
        pass

    ## 일단 src가 640 x 480 영상인 경우
    def perspectiveTransformation(self, src):
        src_ptr = np.float32([[90,360],[550,360],[640,480],[0,480]])
        dst_ptr = np.float32([[20,0],[620,0],[640,120],[0,120]])
        mtrx = cv2.getPerspectiveTransform(src_ptr, dst_ptr)
        src = cv2.warpPerspective(src, mtrx, (640, 120))
        return src

    def findLaneCenter(self, src, lane_mode):
        src = cv2.GaussianBlur(src, (7,7), sigmaX=0, sigmaY=0)
        src = self._sobel_xy(src)
        src = self._morphology(src)
        src = self._componentsWithStatsFilter(src)
        contours, _ = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        sums = 0
        count = 0
        if np.count_nonzero(src) < len(src)*len(src[0])//4:
            for contour in contours:
                if cv2.contourArea(contour) < 100: continue
                m = cv2.moments(contour)
                sums += int(m['m10']/m['m00'])
                count += 1
        try:
            if lane_mode: sums = sums//count
            else: sums = sums//count + 320
        except:
            if lane_mode: sums = 0
            else: sums = 640
        return src, sums

    def many_white(self, src):
        # return np.sum(src==255)/(src.shape[0]*src.shape[1])
        return 0        #    일단 보류

    def _sobel_xy(self, src):
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobel_x = np.absolute(sobel_x)
        scaled_sobel = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= 60) & (scaled_sobel <= 255)] = 255
        return binary

    def _morphology(self, src):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        src = cv2.erode(src, kernel)
        src = cv2.dilate(src, kernel2)
        return src

    def _componentsWithStatsFilter(self, src):
        min_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        mask = np.isin(labels, valid_labels)
        filtered = (mask * 255).astype(np.uint8)
        return filtered