import cv2
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage


from .center_detector import CenterDetector
from .image_processor import ImageProcessor
from .config import LaneTrackingConfig

class LaneDetection(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        ## INIT STATES
        self.lane_mode = True  # default = True(Left Lane)
        self.state = False
        self.is_intersection = False

        ## INIT module
        self.detector = CenterDetector()


        ## ====== INIT TOPIC CHAINS =======
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
        self.stateSubscriber = self.create_subscription(
            Bool,
            '/lane_state',
            self.stateSubscriber_callback,
            10
        )
        self.intersectionStateSubscriber = self.create_subscription(
            Bool,
            '/intersection_state',
            self.intersectionStateSubscriber_callback,
            10
        )
        self.laneModeSubscriber = self.create_subscription(
            Bool,
            '/lane_mode',
            self.laneModeSubscriber_callback,
            10
        )
        self.velocityPublisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

    def intersectionStateSubscriber_callback(self, msg):
        if not self.is_intersection and msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Start Intersection Mode')
        if self.is_intersection and not msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Finish Intersection Mode')
            if self.lane_mode == True:
                threading.Timer(25.0, self.intersection_left_handling).start()      # 왜 ros timer가 아닌가요?
            else:
                threading.Timer(12.0, self.intersection_right_handling).start()
                threading.Timer(28.0, self.intersection_right_handling2).start()
    
    def intersection_left_handling(self):
        self.lane_mode = False
        cv2.destroyAllWindows()
        
    def intersection_right_handling(self):
        self.lane_mode = True
        cv2.destroyAllWindows()
    
    def intersection_right_handling2(self):
        self.lane_mode = False
        cv2.destroyAllWindows()
    
    def stateSubscriber_callback(self, msg):
        self.state = msg.data
        if self.state == True: print('Start Lane Detection')
        else:
            twist = Twist()
            self.velocityPublisher.publish(twist)
            print('Terminate Lane Detection')
            cv2.destroyAllWindows()
    
    def laneModeSubscriber_callback(self, msg):
        self.lane_mode = msg.data
        if self.lane_mode: self.get_logger().info('Change lane mode to LEFT')
        else: self.get_logger().info('Change lane mode to RIGHT')
        return


    ## ======= lane_controlling ======

    LEFT_LANE_CONFIG = LaneTrackingConfig(
            roi_start_x=0,
            roi_end_x=320,
            target_x=60,
            error_scale=60.0,
            normal_angular_divisor=1.7,
            intersection_angular_divisor=2.5,
            debug_window='left_lane',
            debug_color=(0, 0, 255),
        )
    RIGHT_LANE_CONFIG = LaneTrackingConfig(
            roi_start_x=320,
            roi_end_x=640,
            target_x=260,  # 전체 좌표 580(목표위치) - ROI 시작 좌표 320
            error_scale=77.0,
            normal_angular_divisor=1.3,
            intersection_angular_divisor=3.2,
            debug_window='right_lane',
            debug_color=(255, 0, 0),
        )

    def makeTwist(self, lane_center, config):
        normalized_error = (config.target_x - lane_center) / config.error_scale
        normalized_error = np.clip(normalized_error, -1.65, 1.5)

        twist = Twist()
        if self.is_intersection:
            base_speed = 0.15
            linear_divisor = 1.3
            angular_divisor = config.intersection_angular_divisor
        else:
            base_speed = 0.22
            linear_divisor = 1.5
            angular_divisor = config.normal_angular_divisor

        twist.linear.x = base_speed / (1 + abs(normalized_error / linear_divisor))
        twist.angular.z = normalized_error / angular_divisor

        return twist


    def videoSubscriber_callback(self, msg):
        if not self.state: return

        ## 이미지처리
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src_origin = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            src = ImageProcessor.perspectiveTransformation(src_origin) ## 원근변환
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return


        ## lane center 검출
        if self.lane_mode: config = self.LEFT_LANE_CONFIG
        else: config = self.RIGHT_LANE_CONFIG

        src_detect, lane_center = self.detector.findLaneCenter(src[:, config.roi_start_x:config.roi_end_x], detect_mode='saturation')
        if lane_center == None: lane_center = config.roi_start_x    # 검출 불가시 극단값 처리

        detect_fail = False
        if ImageProcessor.many_white(src_detect) > 0.1:
            print(f'white: {ImageProcessor.many_white(src_detect)}')
            cv2.putText(src, "detect fail", (50,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
            detect_fail = True

        # twist 계산
        twist = self.makeTwist(lane_center, config)
        if not detect_fail:
            self.velocityPublisher.publish(twist)   

        cv2.circle(src, (lane_center + config.roi_start_x, 60), 5, config.debug_color, -1)
        cv2.putText(src_detect, config.debug_window, (50,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
        cv2.imshow("detected lane", src_detect)
        cv2.imshow('src', src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            twist = Twist()
            self.velocityPublisher.publish(twist)
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = LaneDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
