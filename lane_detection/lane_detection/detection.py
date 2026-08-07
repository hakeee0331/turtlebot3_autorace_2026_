import cv2
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage


from .image_processor import ImageProcessor

class LaneDetection(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        ## INIT TOPIC CHAINS
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
        self.laneModeSubscriber = self.create_subscription(
            Bool,
            '/lane_mode',
            self.laneModeSubscriber_callback,
            10
        )
        self.stateSubscriber = self.create_subscription(
            Bool,
            '/lane_state',
            self.stateSubscriber_callback,
            10
        )
        self.velocityPublisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        self.intersectionStateSubscriber = self.create_subscription(
            Bool,
            '/intersection_state',
            self.intersectionStateSubscriber_callback,
            10
        )

        ## INIT STATES
        self.lane_mode = True  # default = True(yellow)
        self.state = False
        self.is_intersection = False

        ##
        self.ip = ImageProcessor()

        
        
    
    def intersectionStateSubscriber_callback(self, msg):
        if not self.is_intersection and msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Start Intersection Mode')
        if self.is_intersection and not msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Finish Intersection Mode')
            if self.lane_mode == True:
                threading.Timer(25.0, self.intersection_left_handling).start()
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
        if self.lane_mode: self.get_logger().info('Change lane mode to YELLOW')
        else: self.get_logger().info('Change lane mode to WHITE')
        cv2.destroyAllWindows()
        return
    
    def videoSubscriber_callback(self, msg):
        if not self.state: return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            src = self.ip.perspectiveTransformation(src)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return

        src_yellow = np.empty((120,320))
        src_white = np.empty((120,320))

        if self.lane_mode:
            src_yellow, center_yellow = self.ip.findLaneCenter(src[:,:320], self.lane_mode)
            if self.ip.many_white(src_yellow) > 0.1:
                print(self.ip.many_white(src))
                return
            twist = Twist()
            angle = (60-center_yellow)/60
            if angle < -1.65: angle = -1.65
            elif angle > 1.5: angle = 1.5
            if self.is_intersection:
                twist.linear.x = 0.15 / (1 + abs(angle)/1.3)
                twist.angular.z = angle/2.5
            else:
                twist.linear.x = 0.22 / (1 + abs(angle/1.5))
                twist.angular.z = angle/1.7 #1.5
            self.velocityPublisher.publish(twist)
            cv2.circle(src, (center_yellow,60), 5, (0,0,255), -1)
            cv2.imshow('src_yellow', src_yellow)
            
        else:
            src_white, center_white = self.ip.findLaneCenter(src[:,320:], self.lane_mode)
            if self.ip.many_white(src) > 0.1:
                print(self.ip.many_white(src_white))
                return
            twist = Twist()
            angle = (580-center_white)/77 #---------------------------------
            if angle < -1.65: angle = -1.65
            elif angle > 1.5: angle = 1.6
            if self.is_intersection:
                twist.linear.x = 0.15 / (1 + abs(angle)/1.3)
                twist.angular.z = angle/3.2
            else:
                twist.linear.x = 0.22 / (1 + abs(angle)/1.5)
                twist.angular.z = angle/1.3 #1.4
            self.velocityPublisher.publish(twist)
            cv2.circle(src, (center_white,60), 5, (255,0,0), -1)
            cv2.imshow('src_white', src_white)
        
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
