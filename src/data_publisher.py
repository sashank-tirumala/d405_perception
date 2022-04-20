#!/usr/bin/env python
import pyrealsense2 as rs
import numpy as np
import cv2
import ros_numpy
from d405.msg import rgbd 
import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image

if(__name__ == "__main__"):
# Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    #Start a ROS Publisher
    rospy.init_node("publisher")
    pub = rospy.Publisher('d405_rgbd', rgbd, queue_size=10)
    pub_rgb = rospy.Publisher('d405_rgb', Image, queue_size=10)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        msg = rgbd()
        msg.depth = ros_numpy.msgify(Image, depth_image, encoding='16UC1')
        msg.rgb = ros_numpy.msgify(Image, color_image, encoding='bgr8')
        pub.publish(msg)
        pub_rgb.publish(msg.rgb)

