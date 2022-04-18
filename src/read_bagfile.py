import cv2
import rosbag
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import ros_numpy
from sensor_msgs.msg import Image

def create_video_array(bagfile_path, image_topic = ['/d405_rgb'], save_path="./", del_if_dir_exists=False):
    bagfile_name = bagfile_path.split("/")
    bagfile_name = bagfile_name[-1][:-4]
    img_dir_path = save_path+"/"+bagfile_name
    if(os.path.isdir(img_dir_path)):
        print("Directory already exists!")
        if(del_if_dir_exists):
            print("Deleting and creating again!")
            shutil.rmtree(img_dir_path, ignore_errors=True)
        else:
            exit()
    print(img_dir_path)
    os.makedirs(img_dir_path)
    bagfile = rosbag.Bag(bagfile_path)
    i=0
    for topic, msg, t in bagfile.read_messages(topics=image_topic):
        msg.__class__ = Image
        img = ros_numpy.numpify(msg)
        # print(img)
        file_name = save_path + "/" + bagfile_name +"/" + str(i) +".jpg"
        cv2.imwrite(file_name, img)
        i=i+1
   
if(__name__ == "__main__"):
    create_video_array(bagfile_path="/home/catkin_ws/src/d405/bag/Camera.bag", save_path="/media/YertleDrive4/layer_grasp/images", del_if_dir_exists=True)
    
    pass