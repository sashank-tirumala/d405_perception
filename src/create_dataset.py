import numpy as np
import cv2
import argparse
import os
import shutil
import rosbag
from sensor_msgs.msg import Image
import ros_numpy
from alive_progress import alive_bar

def create_dataset(bagfile_name, dataset_name, del_if_exists=True, rgbd_topic="d405_rgbd"):
    """
    Input: 
    -bagfile_path: name of bagfile
    -dataset_name: name of the folder where you will store this data
    -del_if_exists: If dataset of given name already exists, then delete the dataset
    Output:
    Nothing-> You will create a folder with 2 subdirs: rgb, depth, each of the images will be stored as numpy arrays. 
    [This is path fixed code, all data needs to be organized in a certain way]
    """
    root = "/media/YertleDrive4/layer_grasp"
    bagfile = rosbag.Bag(root+"/bagfiles/"+bagfile_name+".bag")
    dataset_dir_path = root+"/dataset/"+dataset_name
    if(os.path.isdir(dataset_dir_path)):
        print("Directory already exists!")
        if(del_if_exists):
            print("Deleting and creating again!")
            shutil.rmtree(dataset_dir_path, ignore_errors=True)
        else:
            exit()
    rgb_path = dataset_dir_path+"/rgb"
    depth_path = dataset_dir_path+"/depth"
    os.makedirs(dataset_dir_path)
    os.makedirs(rgb_path)   
    os.makedirs(depth_path)
    total_num = bagfile.get_message_count()
    i=0
    with alive_bar(total_num) as bar:
        for topic, msg, t in bagfile.read_messages():
            if(topic == rgbd_topic):
                msg.rgb.__class__ = Image
                msg.depth.__class__ = Image
                rgb_img = ros_numpy.numpify(msg.rgb)
                depth_img = ros_numpy.numpify(msg.depth)
                np.save(rgb_path+"/"+str(i)+".npy", rgb_img)
                np.save(depth_path+"/"+str(i)+".npy", depth_img)
                bar()
                i+=1

        
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='creates dataset from bagfile')
    parser.add_argument('-b','--bagfile_name', help='Name of bagfile', required=True)
    parser.add_argument('-d','--dataset_name', help='Name of Dataset', required=True)
    # parser.add_argument('-dt','--depth_topic', help='Topic of depth image', default="d405_depth")
    # parser.add_argument('-rgbt','--rgb_topic', help='Topic of RGB Image', default="d405_rgb")
    parser.add_argument('-rgbd','--rgbd_topic', help='Topic of RGB Image', default="d405_rgbd")
    parser.add_argument('-del','--del_if_exists', help='Delete dataset if it already exists', default=True)

    args = vars(parser.parse_args())
    # create_dataset(bagfile_name = args["bagfile_name"], 
    # dataset_name=args["dataset_name"], 
    # del_if_exists=args["del"], 
    # rgb_topic = args["rgb_topic"], 
    # depth_topic=args["d405_depth"])
    create_dataset(**args)
#images to consider,0, 176, 277, 372, 535, 690, 918