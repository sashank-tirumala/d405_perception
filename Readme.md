## Execution (in theCat only, uses a singularity container)
### Run D405 Node to publish color_img (Depth image facing issues with ROS msg)
1. `sudo singularity shell -B /media/YertleDrive4:/media/YertleDrive4 -w noetic/`
2. `source /home/catkin_ws/devel/setup.bash`
3. `roscore`
4. Open another terminal with Singularity and source that terminal by performing step 2 again
5. On the new terminal run `rosrun d405 data_publisher.py`. Data currently is published to `d405_rgb` topic. It is of type sensor_msgs.msg.Image. 
6. In order to see the RGB output use: `rosrun image_view image_view image:=d405_rgb`
7. In order to collect data use: `rosbag record -O <name> d405_rgbd`
8. In order to create a directory with RGB and Depth images use:  ` python create_dataset.py -b <bagfilename> -d <datasetname>`
9. If you cannot open the display, exit singularity `xhost +local:`