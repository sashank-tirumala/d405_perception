import numpy as np
import cv2
import argparse
import os
# parser =argparse.ArgumentParser()
# parser.add_argument('input_img', help = 'the input image file')
# args = parser.parse_args()

def nothing(x):
    pass

def colormask(filename):
    cv2.namedWindow(filename,1)

    #set trackbar
    hh = 'hue high'
    hl = 'hue low'
    sh = 'saturation high'
    sl = 'saturation low'
    vh = 'value high'
    vl = 'value low'
    mode = 'mode'

    #set ranges
    cv2.createTrackbar(hh, filename, 0,179, nothing)
    cv2.createTrackbar(hl, filename, 0,179, nothing)
    cv2.createTrackbar(sh, filename, 0,255, nothing)
    cv2.createTrackbar(sl, filename, 0,255, nothing)
    cv2.createTrackbar(vh, filename, 0,255, nothing)
    cv2.createTrackbar(vl, filename, 0,255, nothing)

    thv= 'th1'
    cv2.createTrackbar(thv, filename, 127,255, nothing)

    #read img in both rgb and grayscale
    max_num = len(os.listdir(filename))
    nums = np.random.randint(0,max_num, size=3)
    imgs = []
    for num in nums:
        img = np.load(filename+"/"+str(num)+".npy")
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=1)
    nums = np.random.randint(0,max_num, size=3)
    imgs2 = []
    for num in nums:
        img = np.load(filename+"/"+str(num)+".npy")
        imgs2.append(img)
    imgs2 = np.concatenate(imgs2, axis=1)
    imgs =  np.concatenate([imgs, imgs2], axis=0)
    scale_percent = 50 # percent of original size
    width = int(imgs.shape[1] * scale_percent / 100)
    height = int(imgs.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgs = cv2.resize(imgs, dim, interpolation = cv2.INTER_AREA)

    #convert rgb to hsv
    hsv_img = cv2.cvtColor(imgs, cv2.COLOR_BGR2HSV)

    while True:
        hul= cv2.getTrackbarPos(hl,filename)
        huh= cv2.getTrackbarPos(hh,filename)
        sal= cv2.getTrackbarPos(sl,filename)
        sah= cv2.getTrackbarPos(sh,filename)
        val= cv2.getTrackbarPos(vl,filename)
        vah= cv2.getTrackbarPos(vh,filename)
        thva= cv2.getTrackbarPos(thv,filename)


        hsvl = np.array([hul, sal, val], np.uint8)
        hsvh = np.array([huh, sah, vah], np.uint8)

        mask = cv2.inRange(hsv_img, hsvl, hsvh)

        res = cv2.bitwise_and(imgs, imgs, mask=mask)

        #set image for differnt modes
        #convert black to white
        res[np.where((res==[0,0,0]).all(axis=2))] = [255,255,255]

        
        cv2.imshow(filename,res)
       

        #press 'Esc' to close the window
        ch = cv2.waitKey(5)
        if ch== 27:
            break
    cv2.destroyAllWindows()

    return mask
if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Helps you collect HSV Data')
    parser.add_argument('-p','--path', help='path to image', required=True)
    args = vars(parser.parse_args())
    colormask(args["path"])
#/media/YertleDrive4/layer_grasp/dataset/test/rgb/100.npy
#/media/YertleDrive4/layer_grasp/dataset/test/rgb/110.npy
#/media/YertleDrive4/layer_grasp/dataset/test/rgb/886.npy
#/media/YertleDrive4/layer_grasp/dataset/test/rgb/1056.npy



#Noting Down the values for different clothes: 
#We might want to subtract different masks to be smart about this -- to do later
#Blue hue [43 95] saturation [138 255] value[60 255]
#Red hue [0 179] saturation [72 218] value [94 179]
#Grey hue[0 179] saturation [0 255] value [0 100]
#white hue [65 108] saturation [88 120] value [104 255]
#Yellow hue [65 108] saturation [0 88] value [104 255]
#Background hue [0 40] saturation [0 255] value [0 255]


#Blue hue [77 107] sat [149 255] value [120 255]
#Grey hue [77 107] sat [0 255] value [0 115]