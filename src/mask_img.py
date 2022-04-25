import numpy as np
import cv2
import argparse
#Blue hue [43 95] saturation [138 255] value[60 255]
#Red hue [0 179] saturation [72 218] value [94 179]
#Grey hue[0 179] saturation [0 255] value [0 100]
#white hue [65 108] saturation [88 120] value [104 255]
#Yellow hue [65 108] saturation [0 88] value [104 255]
#Background hue [0 40] saturation [0 255] value [0 255]
if(__name__ == "__main__"):
    img_nums = [0, 176, 277, 372, 535, 690, 918]
    for img_num in img_nums:
        img = cv2.imread("/media/YertleDrive4/layer_grasp/images/2022-04-18-16-39-37/"+str(img_num)+".jpg",1)
        color_list=["red","blue","yellow","white","grey","background"]
        color_values={
            "red" : [0, 0, 255],
            "blue" : [255, 0, 0],
            "yellow" : [0, 255, 255],
            "white" : [255, 255, 255],
            "grey" : [100, 100, 100],
            "background" : [0, 0, 0] 
        }
        mask_imgs={}
        mask_vals = {
            "red_high": np.array([179, 218, 179], np.uint8),
            "red_low": np.array([0, 72, 94], np.uint8),
            "blue_high": np.array([95, 255, 255], np.uint8),
            "blue_low": np.array([43, 138, 60], np.uint8),
            "yellow_high": np.array([108, 88, 255], np.uint8),
            "yellow_low": np.array([65, 0, 104], np.uint8),
            "white_high": np.array([108, 120, 255], np.uint8),
            "white_low": np.array([65, 88, 104], np.uint8),
            "grey_high": np.array([179, 255, 100], np.uint8),
            "grey_low": np.array([0, 0, 0], np.uint8),
            "background_high": np.array([40, 255, 255], np.uint8),
            "background_low": np.array([0, 0, 0], np.uint8),
        }
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_imgs["red"] = cv2.inRange(hsv_img, mask_vals["red_low"], mask_vals["red_high"])
        img1 = cv2.cvtColor(mask_imgs["red"], cv2.COLOR_GRAY2BGR)
        img1[np.where((img1==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        masked_imgs = []
        for color in color_list:
            mask_imgs[color] = cv2.inRange(hsv_img, mask_vals[color+"_low"], mask_vals[color+"_high"])
            mask_imgs[color] = cv2.cvtColor(mask_imgs[color], cv2.COLOR_GRAY2BGR)
            # mask_imgs[color][np.where((mask_imgs[color]==[255, 255, 255]).all(axis=2))] = color_values[color]
            masked_imgs.append(np.concatenate((img,mask_imgs[color]), axis=1))


        # hori = np.concatenate(masked_imgs, axis=1)
        i=0
        for img in masked_imgs:
            cv2.imshow(color_list[i], img)
            i+=1

        cv2.waitKey(0)
        cv2.destroyAllWindows()

#images to consider,0, 176, 277, 372, 535, 690, 918