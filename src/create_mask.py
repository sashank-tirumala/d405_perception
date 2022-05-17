import numpy as np
import cv2
import argparse
import IPython
import os
import shutil
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#Blue hue [77 107] sat [149 255] value [120 255]
#Grey hue [77 107] sat [0 255] value [0 115]
def generate_mask_for_image(rgb_img, mask_vals, color_label):
    """
    input:
    rgb_img = OpenCV RGB Image
    mask_vals = Dictionary that contains HSV Values for rgb_img
    color_label = Matches a color to a label
    output:
    masked_img : 2D with each pixel given a particular label
    """
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    mask_imgs={}
    color_list = list(color_label.keys())
    mask_img_list = []
    for color in color_list:
        mask_imgs[color] = cv2.inRange(hsv_img, mask_vals[color+"_low"], mask_vals[color+"_high"])
        # mask_imgs[color] = cv2.cvtColor(mask_imgs[color], cv2.COLOR_GRAY2BGR)
        # print(mask_imgs[color][120:130, 120:130])
        mask_imgs[color][np.where(mask_imgs[color]==255)] = color_label[color]
        mask_img_list.append(mask_imgs[color])
    mask_img_list = np.stack(mask_img_list, axis=-1)
    final_img = np.amax(mask_img_list, axis=-1)
    print(mask_img_list.shape)
    return mask_img_list, final_img

def visualize_image(rgb_ori_img, mask_img, color_label, color_values):
    rgb_img = np.stack([mask_img]*3, axis=-1)
    color_list = list(color_label.keys())
    color_val = list(color_label.values())
    for i in range(len(color_list)):
        color = color_list[i]
        val = color_val[i]
        rgb_img[np.where((rgb_img==[val]*3).all(axis=2))] = color_values[color]
    hori = np.concatenate((rgb_ori_img, rgb_img), axis=1)
    cv2.imshow("img", hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_dataset(dataset_dir, mask_vals ,color_label):
    rgb_dir = dataset_dir + "/rgb"
    images = os.listdir(rgb_dir)
    mask_dir = dataset_dir + "/masks"
    if(os.path.isdir(mask_dir)):
        print("Mask Directory already exists!")
        print("Deleting and creating again!")
        shutil.rmtree(mask_dir, ignore_errors=True)
    os.makedirs(mask_dir)
    color_val = list(color_label.values())
    for val in color_val:
        dirn = mask_dir+"/"+str(val)
        os.makedirs(dirn)
    for i in range(len(images)):
        # print(rgb_dir+"/"+images[i])
        img = np.load(rgb_dir+"/"+images[i])
        # print(img.shape)
        mask, _ = generate_mask_for_image(img, mask_vals, color_label)
        for j in range(mask.shape[-1]):
            val = int(np.max(mask[:,:,j]))
            save_path = mask_dir+"/"+str(val)
            temp = mask[:,:,j]/val
            np.save(save_path+"/"+images[i],temp)
        i+=1
    print("done generating dataset")

def test_masks(dataset_dir):
    mask_vals = {
            "blue_low": np.array([77, 149, 120], np.uint8),
            "blue_high": np.array([107, 255, 255], np.uint8),
            "grey_low": np.array([77, 0, 0], np.uint8),
            "grey_high": np.array([107, 255, 115], np.uint8),
        }
    color_label ={
        'blue':2,
        'grey':1
    }
    color_values={
            "blue" : [255, 0, 0],
            "grey" : [100, 100, 100],
        }
    rgb_dir = dataset_dir + "/rgb"
    images = os.listdir(rgb_dir)[:5]
    print(images)
    mask_dir = dataset_dir + "/masks"
    mask_len = len(os.listdir(mask_dir))
    fname = os.listdir(mask_dir+"/"+os.listdir(mask_dir)[0])[0]
    frgb = rgb_dir+"/"+fname
    img_rgb = np.load(frgb)
    mask_imgs=[]
    print(mask_len)
    for i in range(1, mask_len+1):
        mask_img = np.load(mask_dir+"/"+str(i)+"/"+fname)*i
        mask_imgs.append(mask_img)

    mask_img_list, final_img = generate_mask_for_image(img_rgb, mask_vals, color_label)
    # for i in range(len(mask_img_list)):
    #     im1 = mask_img_list[i]
    #     im2 = mask_imgs[i]
    #     print(im1 == im2)
    mask_imgs = np.stack(mask_imgs, axis=-1)
    final_img2 = np.amax(mask_imgs, axis=-1)
    print((final_img == final_img2).all())

    # visualize_image(img_rgb, final_img2, color_label, color_values)

def visualize_mask(img_rgb, mask_vals, color_label):
    mask_img_list, final_img = generate_mask_for_image(img_rgb, mask_vals, color_label)
    img_rgb = img_rgb[..., ::-1]
    cols = mask_img_list.shape[-1]
    rows=1
    fig = plt.figure()
    fig.tight_layout()
    ax = []
    ax.append(fig.add_subplot(1, cols+1, 1) )
    ax[-1].set_title("RGB Image")
    plt.imshow(img_rgb)
    for i in range(cols):
        ax.append(fig.add_subplot(1, cols+1, i+2) )
        cur_mask = mask_img_list[:,:,i]
        val = np.max(cur_mask)
        cur_mask[np.where(cur_mask == val)] = 255
        ax[-1].set_title(str(i+2))
        plt.imshow(cur_mask)
    
    plt.savefig("visualize_mask")
    plt.close()
    plt.show()    

if(__name__ == "__main__"):
    # test_masks("/media/YertleDrive4/layer_grasp/dataset/test")
    img = np.load("/media/YertleDrive4/layer_grasp/dataset/4cloth/rgb/217.npy")
    # mask_vals = {
    #         "blue_low": np.array([77, 149, 120], np.uint8),
    #         "blue_high": np.array([107, 255, 255], np.uint8),
    #         "grey_low": np.array([77, 0, 0], np.uint8),
    #         "grey_high": np.array([107, 255, 115], np.uint8),
    #     }
    mask_vals = {
        "blue_low": np.array([86, 189, 56], np.uint8),
        "blue_high": np.array([105, 255, 255], np.uint8),
        "grey_low": np.array([93, 35, 36], np.uint8),
        "grey_high": np.array([134, 207, 160], np.uint8),
        "white_low": np.array([80, 36, 166], np.uint8),
        "white_high": np.array([110, 166, 255], np.uint8),
        "green_low": np.array([21, 105, 54], np.uint8),
        "green_high": np.array([86, 220, 255], np.uint8),

    }
    # color_label ={
    #     'blue':2,
    #     'grey':1
    # }
    color_label ={
        'grey':4,
        'green':3,
        'white':2,
        'blue':1
    }
    color_values={
            "blue" : [255, 0, 0],
            "grey" : [100, 100, 100],
        }
    # generate_dataset("/media/YertleDrive4/layer_grasp/dataset/test", mask_vals, color_label)
    visualize_mask(img, mask_vals, color_label)
    # mask_img_list, final_img = generate_mask_for_image(img, mask_vals, color_label)
    # visualize_image(img, final_img, color_label, color_values )
    # img_nums = [0, 176, 277, 372, 535, 690, 918]
    # for img_num in img_nums:
    #     img = cv2.imread("/media/YertleDrive4/layer_grasp/images/2022-04-18-16-39-37/"+str(img_num)+".jpg",1)
    #     color_list=["red","blue","yellow","white","grey","background"]
    #     color_values={
    #         "red" : [0, 0, 255],
    #         "blue" : [255, 0, 0],
    #         "yellow" : [0, 255, 255],
    #         "white" : [255, 255, 255],
    #         "grey" : [100, 100, 100],
    #         "background" : [0, 0, 0] 
    #     }
    #     mask_imgs={}
    #     mask_vals = {
    #         "red_high": np.array([179, 218, 179], np.uint8),
    #         "red_low": np.array([0, 72, 94], np.uint8),
    #         "blue_high": np.array([95, 255, 255], np.uint8),
    #         "blue_low": np.array([43, 138, 60], np.uint8),
    #         "yellow_high": np.array([108, 88, 255], np.uint8),
    #         "yellow_low": np.array([65, 0, 104], np.uint8),
    #         "white_high": np.array([108, 120, 255], np.uint8),
    #         "white_low": np.array([65, 88, 104], np.uint8),
    #         "grey_high": np.array([179, 255, 100], np.uint8),
    #         "grey_low": np.array([0, 0, 0], np.uint8),
    #         "background_high": np.array([40, 255, 255], np.uint8),
    #         "background_low": np.array([0, 0, 0], np.uint8),
    #     }
    #     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     mask_imgs["red"] = cv2.inRange(hsv_img, mask_vals["red_low"], mask_vals["red_high"])
    #     img1 = cv2.cvtColor(mask_imgs["red"], cv2.COLOR_GRAY2BGR)
    #     img1[np.where((img1==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
    #     masked_imgs = []
    #     for color in color_list:
    #         mask_imgs[color] = cv2.inRange(hsv_img, mask_vals[color+"_low"], mask_vals[color+"_high"])
    #         mask_imgs[color] = cv2.cvtColor(mask_imgs[color], cv2.COLOR_GRAY2BGR)
    #         # mask_imgs[color][np.where((mask_imgs[color]==[255, 255, 255]).all(axis=2))] = color_values[color]
    #         masked_imgs.append(np.concatenate((img,mask_imgs[color]), axis=1))


    #     # hori = np.concatenate(masked_imgs, axis=1)
    #     i=0
    #     for img in masked_imgs:
    #         cv2.imshow(color_list[i], img)
    #         i+=1

    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

#images to consider,0, 176, 277, 372, 535, 690, 918