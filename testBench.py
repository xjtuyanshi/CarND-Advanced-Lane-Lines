import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
import helperFunctions as hp
import os
dist_pickle = pickle.load(open("calibrated_camera_info.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
images = glob.glob('test_images/test_video_frames_3/*.jpg')
#images = glob.glob('test_images/*.jpg')

def test_wraped_effect(origin_img,warped_img,src,dst,img_name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(origin_img, cmap='gray')
    for i in range(4):
        plt.plot(src[i][0], src[i][1], 'rs')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(warped_img, cmap='gray')
    for i in range(4):
        plt.plot(dst[i][0], dst[i][1], 'rs')
    plt.title('Bird-eye view')

    plt.savefig(os.path.join("testPerspectiveTransformedImages", img_name))
    plt.show()


# test perspective transform
def test_curvature(images):
    for test_img in images:
        img = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2RGB)
        undist = hp.undistort_image(img, mtx, dist)
        warped_img, minV, src, dst = hp.get_warped_img(undist)
        sobel_color_combined_binary = hp.sobel_color_combined(warped_img)
        polyfit_image, left_fit, right_fit, left_fitx, right_fitx, ploty, \
        left_curverad, right_curverad, center_diff,  top_lane_width, bottom_lane_width = hp.find_lane_pixels(sobel_color_combined_binary)
        f, ((ax1, ax2)) = plt.subplots(1, 2, sharey='col', sharex='row', figsize=(10, 4))
        f.tight_layout()

        ax1.set_title('Warped Image', fontsize=16)
        ax1.imshow(warped_img, cmap='gray')
        ax2.set_title('Poly fitted Image', fontsize=16)
        ax2.imshow(polyfit_image, cmap='gray')
        plt.show()

def warp_test(images):
    counter = 0
    for test_img in images:
        img = mpimg.imread(test_img)
        #undistort images
        undist = hp.undistort_image(img, mtx, dist)
        warped_img, minV, src, dst = hp.get_warped_img(undist)
        test_wraped_effect(img,warped_img,src,dst,test_img)
        hls_color_thres = hp.hls_select(warped_img, thresh=(150, 255))
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undist , cmap='gray')
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(hls_color_thres, cmap='gray')
        ax2.set_title('after', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        counter += 1
        if counter > 600 :
            break

def color_gradient_test(images):
    for test_img in images:
        img = cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2RGB)
        undist = hp.undistort_image(img, mtx, dist)
        warped_img, minV, src, dst = hp.get_warped_img(undist)
        sobel_combined_binary = hp.sobel_combined(warped_img)
        s_binary = hp.hls_select(warped_img, (120, 255))
        b_binary = hp.LAB_select(warped_img, (155, 200))
        l_binary = hp.LUV_select(warped_img, (205, 255))
        color_combined_binary = hp.color_channel_combined(warped_img)
        sobel_color_combined_binary = hp.sobel_color_combined(warped_img)
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6),(ax7, ax8, ax9)) = plt.subplots(3, 3, sharey='col', sharex='row', figsize=(10, 4))
        f.tight_layout()

        ax1.set_title('Undist Image', fontsize=16)
        ax1.imshow(undist, cmap='gray')

        ax2.set_title('Warped Image', fontsize=16)
        ax2.imshow(warped_img,cmap='gray')

        ax3.set_title('sobel binary combined', fontsize=16)
        ax3.imshow(sobel_combined_binary, cmap='gray')

        ax4.set_title('s binary threshold', fontsize=16)
        ax4.imshow(s_binary, cmap='gray')
        ax5.set_title('l binary threshold', fontsize=16)
        ax5.imshow(l_binary, cmap='gray')
        ax6.set_title('b binary threshold', fontsize=16)
        ax6.imshow(b_binary, cmap='gray')

        ax7.set_title('Combined color thresholds', fontsize=16)
        ax7.imshow(color_combined_binary, cmap='gray')

        ax8.set_title('Combined sobel color thresholds', fontsize=16)
        ax8.imshow(sobel_color_combined_binary, cmap='gray')
        plt.show()

#warp_test(images)
#test_curvature(images)
color_gradient_test(images)