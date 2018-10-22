import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
import helperFunctions as hp
import os
from helperFunctions import *
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,last_n_frames = 5 ,max_n_fails =10):
        # keep parameters of last n frames
        self.last_n_frames = last_n_frames
        # Max # of failed detected, restart to search lines
        self.max_n_fails = max_n_fails
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients over the last n iterations
        self.recent_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the this  fit
        self.current_fit = [np.array([False])]
        # polynomial coefficients for the last fit
        self.last_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # n accumulated fails
        self.n_fails = 0
    def reset(self):
        self.detected = False
        self.best_fit = None
        self.n_fails = 0
        self.radius_of_curvature = None
        self.center_dist=0
        self.lane_width =0
        self.recent_xfitted = []
        self.recent_fit = []
        self.current_fit =[np.array([False])]
        self.last_fit = None

    def sanity_check(self, img_shape, current_fit,last_fit, left_curverad, right_curverad,top_lane_width,bottom_lane_width):
        # Checking that they have similar curvature
        #print(str(left_curverad) +"," +str(right_curverad) +","+ str(top_lane_width) +","+ str(bottom_lane_width))
        if abs(left_curverad - right_curverad) >3000:
            return False
        # Checking that they are separated by approximately the right distance horizontally
        if abs(top_lane_width - 3.7) >1 and abs(bottom_lane_width - 3.7)  > 1:
            return False

        # Checking that they are roughly parallel
        if abs(top_lane_width -bottom_lane_width)> 2:
            return False
        if last_fit is not None:
            self.diffs = (current_fit[0]-last_fit[0])+(current_fit[1]-last_fit[1])
            error_diff = abs(self.diffs[0])
            if error_diff > .0005:
                return False

        return True

    def process_image(self,img,diagnostic_mode =True):
        diagnostic_mode_on =diagnostic_mode
        # 1. undistorted images
        undist = undistort_image(img, mtx, dist)
        # 2. get perspective view
        warped_img,Minv,src,dst= get_warped_img(undist)
        # 3. color threshold  and sobel combined
        sobel_color_combined_binary = hp.color_channel_combined(warped_img)
        # 4. find lanes
        if self.best_fit is None or  self.n_fails >= self.max_n_fails:
            self.reset()
            polyfit_image,left_fit, right_fit, left_fitx, right_fitx, ploty, left_curverad, right_curverad, \
            center_diff, top_lane_width, bottom_lane_width = find_lane_pixels(sobel_color_combined_binary)
        else:
            try:
                polyfit_image,left_fit, right_fit, left_fitx, right_fitx, ploty, left_curverad, right_curverad,\
                center_diff, top_lane_width, bottom_lane_width= search_around_poly(
                    sobel_color_combined_binary,self.best_fit[0], self.best_fit[1])
            except:
                polyfit_image, left_fit, right_fit, left_fitx, right_fitx, ploty, left_curverad, right_curverad, \
                center_diff, top_lane_width, bottom_lane_width = find_lane_pixels(sobel_color_combined_binary)

        self.current_fit = (left_fit, right_fit)

        if self.sanity_check(img.shape,self.current_fit,self.last_fit,left_curverad,right_curverad,top_lane_width,bottom_lane_width):
            self.detected = True
            if len(self.recent_xfitted) >= self.last_n_frames:
                self.recent_xfitted.pop(0)
                self.recent_fit.pop(0)
            self.recent_xfitted.append((left_fitx,right_fitx))
            self.recent_fit.append(self.current_fit)
            self.bestx = np.average(self.recent_xfitted,axis=0)
            self.best_fit =  np.average(self.recent_fit,axis=0)
            self.radius_of_curvature = (left_curverad, right_curverad)
            self.center_dist = center_diff
            self.lane_width = (top_lane_width + bottom_lane_width) / 2
            final_image =draw_lanes(undist,sobel_color_combined_binary,Minv,left_fitx,right_fitx)
            result = draw_info(diagnostic_mode_on,final_image, sobel_color_combined_binary, polyfit_image, left_curverad, right_curverad,
                               center_diff,True, False)
        else:# use the average values of last n frames
            self.n_fails += 1
            if self.bestx is not None:
                final_image = draw_lanes(undist, sobel_color_combined_binary, Minv, self.bestx[0], self.bestx[1])
                result =draw_info(diagnostic_mode_on,final_image, sobel_color_combined_binary, polyfit_image, left_curverad, right_curverad, center_diff,
                          False, True)
            else:
                final_image = draw_lanes(undist, sobel_color_combined_binary, Minv, left_fitx, right_fitx)
                result = draw_info(diagnostic_mode_on,final_image, sobel_color_combined_binary, polyfit_image, left_curverad, right_curverad,
                                   center_diff, False, False)
        self.last_fit = self.current_fit
        return result


if __name__ =="__main__" :
    dist_pickle = pickle.load(open("calibrated_camera_info.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    images = glob.glob('test_images/*.jpg')
    # for test_img_file in images:
    #     line = Line()
    #     img = cv2.cvtColor(cv2.imread(test_img_file), cv2.COLOR_BGR2RGB)
    #     result =line.process_image(img,False)
    #     plt.imshow(result)
    #     plt.show()

    videos =  glob.glob('*.mp4')
    for file_name in videos:
        print("working on video " +file_name)
        line = Line()
        #clip1 = VideoFileClip(file_name).subclip(0, 15)
        clip1 = VideoFileClip(file_name)
        white_clip = clip1.fl_image(line.process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(file_name.split(".")[0] +"_output.mp4", audio=False)