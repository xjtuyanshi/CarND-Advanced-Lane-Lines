import numpy as np
import cv2
import matplotlib.pyplot as plt

def undistort_image(img, mtx, dist):
    # Undistort image
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
     Calculate directional gradient
     Apply threshold
    :param image: gray image
    :param orient:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """
     Calculate gradient magnitude
    :param image: gray-scaled image
    :param sobel_kernel:
    :param mag_thresh:
    :return:
    """
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
     Calculate gradient direction
    :param image:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def sobel_combined(image):
    """
    Apply combined sobel filter
    :param image:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mag_binary = mag_thresh(gray, 9, (20, 150))
    dir_binary = dir_threshold(gray, 9, (.6, 1.1))
    gradx = abs_sobel_thresh(gray, 'x', 9, (50, 200))
    grady = abs_sobel_thresh(gray, 'y', 95, (50, 200))
    sobel_combined = np.zeros_like(dir_binary)
    sobel_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return sobel_combined


def hls_select(img, thresh=(0, 255)):
    """
     Define a function that thresholds the S-channel of HLS
     Use exclusive lower bound (>) and inclusive upper (<=)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def LUV_select(img, thresh=(0, 255)):
    """
     Define a function that thresholds the L-channel of LUV
     Use exclusive lower bound (>) and inclusive upper (<=)
    """
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def LAB_select(img, thresh=(0, 255)):
    """
     Define a function that thresholds the B-channel of LAB
     Use exclusive lower bound (>) and inclusive upper (<=)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

def color_channel_combined(image):
    """
       Apply combined color channel result
       """
    s_binary = hls_select(image,(120,255))
    b_binary = LAB_select(image, (155,200))
    l_binary = LUV_select(image,(195,255))
    color_combined_output = np.zeros_like(s_binary)
    color_combined_output[(s_binary ==1) |(b_binary ==1 )|(l_binary ==1 ) ] = 1
    return color_combined_output

def sobel_color_combined(image):
    """
    Apply combined sobel and all selected color channels

    """
    sobel_combined_img = sobel_combined(image)
    color_channel_combined_img =color_channel_combined(image)
    color_channel_combined_binary= np.zeros_like(sobel_combined_img)
    color_channel_combined_binary[(sobel_combined_img == 1) | color_channel_combined_img == 1] = 1
    return color_channel_combined_binary
def get_warped_img(img):
    img_size = (img.shape[1], img.shape[0])
    # # road area of the image
    # mid_x = img_size[0] / 2
    # upper_y = img_size[1] / 1.5
    # lower_y = img_size[1]
    # upper_left_x = 0.8 * mid_x
    # lower_left_x = 0.22 * mid_x
    # upper_right_x = 1.25 * mid_x
    # lower_right_x = 1.95 * mid_x
    # src = np.float32([[upper_left_x, upper_y], [lower_left_x, lower_y],
    #                   [upper_right_x, upper_y], [lower_right_x, lower_y]])
    # # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # dst = np.float32([[0, 0], [0, img_size[1]], [img_size[0], 0], [img_size[0], img_size[1]]])

    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    #  get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    #  top-down view
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)

    return warped,Minv,src,dst


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def find_lane_pixels(binary_warped):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit, left_fitx, right_fitx, ploty, \
    left_curverad, right_curverad, center_diff, top_lane_width, bottom_lane_width = fit_poly(binary_warped.shape, leftx,
                                                                                             lefty, rightx, righty)
    polyfit_image = visualize_ployfit(binary_warped, left_lane_inds, right_lane_inds, left_fitx, right_fitx, nonzerox,
                                      nonzeroy, ploty,margin)
    return polyfit_image, left_fit, right_fit, left_fitx, right_fitx, ploty, left_curverad, right_curverad, center_diff, top_lane_width, bottom_lane_width


def search_around_poly(binary_warped, prev_frame_left_fit, prev_frame_right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 80
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (prev_frame_left_fit[0] * (nonzeroy ** 2) + prev_frame_left_fit[1] * nonzeroy +
                                   prev_frame_left_fit[2] - margin)) &
                      (nonzerox < (prev_frame_left_fit[0] * (nonzeroy ** 2) +prev_frame_left_fit[1] * nonzeroy +
                                   prev_frame_left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (prev_frame_right_fit[0] * (nonzeroy ** 2) + prev_frame_right_fit[1] * nonzeroy +
                                    prev_frame_right_fit[2] - margin)) &
                       ( nonzerox < (prev_frame_right_fit[0] * (nonzeroy ** 2) +prev_frame_right_fit[1] * nonzeroy +
                                     prev_frame_right_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty,\
    left_curverad, right_curverad, center_diff, top_lane_width, bottom_lane_width=fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    polyfit_image = visualize_ployfit(binary_warped, left_lane_inds, right_lane_inds, left_fitx, right_fitx, nonzerox, nonzeroy, ploty,
                      margin)
    return polyfit_image,left_fit,right_fit,left_fitx, right_fitx, ploty,left_curverad, right_curverad,center_diff,top_lane_width,bottom_lane_width


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    #Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    left_curverad, right_curverad, center_diff,top_lane_width,bottom_lane_width = measure_curvature_pixels(img_shape,leftx, lefty, righty, rightx)
    return left_fit,right_fit,left_fitx, right_fitx, ploty,left_curverad, right_curverad,center_diff,top_lane_width,bottom_lane_width

def measure_curvature_pixels(img_shape, leftx,lefty,righty,rightx):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define y-value where we want radius of curvature
    # choose the maximum y-value, corresponding to the bottom of the image
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = img_shape[0]
    # yvalue with unit meters
    ym= img_shape[0]*ym_per_pix
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * ym+ left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] *ym + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    center_car = xm_per_pix*img_shape[1]/2
    center_lane = ((right_fit_cr[0] * ym ** 2 + right_fit_cr[1] * ym + right_fit_cr[2])+(left_fit_cr[0] * ym ** 2 + left_fit_cr[1] * ym + left_fit_cr[2]))/2
    center_diff = center_lane-center_car
    # to get lane width bottom of image and top of the image
    bottom_leftx = (left_fit_cr[0] * ym ** 2 +  right_fit_cr[1]  * ym+ left_fit_cr[2])
    bottom_rightx = right_fit_cr[0] * ym ** 2 + right_fit_cr[1] * ym + right_fit_cr[2]
    top_leftx =  left_fit_cr[2]
    top_rightx = right_fit_cr[2]
    bottom_lane_width = (bottom_rightx-bottom_leftx)
    top_lane_width = top_rightx - top_leftx

    return left_curverad, right_curverad, center_diff,top_lane_width,bottom_lane_width
def visualize_ployfit(binary_warped,left_lane_inds,right_lane_inds,left_fitx,right_fitx,nonzerox,nonzeroy,ploty,margin):
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    return result


def draw_lanes(undist_image,warped,Minv,left_fitx,right_fitx,):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty =np.linspace(0, undist_image.shape[0]-1, undist_image.shape[0])
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (39,174,96))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist_image.shape[1], undist_image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp,0.8, 0)
    # plt.imshow(result)
    # plt.show()
    return result


def draw_info(diagnostic_mode,lanes,color_gradient_image,polyfit_image,left_curverad, right_curverad,center_dist,is_detected,use_last_n_frames_result):
    result = np.copy(lanes)

    # Add text to the original image
    font = cv2. FONT_HERSHEY_SIMPLEX
    text = 'Radius of left curvature: ' + '{:04.2f}'.format(left_curverad) + '(m)'
    cv2.putText(result, text, (40, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    text = 'Radius of right curvature: ' + '{:04.2f}'.format(right_curverad) + '(m)'
    cv2.putText(result, text, (40, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if center_dist >= 0:
        text = 'Vehicle is : {:04.2f}'.format(center_dist) + 'm right of center'
    else:
        text = 'Vehicle is : {:04.2f}'.format(abs(center_dist)) + 'm  left of center'
    cv2.putText(result, text, (40, 130), font, 1 , (255, 255, 255), 2, cv2.LINE_AA)
    #
    if is_detected:
        color = (46, 204, 113)
        text = "Good detect"
    elif not is_detected and use_last_n_frames_result:
        color = (255, 203, 5)
        text = "Bad detect,use previous frames"
    else:
        color = (207, 0, 15)
        text = "Bad detect, no previous frames to use"

    cv2.putText(result, text, (50, 200), font, 1, color, 2, cv2.LINE_AA)
    # Add bird eye view and poly fit images to the original image
    if diagnostic_mode:
        poly_fit_img = cv2.resize(polyfit_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        color_gradient_img_channels = np.uint8(np.dstack((color_gradient_image, color_gradient_image, color_gradient_image)) * 255)
        color_grad_img = cv2.resize(color_gradient_img_channels, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        end_y_poly_fit = 50 + poly_fit_img.shape[0]
        end_x_poly_fit = lanes.shape[1] - 50
        start_x_polyfit = end_x_poly_fit - poly_fit_img.shape[1]
        end_y_color_grad_img = end_y_poly_fit+50 +color_grad_img.shape[0]
        result[50:end_y_poly_fit, start_x_polyfit:end_x_poly_fit] = poly_fit_img
        result[end_y_poly_fit+50 :end_y_color_grad_img, start_x_polyfit:end_x_poly_fit] = color_grad_img
    return result