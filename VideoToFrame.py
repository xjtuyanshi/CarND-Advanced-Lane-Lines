import cv2
import os
frame_path = r"test_images\test_video_frames_2"
vidcap = cv2.VideoCapture('challenge_video.mp4')
success,image = vidcap.read()
count = 0
while success:
  image_name = "frame%d.jpg" % count
  cv2.imwrite(os.path.join(frame_path,image_name), image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1