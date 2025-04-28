# +
import numpy as np
import cv2 as cv
import math

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

VECTOR_LENGTH = 60

def visualize_pose(frame, pose, center_hue=YELLOW):
    '''
    Returns a frame that, when displayed as an image
    Is the original frame with annotations showing the pose
    of the robot given by pose
    Parameters:
        frame: The original frame; a nxmx3 numpy array representing
            an RGB image on the scale 0 to 255
        pose: A 1x7 numpy array representing:
            [tracked_point_x,  tracked_point _y, shaft_axis_x, shaft_axis_y,
                head_axis_x, head_axis_y, clasper_angle]
    Returns:
        The annotated frame; a nxmx3 numpy array representing
            an RGB image on the scale 0 to 255
    '''
    for i in [0,1]:
        tracked_point_x = pose[i,0]
        tracked_point_y = pose[i,1]
        shaft_axis_x = pose[i,2]
        shaft_axis_y = pose[i,3]
        head_axis_x = pose[i,4]
        head_axis_y = pose[i,5]
        clasper_angle = pose[i,6]

        # Draw the shaft axis
        start_point = (round(tracked_point_x), round(tracked_point_y))
        end_point = (round(tracked_point_x + shaft_axis_x*VECTOR_LENGTH), \
                    round(tracked_point_y + shaft_axis_y*VECTOR_LENGTH))
        print(frame.shape)
        print(cv.line(frame, start_point, end_point, BLUE, 5).shape)
        frame = cv.line(frame, start_point, end_point, BLUE, 5)

        # Draw the tool axis
        end_point = (round(tracked_point_x + head_axis_x*VECTOR_LENGTH), \
                    round(tracked_point_y + head_axis_y*VECTOR_LENGTH))
        frame = cv.line(frame, start_point, end_point, RED, 5)

        # Draw the tracked point
        frame = cv.circle(frame, start_point, 10, center_hue, -1)

        # Draw the angle
        axesLength = (20,20)
        angle = math.atan2(head_axis_y, head_axis_x)
        startAngle = 0
        endAngle = clasper_angle
        frame = cv.ellipse(frame, start_point, axesLength, angle, startAngle, endAngle, GREEN, 5)

    return frame
