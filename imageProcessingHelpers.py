
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# +
def opening_and_closing(im):
    frame = im.cpu().detach().numpy().astype(np.uint8)*255
    # Edges are weird. Get them out
    frame[0:15,:] = 0
    frame[-15:-1,:] = 0
    frame[:,0:15] = 0
    frame[:,-15:-1] = 0


    # Dilate and erode
#     kernel = np.ones((2,2), np.uint8)
#     frame = cv.erode(frame, kernel, iterations=1) 
#     kernel = np.ones((5, 5), np.uint8)
#     frame = cv.dilate(frame, kernel, iterations=1) 
    kernel = np.ones((3, 3), np.uint8)
    frame = cv.erode(frame, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    frame = cv.dilate(frame, kernel, iterations=2)
    
    frame = torch.from_numpy(frame).float()
    
    return frame

def opening_and_closing_batched(im):
    frame = im.cpu().detach().permute(1,2,0).numpy().astype(np.uint8)*255
    # Edges are weird. Get them out
    frame[0:15,:,:] = 0
    frame[-15:-1,:,:] = 0
    frame[:,0:15,:] = 0
    frame[:,-15:-1,:] = 0


    # Dilate and erode
#     kernel = np.ones((2,2), np.uint8)
#     frame = cv.erode(frame, kernel, iterations=1) 
#     kernel = np.ones((5, 5), np.uint8)
#     frame = cv.dilate(frame, kernel, iterations=1) 
    kernel = np.ones((3, 3), np.uint8)
    frame = cv.erode(frame, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    frame = cv.dilate(frame, kernel, iterations=2)
    
    frame = torch.from_numpy(frame).float().permute(2,0,1)
    
    return frame
# -



# +
def get_largest_blobs(im, second=True):
    '''
    Helper function that returns the largest one or 2 blobs in image
    '''
    frame = im.cpu().detach().numpy().astype(np.uint8)
    # Keep the two biggest curves
    contours, hierarchy = cv.findContours(frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )

    min_contour = 5000
    maxContour = 0
    secondContour = 0
    maxContourData = None
    secondContourData = None
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > secondContour and contourSize > min_contour:
            if contourSize > maxContour:
                secondContour = maxContour
                secondContourData = maxContourData
                maxContour = contourSize
                maxContourData = contour
            else:
                secondContour = contourSize
                secondContourData = contour

    left_guess = np.zeros(frame.shape, np.uint8)
    right_guess = np.zeros(frame.shape, np.uint8)
    if secondContour > 0 and second:
        # Find which is right and left
        M_first = cv.moments(maxContourData)
        cX_first = int(M_first["m10"] / M_first["m00"])

        M_second = cv.moments(secondContourData)
        cX_second = int(M_second["m10"] / M_second["m00"])

        if cX_second > cX_first: # first is left
            cv.drawContours(left_guess, [maxContourData], -1, 255, cv.FILLED)
            cv.drawContours(right_guess, [secondContourData], -1, 255, cv.FILLED)
        else: # first is right
            cv.drawContours(left_guess, [secondContourData], -1, 255, cv.FILLED)
            cv.drawContours(right_guess, [maxContourData], -1, 255, cv.FILLED)
    elif maxContour > 0:
        cv.drawContours(right_guess, [maxContourData], -1, 255, cv.FILLED)

    return torch.from_numpy(left_guess).float(), torch.from_numpy(right_guess).float()


def get_largest_blobs_batched(im, second=True):
    '''
    Helper function that returns the largest one or 2 blobs in image
    '''
    frame = im.cpu().detach().permute(1,2,0).numpy().astype(np.uint8)
    # Keep the two biggest curves
    contours, hierarchy = cv.findContours(frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )

    min_contour = 5000
    maxContour = 0
    secondContour = 0
    maxContourData = None
    secondContourData = None
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > secondContour and contourSize > min_contour:
            if contourSize > maxContour:
                secondContour = maxContour
                secondContourData = maxContourData
                maxContour = contourSize
                maxContourData = contour
            else:
                secondContour = contourSize
                secondContourData = contour

    left_guess = np.zeros(frame.shape, np.uint8)
    right_guess = np.zeros(frame.shape, np.uint8)
    if secondContour > 0 and second:
        # Find which is right and left
        M_first = cv.moments(maxContourData)
        cX_first = int(M_first["m10"] / M_first["m00"])

        M_second = cv.moments(secondContourData)
        cX_second = int(M_second["m10"] / M_second["m00"])

        if cX_second > cX_first: # first is left
            cv.drawContours(left_guess, [maxContourData], -1, 255, cv.FILLED)
            cv.drawContours(right_guess, [secondContourData], -1, 255, cv.FILLED)
        else: # first is right
            cv.drawContours(left_guess, [secondContourData], -1, 255, cv.FILLED)
            cv.drawContours(right_guess, [maxContourData], -1, 255, cv.FILLED)
    elif maxContour > 0:
        cv.drawContours(right_guess, [maxContourData], -1, 255, cv.FILLED)

    return torch.from_numpy(left_guess).float().permute(2,0,1), torch.from_numpy(right_guess).float().permute(2,0,1)
