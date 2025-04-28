# +
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math

def model_output_to_prediction_ce(output):
    print(output.shape)
    maxes, _ = output.max(-3, keepdim=True)
    print(maxes.shape)
    mask = output >= maxes
    return mask.float()
    
def model_output_to_prediction_bce(output, threshold):
    output = F.sigmoid(output)
    return (output > threshold).float()



# -

def iou(output, truth):
    '''
    params:
        output: BxCxHxW tensor model predicted output for C classes, B batches, of height H and width W, one hot in C
        truth: BxCxHxW tensor ground truth for each of C classes, B batches, of height H and width W
    output:
        iou: BxC tensor IOU score for each class
    '''
    epsilon = 1e-6

    truth = truth > 0 # convert to boolean
    output = output > 0
   
        
    intersection = (output & truth).float().sum((-2, -1))  # Will be zero if Truth=0 or Prediction=0
    union = (output | truth).float().sum((-2, -1))         # Will be zero if both are 0
    
    iou = (intersection) / (union + epsilon)
    
    return iou


def dice(output, truth):
    '''
    params:
        output: BxCxHxW tensor model predicted output for C classes, B batches, of height H and width W, one hot in C
        truth: BxCxHxW tensor ground truth for each of C classes, B batches, of height H and width W
    output:
        dice: BxC tensor dice score for each class
    '''
    epsilon = 1e-6
    
    truth = truth > 0 # convert to boolean
    output = output > 0 # convert to boolean
    
    intersection = (output & truth).float().sum((-2, -1))
    truth_count = truth.sum((-2, -1))
    output_count = output.sum((-2, -1))
    
    dice = (2*intersection)/(truth_count + output_count + epsilon)
    
    return dice


def make_histograms(data, labels, xlabel=["Rate"], ylabel=["Count"], xlim=None, n_bins=20):
    '''
    Makes a figure with a histogram subplot for each column in data.
    Parameters:
    data: A numpy array with n columns. n subplots will be created.
    labels: A list of n strings, the titles for each plot
    xlabel: the x axis label to be used for all subplots. Default is "Rate"
    ylabel: the y axis label to be used for all plots. Default is "Count"
    xlim: A list of length 2 with the lower and upper bound of the x axis.
    n_bins: Number of histogram bins to use, default 20.
    returns: None
    '''
    print(data.shape)

    figure = plt.figure()
    for i in range(0,len(labels)):
        print(i)
        currentData = data[:,i]
        currentData =  currentData[~currentData.isnan()]
        print(labels[i] + " Mean: %f" % torch.mean(currentData))
#        print(labels[i] + " RMS: %f" % np.sqrt(np.mean(np.square(currentData))))
#         print(labels[i] + " Percent Within 10 units: %f %%" % (100*(np.sum(np.absolute(currentData) <= 10)/len(currentData))))
#         print(labels[i] + " Median absolute error: %f" % np.median(np.absolute(currentData)))
        #plt.subplot(-(data.shape[1]//-2), 2, i+1)
        #plt.subplot(1, len(labels), i+1)
        plt.subplot(2, 3, i+1)
        plt.hist(currentData, bins=n_bins, range=xlim)
        plt.xlim(xlim)
        plt.xlabel(xlabel[i] if len(xlabel) > 1 else xlabel[0])
        plt.ylabel(ylabel[i]if len(ylabel) > 1 else ylabel[0])
        plt.title(labels[i])
        

    plt.tight_layout()
    
    plt.show()

def show_images(original, truth, output, labels):
    C = truth.shape[0]
    print("Original")
    print(original.shape)
    
    figure = plt.figure()
    plt.imshow(original.permute(1, 2, 0))
    plt.title("Original Image")
    plt.show()

    print("Truth:")
    print(truth.shape)
    print(torch.min(truth))
    print(torch.max(truth))
    print("Ground Truth:")
    figure = plt.figure()
    for i in range(0,C):
        plt.subplot(1,C,i+1)
        plt.imshow(truth[i,...], vmin=0, vmax=1, cmap='gray')
        plt.title(labels[i])
    plt.show()
    
    print("Output:")
    print(output.shape)
    print(torch.min(output))
    print(torch.max(output))
    figure = plt.figure()
    for i in range(0,C):
        plt.subplot(1,C,i+1)
        plt.imshow(output[i,...], cmap='gray') #, vmin=0, vmax=1, )
        plt.title(labels[i])
    plt.show()


def get_tracking_error(truth, guess):
    truth_tracked_point_x = truth[0]
    truth_tracked_point_y = truth[1]
    truth_shaft = np.array([truth[2], truth[3]])
    truth_head = np.array([truth[4], truth[5]])
    truth_clasper_angle = truth[6]

    guess_tracked_point_x = guess[0]
    guess_tracked_point_y = guess[1]
    guess_shaft = np.array([guess[2], guess[3]])
    guess_head = np.array([guess[4], guess[5]])
    guess_clasper_angle = guess[6]

    dx = guess_tracked_point_x - truth_tracked_point_x
    dy = guess_tracked_point_y - truth_tracked_point_y
    d_euclidean = (dx**2 + dy**2) **.5

    try:
        d_shaft_angle = math.acos(np.dot(truth_shaft, guess_shaft)/(np.linalg.norm(truth_shaft)*np.linalg.norm(guess_shaft)))*180/math.pi
        if d_shaft_angle > 90:
            d_shaft_angle = 180 - d_shaft_angle
    except:
        d_shaft_angle = 0

    try:
        d_head_angle = math.acos(np.dot(truth_head, guess_head)/(np.linalg.norm(truth_head)*np.linalg.norm(guess_head)))*180/math.pi
        if d_head_angle > 90:
            d_head_angle = 180 - d_head_angle
    except:
        d_head_angle = 0

    d_clasper = truth_clasper_angle - guess_clasper_angle

    return dx, dy, d_shaft_angle, d_head_angle, d_euclidean, d_clasper      
