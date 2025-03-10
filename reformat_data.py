import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
import shutil
import subprocess
import imageio

def yield_video(path):
    '''
    A generator. Set up a for loop of the form 

    for frame in yield_video("my\\path\\video.avi"):
        # Do something

    path: the string path to the video
    yields: an mxnx3 numpy array where
        - m is the height of each video frame
        - n is the width of each frame
        - 3 is for three color channels, R, G, B
    '''
    print(path)
    video = imageio.get_reader(path)
    for frame in video:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB).transpose(2, 0, 1)
        yield frame
#     print(path)
#     cap = cv.VideoCapture(path)
#     #cap = cv.cudacodec.VideoReader(path)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     while True:
#         ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
#         if ret:
#             img = cv.cvtColor(img, cv.COLOR_BGR2RGB).transpose(2, 0, 1)
#             yield img
#         else:
#             break

def yield_pose(path):
    '''
    A generator. Set up a for loop of the form 

    for frame in yield_pose("my\\path\\pose.txt"):
        # Do something

    path: the string path to the txt file containing poses in csv format
    yields: an 1xn numpy array where
        - n is the number of fields in the pose description table
    '''
    pose_data = np.loadtxt(path, delimiter=' ', usecols=(0,1,2,3,4,5,6))
    for row in pose_data:
        #print(row.shape)
        yield np.expand_dims(row, axis=0)


def yield_videos(path):
    '''
    A generator. Set up a for loop of the form 

    for frame in yield_videos("my\\path\\folder"):
        # Do something

    path: the string path to the folder in which to search for videos
    yields: an mxnx3 numpy array where
        - m is the height of each video frame
        - n is the width of each frame
        - 3 is for three color channels, R, G, B
    '''
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1].lower() == '.avi':
                file_path = os.path.join(subdir, file)
                for frame in yield_video(file_path):
                    yield frame.transpose(2, 0, 1)

def yield_zeros():
    '''
    Helper function that just yields a zero array forever
    Don't ask
    '''
    while True:
        yield np.zeros((3, 576, 720), dtype=np.uint8)

def yield_empty_pose():
    '''
    Helper function that just yields a 1x7 zero array forever
    Don't ask
    '''
    while True:
        yield np.zeros((1,7), dtype=np.uint8)

def load_video(path):
    '''
    Loads a single entire video
    Parameters:
    path: the string path to the video
    returns: an fxmxnx3xp numpy array where
        - f is the total number of frames in the video
        - m is the height of each video frame
        - n is the width of each frame
        - 3 is for three color channels, R, G, B        
        The values will be floating point between 0 and 1. 
    '''
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frames = []
    while True:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB).transpose(2, 0, 1)
            frames.append(img)
        else:
            break
    video = np.stack(frames, axis=0)
    
    # When everything done, release the capture
    cap.release()

    return video


def convert_segmenation_to_numpy(path, outname):
    '''
    Loads all videos for segmentation training
    Parameters:
    path: the string path to the folder to search
    returns: None, saves a npz file called segmentaiton_training
    with 
    data: an fx3xmxn numpy array where
        - f is the total number of frames in the video
        - 3 is for three color channels, R, G, B 
        - m is the height of each video frame
        - n is the width of each frame
        The values will be floating point between 0 and 1. 
    left_mask: an fx1xmxn numpy array where
        - f is the total number of frames in the video
        - 1 channel
        - m is the height of each video frame
        - n is the width of each frame
        The values will be binary 0 or 1. 
    right_mask: either None, or an fx1xmxn numpy array where
        - f is the total number of frames in the video
        - 1 channel
        - m is the height of each video frame
        - n is the width of each frame
        The values will be binary 0 or 1. 
    '''
    data_array = []
    left_mask_array = []
    right_mask_array = []
    i = 0
    for subdir, dirs, files in os.walk(path):

        if 'Video1.avi' in files:
            video_path = os.path.join(subdir, 'Video1.avi')
            vid_generator = yield_video(video_path)

            left_generator = None
            right_generator = None
            if 'Segmentation1.avi' in files:
                left_path = os.path.join(subdir, 'Segmentation1.avi')
                left_generator = yield_video(left_path)
            elif 'Left_Instrument_Segmentation1.avi' in files:
                left_path = os.path.join(subdir, 'Left_Instrument_Segmentation1.avi')
                left_generator = yield_video(left_path)
            if 'Right_Instrument_Segmentation1.avi' in files:
                right_path = os.path.join(subdir, 'Right_Instrument_Segmentation1.avi')
                right_generator = yield_video(right_path)
            else:
                right_generator = yield_zeros()

            for frame in vid_generator:
                data_array.append(frame)
                left_mask_array.append(next(left_generator))
                right_mask_array.append(next(right_generator))
                i += 1
        print(".")
        
    data = np.stack(data_array, axis=0)
    left_mask = np.stack(left_mask_array, axis=0)
    right_mask = np.stack(right_mask_array, axis=0)

    np.savez(os.path.join(path, outname), data=data, left_mask=left_mask, right_mask=right_mask)

def convert_tracking_to_numpy(path, outname):
    data_array = []
    left_pose_array = []
    right_pose_array = []
    for subdir, dirs, files in os.walk(path):

        if 'Video1.avi' in files:
            video_path = os.path.join(subdir, 'Video1.avi')
            vid_generator = yield_video(video_path)

            left_generator = None
            right_generator = None
            if 'Pose.txt' in files:
                left_path = os.path.join(subdir, 'Pose.txt')
                left_generator = yield_pose(left_path)
            elif 'Left_Instrument_Pose.txt' in files:
                left_path = os.path.join(subdir, 'Left_Instrument_Pose.txt')
                left_generator = yield_pose(left_path)
            if 'Right_Instrument_Pose.txt' in files:
                right_path = os.path.join(subdir, 'Right_Instrument_Pose.txt')
                right_generator = yield_pose(right_path)
            else:
                right_generator = yield_empty_pose()

            for frame in vid_generator:
                data_array.append(frame)
                left_pose_array.append(next(left_generator))
                right_pose_array.append(next(right_generator))
        print(".")
    data = np.stack(data_array, axis=0)
    left_pose = np.stack(left_pose_array, axis=0)
    right_pose = np.stack(right_pose_array, axis=0)

    np.savez(os.path.join(path, outname), data=data, left_pose=left_pose, right_pose=right_pose)

def binarize_mask(mask_frame):
    # turn 3xhxw greyscale into 2xhxw mask
    # first channel for shaft
    # second channel for head
    new_mask = np.empty((2,mask_frame.shape[1], mask_frame.shape[2]))
    new_mask[0,:,:] = mask_frame[1,:,:] > 150 # shaft
    new_mask[1,:,:] =  mask_frame[1,:,:] > 60 - new_mask[0,:,:] # head
    return new_mask

def convert_segmenation_to_numpy_individual(path, outpath):
    '''
    Loads all videos for segmentation training
    Parameters:
    path: the string path to the folder to search
    returns: None, saves a npz file called segmentaiton_training
    with 
    data: an fx3xmxn numpy array where
        - f is the total number of frames in the video
        - 3 is for three color channels, R, G, B 
        - m is the height of each video frame
        - n is the width of each frame
        The values will be floating point between 0 and 1. 
    left_mask: an fx1xmxn numpy array where
        - f is the total number of frames in the video
        - 1 channel
        - m is the height of each video frame
        - n is the width of each frame
        The values will be binary 0 or 1. 
    right_mask: either None, or an fx1xmxn numpy array where
        - f is the total number of frames in the video
        - 1 channel
        - m is the height of each video frame
        - n is the width of each frame
        The values will be binary 0 or 1. 
    '''

    i = 0
    for subdir, dirs, files in os.walk(path):

        if 'Video1.avi' in files:
            video_path = os.path.join(subdir, 'Video1.avi')
            vid_generator = yield_video(video_path)

            left_generator = None
            right_generator = None
            if 'Segmentation1.avi' in files:
                left_path = os.path.join(subdir, 'Segmentation1.avi')
                left_generator = yield_video(left_path)
            elif 'Left_Instrument_Segmentation1.avi' in files:
                left_path = os.path.join(subdir, 'Left_Instrument_Segmentation1.avi')
                left_generator = yield_video(left_path)
            if 'Right_Instrument_Segmentation1.avi' in files:
                "right"
                right_path = os.path.join(subdir, 'Right_Instrument_Segmentation1.avi')
                right_generator = yield_video(right_path)
            else:
                right_generator = yield_zeros()

            for frame in vid_generator:

                np.savez(os.path.join(outpath, str(i)+".npz"), data=frame, \
                        left_mask=binarize_mask(next(left_generator)), \
                        right_mask=binarize_mask(next(right_generator)))
                i += 1
        #print(".")

    np.savez(os.path.join(outpath, "metadata.npz"), length=i)




def convert_tracking_to_numpy_individual(path, outpath):
    i = 0
    for subdir, dirs, files in os.walk(path):

        if 'Video1.avi' in files:
            video_path = os.path.join(subdir, 'Video1.avi')
            vid_generator = yield_video(video_path)

            left_generator = None
            right_generator = None
            if 'Pose.txt' in files:
                left_path = os.path.join(subdir, 'Pose.txt')
                left_generator = yield_pose(left_path)
            elif 'Left_Instrument_Pose.txt' in files:
                left_path = os.path.join(subdir, 'Left_Instrument_Pose.txt')
                left_generator = yield_pose(left_path)
            if 'Right_Instrument_Pose.txt' in files:
                right_path = os.path.join(subdir, 'Right_Instrument_Pose.txt')
                right_generator = yield_pose(right_path)
            else:
                right_generator = yield_empty_pose()

            for frame in vid_generator:
                np.savez(os.path.join(outpath, str(i)+".npz"), data=frame, \
                        left_mask=next(left_generator), \
                        right_mask=next(right_generator))
                i += 1
        #print(".")

    np.savez(os.path.join(outpath, "metadata.npz"), length=i)



def convert_all_data(og_data, new_data, subfolders):
    shutil.rmtree(new_data, ignore_errors=True)
    for subfolder, seg in subfolders.items():
        oldpath = os.path.join(og_data, subfolder)
        newpath = os.path.join(new_data, subfolder)
        os.makedirs(newpath)
        if seg:
            convert_segmenation_to_numpy_individual(oldpath, newpath)
        else:
            convert_tracking_to_numpy_individual(oldpath, newpath)


    # convert_segmenation_to_numpy_individual("Segmentation_train", r"C:\Users\amber\Documents\github\CS_7643_project\data\segmentation_train")
    # convert_segmenation_to_numpy_individual("Segmentation_test", r"C:\Users\amber\Documents\github\CS_7643_project\data\segmentation_test")
    # convert_tracking_to_numpy_individual("Tracking_train", r"C:\Users\amber\Documents\github\CS_7643_project\data\tracking_train")
    # convert_tracking_to_numpy_individual("Tracking_test", r"C:\Users\amber\Documents\github\CS_7643_project\data\tracking_test")
