import numpy as np
import cv2
import imutils
from localize import ImageProcessing, ContourCenterCheck, CenterRegister, PointCheck, Norm

from deform_visualize import plot_list_new, plot_list_new_real_arrow
from pole_predictor import Predictor



def preprocessing(frame, ret):
    ''' rotate and crop and resize '''
    height=frame.shape[0]
    width=frame.shape[1]
    # print(ret, height, width)
    if ret:
        # rotate the image to standard direction
        angle=30   # -13
        rotated = imutils.rotate(frame, angle)

        crop_img = rotated[:, int((width-height)/2):int((width+height)/2)]  # crop to be square
        resized = cv2.resize(crop_img, (256, 256), interpolation = cv2.INTER_AREA)

    # Display the resulting frame
    # cv2.imshow('frame', resized)

    return resized[:,:, 0]  # only 1 channel

def load_policy(): 
    model_path = './model/pole/class_obj'
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 2  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    predictor = Predictor(obs_dim, state_dim, lr)
    predictor.load(model_path) 
    return predictor

def load_policy_threshold(): 
    model_path = './model/pole/class_obj'
    obs_dim = 91  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 2  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    predictor = Predictor(obs_dim, state_dim, lr)
    predictor.load(model_path) 
    return predictor

def load_policy_weightedaverage(): 
    model_path = './model/pole/class_obj'
    obs_dim = 3  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 2  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    predictor = Predictor(obs_dim, state_dim, lr)
    predictor.load(model_path) 
    return predictor
# def load_policy():contour_centers
#     replay_buffer = ReplayBuffer(1e6)
#     td3_trainer=TD3_Trainer(replay_buffer)
#     model_path='./model/td3_all'
#     td3_trainer.load_model(model_path)
#     return td3_trainer

def get_action(policy, state):
    colli_pos = policy.predict_one_value(state)[0][6:]
    norm2sim=0.5674

    return colli_pos/norm2sim

def initialize_camera():

    cap = cv2.VideoCapture(0)
    for i in range(10): # skip initial frames
	    ret, frame = cap.read()
    return cap

def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()

def get_state(cap, idx, original_contour_centers=None):
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img=preprocessing(frame, ret)

    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)

    contour_centers = PointCheck(contour_centers, original_contour_centers, max_dis=15)  # larger than 15 is mis-registered
    
    norm_pos=Norm(contour_centers)
    # norm2sim=0.5674
    # norm_pos_sim=norm_pos*norm2sim  # transform norm to sim
    # norm_pos_=norm_pos_sim.reshape(-1) 

    norm_pos_ = norm_pos.reshape(-1)
    #plot_list_new(norm_pos, idx)

    

    return norm_pos_,  norm_pos




def get_state_threshold(cap, idx, xy0, original_contour_centers=None):
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img=preprocessing(frame, ret)

    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)

    contour_centers = PointCheck(contour_centers, original_contour_centers, max_dis=15)  # larger than 15 is mis-registered
    
    norm_pos=Norm(contour_centers)
    norm2sim=0.5674
    norm_pos_sim=norm_pos*norm2sim  # transform norm to sim
    [x, y]=np.transpose(norm_pos_sim)
    [x0, y0]=xy0

    dis_threshold=0.05

    dis=np.abs(x-x0)+np.abs(y-y0)
    dis_threshold = max(1.2*np.average(dis), dis_threshold)
    dis_idx=np.argwhere(dis>dis_threshold).reshape(-1)
    dis[dis<=dis_threshold]=0
    dis[dis>dis_threshold]=1

    processed_state=dis
    

    return processed_state


def get_state_weightedaverage(cap, idx, xy0, original_contour_centers=None):
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img=preprocessing(frame, ret)

    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)

    contour_centers = PointCheck(contour_centers, original_contour_centers, max_dis=15)  # larger than 15 is mis-registered
    
    norm_pos=Norm(contour_centers)
    norm2sim=0.5674
    norm_pos_sim=norm_pos*norm2sim  # transform norm to sim
    [x, y]=np.transpose(norm_pos_sim)
    [x0, y0]=xy0

    scaling=10.
    dis=np.abs(x-x0)+np.abs(y-y0)
    coord_list=np.transpose([x,y])
    weighted_average = np.dot(dis[np.newaxis,:]**2, coord_list)/(np.sum(dis**2)+1e-5) # square to show more difference

    processed_state=[np.average(dis)*scaling] # amount of current displacement, normalized around 1
    processed_state = np.concatenate((processed_state, weighted_average[0]*scaling))  # reduce 1 dim in weighted_average, normalized around 1
    

    return processed_state

def get_ini_pos(cap):
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img=preprocessing(frame, ret)

    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)

    # contour_centers = PointCheck(contour_centers, original_contour_centers=None, max_dis=15)  # larger than 15 is mis-registered
    
    norm_pos=Norm(contour_centers)
    plot_list_new_real_arrow(norm_pos, idx=0)
    norm2sim=0.5674
    norm_pos_sim=norm_pos*norm2sim  # transform norm to sim
    xy=np.transpose(norm_pos_sim)  # [[x], [y]]

    return xy

