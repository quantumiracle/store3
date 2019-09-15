import numpy as np
import cv2
import imutils
from localize import ImageProcessing, ContourCenterCheck, CenterRegister, PointCheck, Norm

from deform_visualize import plot_list_new
from td3_multiprocess_all import TD3_Trainer, ReplayBuffer


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

def InitPredict(model_path = './model/class_obj'): 
    obs_dim = 182  # total 280: 0 object index, 1-3 rotation value, 4-6 average contact point position, 7-279 pins positions
    state_dim = 5  # 3 rotation, 2 position
    lr=2e-2
    num_obj=3  # number of objects
    classifier = Classifier(obs_dim, state_dim, lr, num_obj)
    classifier.load(model_path) 
    return classifier

def load_policy():
    replay_buffer = ReplayBuffer(1e6)
    td3_trainer=TD3_Trainer(replay_buffer)
    model_path='./model/td3_all'
    td3_trainer.load_model(model_path)
    return td3_trainer

def get_action(policy, state):
    script_action = policy.policy_net.get_action(state)
    real_sensor_size=0.39
    sim_sensor_size=3.9
    sim_action_unit=0.02
    sim_action=script_action*sim_action_unit
    action=sim_action*real_sensor_size/sim_sensor_size
    return action

def initialize_camera():

    cap = cv2.VideoCapture(0)
    for i in range(10): # skip initial frames
	ret, frame = cap.read()
    return cap

def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()

def get_state(cap, idx):
    NUM_PINS=127
    SELECT_NUM_PINS=91 # drop the most outside circle of pins  91 or 61 
    last_contour_centers=None
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img=preprocessing(frame, ret)

    cimg, edge_detected_image, contour_centers = ImageProcessing(img)

    cimg, contour_centers = ContourCenterCheck(contour_centers, cimg, NUM_PINS=NUM_PINS)

    contour_centers=CenterRegister(contour_centers, cimg)

    contour_centers = PointCheck(contour_centers, last_contour_centers, max_dis=15)  # larger than 15 is mis-registered

    last_contour_centers = contour_centers
    
    norm_pos=Norm(contour_centers)

    # plot_list_new(norm_pos, idx)

    norm_pos_=norm_pos.reshape(-1)  # (x,y,x,y...)

    return norm_pos_, norm_pos



