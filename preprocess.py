import pandas as pd
# from src.data_management.read_csv import *
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm
from multiprocessing import Process, Value, Manager
import numpy as np
import scipy as sp
import scipy.signal as sg
import time
from NN_Lib import get_lane_pertenence
import tensorflow as tf
from multiprocessing import Pool



# ********************
# Read data
# ********************

vehicles = []
ids_array = []
sequences = []
lanes_pose = [None]*60

for seq in tqdm(range(1, 60)):

    # if True:
    try:
        test_data = np.load('vehicles/vehicle_' + '{0:0=2d}'.format(seq) + '.npy')
        track_meta = pd.read_csv('data/' + '{0:0=2d}'.format(seq) + '_tracksMeta.csv')

        header = ['id', 'frame', 'X', 'Y', 'W', 'H', 'xVelocity', 'yVelocity', 'xAcceleration',
               'yAcceleration', 'frontSightDistance', 'backSightDistance', 'thw',
               'ttc', 'dhw', 'precedingXVelocity', 'precedingId', 'followingId',
               'leftFollowingId', 'leftAlongsideId', 'leftPrecedingId',
               'rightFollowingId', 'rightAlongsideId', 'rightPrecedingId', 'laneId',
               'f_d', 'b_d', 'bl_d', 'l_d', 'fl_d', 'br_d', 'r_d', 'fr_d']

        data = pd.DataFrame(data=test_data, columns=header)
        test_data = []
        

        ## FILTER

        max_distance = 100.0

        data.loc[data['xVelocity'] < 0, ['f_d', 'fl_d', 'fr_d', 'l_d', 'r_d', 'b_d', 'bl_d', 'br_d']] = -1.0 * data.loc[data['xVelocity'] < 0, ['f_d', 'fl_d', 'fr_d', 'l_d', 'r_d', 'b_d', 'bl_d', 'br_d']]
        data.loc[:, ['b_d', 'bl_d', 'br_d']] = -1.0 * data.loc[:, ['b_d', 'bl_d', 'br_d']]

        for idx in (data['id'].drop_duplicates()):

            if(data.loc[data['id'] == idx, 'Y'].shape[0] > 10):
            
                b, a = sg.butter(2, 0.1)
                data.loc[data['id'] == idx, 'Y'] = sg.filtfilt(b, a, data.loc[data['id'] == idx, 'Y'])

                b, a = sg.butter(2, 0.1)
                data.loc[data['id'] == idx, 'xVelocity'] = sg.filtfilt(b, a, data.loc[data['id'] == idx, 'xVelocity'])

                b, a = sg.butter(2, 0.1)
                data.loc[data['id'] == idx, 'yVelocity'] = sg.filtfilt(b, a, data.loc[data['id'] == idx, 'yVelocity'])

                b, a = sg.butter(2, 0.1)
                data.loc[data['id'] == idx, 'xAcceleration'] = sg.filtfilt(b, a, data.loc[data['id'] == idx, 'xAcceleration'])

                b, a = sg.butter(2, 0.1)
                data.loc[data['id'] == idx, 'yAcceleration'] = sg.filtfilt(b, a, data.loc[data['id'] == idx, 'yAcceleration'])

            if(np.max(track_meta[track_meta['drivingDirection'] == 1]['id'] == idx)):
                data.loc[data['id'] == idx, 'yVelocity'] = -1.0 * data.loc[data['id'] == idx, 'yVelocity']
                data.loc[data['id'] == idx, 'Y'] = 100.0 - data.loc[data['id'] == idx, 'Y']
                data.loc[data['id'] == idx, 'X'] = 1000.0 - data.loc[data['id'] == idx, 'X']
                data.loc[data['id'] == idx, 'xAcceleration'] = -1.0 * data.loc[data['id'] == idx, 'xAcceleration']
                data.loc[data['id'] == idx, 'xVelocity'] = -1.0 * data.loc[data['id'] == idx, 'xVelocity']


        for h in ['f_d', 'fl_d', 'fr_d', 'b_d', 'bl_d', 'br_d']:
            data.loc[data[h] == 0, h] = max_distance
            data.loc[:, h] = max_distance - data.loc[:, h]
            data.loc[data[h] < 0, h] = 0.0

        data['seq'] = seq
        ids_array.append(data['id'].drop_duplicates())
        
        # Get lanes poses


        poses = []
        for lane in range(2,9):
            pose = np.mean(data[data['laneId'] == lane]['Y'])
            if pose > 0 :
                poses.append(pose)
        lanes_pose[int(seq)] = poses
        
        extra = data['Y'].apply(get_lane_pertenence, args=([lanes_pose[seq]]))
        data['lane1'] = pd.DataFrame(np.array(extra.tolist())[:,0])
        data['lane2'] = pd.DataFrame(np.array(extra.tolist())[:,1])
        data['lane3'] = pd.DataFrame(np.array(extra.tolist())[:,2])
        
        vehicles.append(data)

                            
    # else:
    except:
        print('Sequence ', seq, ' not found')
        
vehicles = pd.concat(vehicles, axis=0)
vehicles = vehicles.set_index('seq', drop=False)
vehicles = vehicles.fillna(0)

ids_array= np.concatenate(ids_array)






# ********************
# Normalize data
# ********************

norm_headers = ['xVelocity', 'yVelocity', 'xAcceleration', 'ttc', 'l_d', 'r_d', 'f_d', 'b_d', 'bl_d', 'fl_d', 'br_d', 'fr_d']

vehicles.loc[:, 'xVelocity'] = np.cbrt(vehicles.loc[:, 'xVelocity'])
vehicles.loc[:, 'yVelocity'] = np.cbrt(vehicles.loc[:, 'yVelocity'])
vehicles.loc[:, 'xAcceleration'] = np.cbrt(vehicles.loc[:, 'xAcceleration'])
vehicles.loc[:, 'l_d'] = np.cbrt(vehicles.loc[:, 'l_d'])
vehicles.loc[:, 'r_d'] = np.cbrt(vehicles.loc[:, 'r_d'])
vehicles.loc[:, 'f_d'] = np.cbrt(vehicles.loc[:, 'f_d'])
vehicles.loc[:, 'b_d'] = np.cbrt(vehicles.loc[:, 'b_d'])
vehicles.loc[:, 'fl_d'] = np.cbrt(vehicles.loc[:, 'fl_d'])
vehicles.loc[:, 'fr_d'] = np.cbrt(vehicles.loc[:, 'fr_d'])
vehicles.loc[:, 'br_d'] = np.cbrt(vehicles.loc[:, 'br_d'])
vehicles.loc[:, 'bl_d'] = np.cbrt(vehicles.loc[:, 'bl_d'])

norm_data = pd.DataFrame(np.zeros([len(norm_headers),5]))
norm_data.columns = ['min', 'max', 'mean', 'std', 'mode']
norm_data.index = norm_headers

for header in norm_headers[:-6]:
    norm_data.loc[header, 'min'] = np.min(vehicles[header])
    norm_data.loc[header, 'max'] = np.max(vehicles[header])
    norm_data.loc[header, 'mean'] = np.mean(vehicles[header])
    norm_data.loc[header, 'std'] = np.std(vehicles[header])
    norm_data.loc[header, 'mode'] = sp.stats.mode(vehicles[header])[0]

for header in norm_headers[-6:]:
    norm_data.loc[header, 'min'] = np.min(vehicles[header])
    norm_data.loc[header, 'max'] = np.max(vehicles[header])
    norm_data.loc[header, 'mean'] = np.mean(vehicles[vehicles[header]>0][header])
    norm_data.loc[header, 'std'] = np.std(vehicles[vehicles[header]>0][header])
    norm_data.loc[header, 'mode'] = sp.stats.mode(vehicles[vehicles[header]>0][header])[0]

vehicles.loc[:, 'xVelocity'] = (vehicles.loc[:, 'xVelocity'] - norm_data.loc['xVelocity', 'mean'])/norm_data.loc['xVelocity', 'std']
vehicles.loc[:, 'yVelocity'] = (vehicles.loc[:, 'yVelocity'])/norm_data.loc['yVelocity', 'std']
vehicles.loc[:, 'xAcceleration'] = (vehicles.loc[:, 'xAcceleration'])/norm_data.loc['xAcceleration', 'std']
vehicles.loc[:, 'l_d'] = (vehicles.loc[:, 'l_d'])/norm_data.loc['l_d', 'std']
vehicles.loc[:, 'r_d'] = (vehicles.loc[:, 'r_d'])/norm_data.loc['r_d', 'std']
vehicles.loc[:, 'f_d'] = (vehicles.loc[:, 'f_d'])/norm_data.loc['f_d', 'mode']
vehicles.loc[:, 'b_d'] = (vehicles.loc[:, 'b_d'])/norm_data.loc['b_d', 'mode']
vehicles.loc[:, 'fl_d'] = (vehicles.loc[:, 'fl_d'])/norm_data.loc['fl_d', 'mode']
vehicles.loc[:, 'fr_d'] = (vehicles.loc[:, 'fr_d'])/norm_data.loc['fr_d', 'mode']
vehicles.loc[:, 'br_d'] = (vehicles.loc[:, 'br_d'])/norm_data.loc['br_d', 'mode']
vehicles.loc[:, 'bl_d'] = (vehicles.loc[:, 'bl_d'])/norm_data.loc['bl_d', 'mode']





# ********************
# Normalize Y
# ********************

def normalize_y(Y, lanes_pose):
    
    min_pertenence = 0.1

    #Changing id's order because Y was transformed
    lanes_pose = sorted(lanes_pose)
    
    n_lanes = len(lanes_pose)
    
    if n_lanes == 4:
        
        l_w1 = (lanes_pose[1] - lanes_pose[0])
        l_w2 = (lanes_pose[3] - lanes_pose[2])

        
        if Y < (lanes_pose[1] + lanes_pose[2])/2.0:
            pose = np.clip((Y - (lanes_pose[0] - l_w1/2.0))/(2 * l_w1), 0, 1)
        else:
            pose = np.clip(((lanes_pose[3] + l_w2/2.0) - Y)/(2 * l_w2), 0, 1)
        
    else:

        l_w1 = (lanes_pose[1] - lanes_pose[0])
        l_w2 = (lanes_pose[2] - lanes_pose[1])
        l_w3 = (lanes_pose[4] - lanes_pose[3])
        l_w4 = (lanes_pose[5] - lanes_pose[4])

        
        if Y < (lanes_pose[3] + lanes_pose[2])/2.0:
            if Y < lanes_pose[1]:
                pose = np.clip((Y - (lanes_pose[0] - l_w1/2.0))/(2 * l_w1), 0, 0.75)
            else:
                pose = np.clip((((Y - lanes_pose[1])/(2 * l_w2)) + 0.75), 0.75, 1.5)             
        else:
            if Y > lanes_pose[4]:
                pose = np.clip(((lanes_pose[5] + l_w4/2.0) - Y)/(2 * l_w4), 0, 0.75)
            else:
                # return 0
                pose = np.clip((((lanes_pose[4]) - Y)/(2 * l_w3) + 0.75), 0.75, 1.5)
            

    return pose

for seq in range(1,59):
    try:
        vehicles.loc[seq, 'normalized_Y'] = vehicles.loc[seq, 'Y'].apply(normalize_y, args=([lanes_pose[seq]]))
        print(seq)
    except:
        print('Seq ', seq, ' not found')




# ********************
# Prepare train data
# ********************

x_headers = ['X', 'Y', 'xAcceleration', 'xVelocity', 'yVelocity', 'f_d', 'b_d', 'bl_d', 'l_d', 'fl_d', 'br_d', 'r_d', 'fr_d', 'lane1', 'lane2', 'lane3', 'normalized_Y']
y_headers = ['X', 'Y', 'xVelocity', 'yVelocity', 'xAcceleration', 'maneuver', 'lane1', 'lane2', 'lane3', 'normalized_Y']


def detect_lane_change(array, drive_dir, dist=80):
    if drive_dir == 1:
        change_idx_l = np.argmax(np.gradient(array))
        change_idx_r = np.argmin(np.gradient(array))
    else:
        change_idx_r = np.argmax(np.gradient(array))
        change_idx_l = np.argmin(np.gradient(array))
    output = np.zeros(array.shape[0])
    if change_idx_l != 0:
        change_idx_l_0 = max(0, change_idx_l - dist)
        change_idx_l_1 = min(array.shape[0], change_idx_l + dist)   
        output[change_idx_l_0:change_idx_l_1] = 1
    if change_idx_r != 0:    
        change_idx_r_0 = max(0, change_idx_r - dist)
        change_idx_r_1 = min(array.shape[0], change_idx_r + dist)
        output[change_idx_r_0:change_idx_r_1] = -1
    return output

def generate_data(seq, return_data=False):
    
    
    vehicles_seq = vehicles.loc[seq].set_index(['seq', 'id'])
    vehicles_seq['maneuver'] = np.zeros(vehicles_seq.shape[0])
    
    X = np.empty((0,X_time_steps, len(x_headers)))
    y = np.empty((0,y_time_steps, len(y_headers)))
    
    track_meta = pd.read_csv('data/' + '{0:0=2d}'.format(int(seq)) + '_tracksMeta.csv')
    for idx in (vehicles.loc[seq, 'id'].drop_duplicates()):
    # for idx in tqdm(track_meta[track_meta['numLaneChanges'] == 0]['id']):
    
        if int(track_meta[track_meta.id == idx ]['drivingDirection']) == 1:
    
            vehicles_seq.loc[(seq, idx), 'maneuver'] = detect_lane_change(vehicles_seq.loc[(seq, idx), 'laneId'], int(track_meta[track_meta.id == idx ]['drivingDirection']))
            slices_x = np.empty((0,X_time_steps))
            slices_y = np.empty((0,y_time_steps))

            for i in np.arange(0, (vehicles_seq.loc[seq, idx].shape[0]-(X_time_steps+y_time_steps)*step_sep), sample_sep):
                slices_x = np.append(slices_x, np.array(range(i, i+X_time_steps*step_sep, step_sep)))

            for i in np.arange((X_time_steps)*step_sep, (vehicles_seq.loc[seq, idx].shape[0]-(y_time_steps)*step_sep), sample_sep):
                slices_y = np.append(slices_y, np.array(range(i, i+y_time_steps*step_sep, step_sep)))


            if len(slices_x)>0:
                X = np.append(X, np.reshape(np.array(vehicles_seq.loc[seq, idx].iloc[slices_x][x_headers]), (int(slices_x.shape[0]/X_time_steps), X_time_steps, len(x_headers))), axis=0)
                y = np.append(y, np.reshape(np.array(vehicles_seq.loc[seq, idx].iloc[slices_y][y_headers]), (int(slices_y.shape[0]/y_time_steps), y_time_steps, len(y_headers))), axis=0)

    np.save('train_data/' + str(y_time_steps) + '/X_' + '{0:0=2d}'.format(int(seq)), np.array(X).astype(np.float32))
    np.save('train_data/' + str(y_time_steps) + '/y_' + '{0:0=2d}'.format(int(seq)), np.array(y).astype(np.float32))
    
    
    if return_data:
        return X, y
    else:
        return 0
    
X_time_steps = 32
y_time_steps = 64
y_delay = 0
max_vehicle_T = 300
vehicles_seq_sep = 15
step_sep = 2
sample_sep = 15
T = 1.0/25.0

    
pool = Pool(processes=7)              
inputs = vehicles.index.drop_duplicates()#[:-2]
result = pool.map(generate_data, inputs)
