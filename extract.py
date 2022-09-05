import pandas as pd
from multiprocessing import Process, Value, Manager
import numpy as np
import os

# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"

# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"
DRIVING_DIRECTION = "drivingDirection"
TRAVELED_DISTANCE = "traveledDistance"
MIN_X_VELOCITY = "minXVelocity"
MAX_X_VELOCITY = "maxXVelocity"
MEAN_X_VELOCITY = "meanXVelocity"
MIN_DHW = "minDHW"
MIN_THW = "minTHW"
MIN_TTC = "minTTC"
NUMBER_LANE_CHANGES = "numLaneChanges"

# VIDEO META
ID = "id"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
MONTH = "month"
WEEKDAY = "weekDay"
START_TIME = "startTime"
DURATION = "duration"
TOTAL_DRIVEN_DISTANCE = "totalDrivenDistance"
TOTAL_DRIVEN_TIME = "totalDrivenTime"
N_VEHICLES = "numVehicles"
N_CARS = "numCars"
N_TRUCKS = "numTrucks"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"

# Modified function based on HighD API
def read_track_csv(arguments):
    """
    This method reads the tracks file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
    :return: a list containing all tracks as dictionaries.
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_path"])

    # Use groupby to aggregate track info. Less error prone than iterating over the data.
    grouped = df.groupby([TRACK_ID], sort=False)
    # Efficiently pre-allocate an empty list of sufficient size
    tracks = [None] * grouped.ngroups
    current_track = 0
    for group_id, rows in grouped:
        #bounding_boxes = np.transpose(np.array([rows[X].values,
        #                                        rows[Y].values,
        #                                        rows[WIDTH].values,
        #                                        rows[HEIGHT].values]))
        tracks[current_track] = {TRACK_ID: np.int64(group_id),  # for compatibility, int would be more space efficient
                                 FRAME: rows[FRAME].values,
                                 #BBOX: bounding_boxes,
                                 'X': rows[X].values,
                                 'Y': rows[Y].values,
                                 'W': rows[WIDTH].values,
                                 'H': rows[HEIGHT].values,
                                 X_VELOCITY: rows[X_VELOCITY].values,
                                 Y_VELOCITY: rows[Y_VELOCITY].values,
                                 X_ACCELERATION: rows[X_ACCELERATION].values,
                                 Y_ACCELERATION: rows[Y_ACCELERATION].values,
                                 FRONT_SIGHT_DISTANCE: rows[FRONT_SIGHT_DISTANCE].values,
                                 BACK_SIGHT_DISTANCE: rows[BACK_SIGHT_DISTANCE].values,
                                 THW: rows[THW].values,
                                 TTC: rows[TTC].values,
                                 DHW: rows[DHW].values,
                                 PRECEDING_X_VELOCITY: rows[PRECEDING_X_VELOCITY].values,
                                 PRECEDING_ID: rows[PRECEDING_ID].values,
                                 FOLLOWING_ID: rows[FOLLOWING_ID].values,
                                 LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                                 LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                                 LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                                 RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                                 RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                                 RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                                 LANE_ID: rows[LANE_ID].values
                                 }
        current_track = current_track + 1
    return tracks


def worker(dataframe_id):
   
    args = {'input_path': 'data/' + "{0:0=2d}".format(dataframe_id) + '_tracks.csv'}

    read_vehicles = pd.DataFrame(read_track_csv(args))
    v = read_vehicles.set_index(['id'])
    unnested_lst = []
    for col in v.columns:
        unnested_lst.append(v[col].apply(pd.Series).stack())
    vehicles = pd.concat(unnested_lst, axis=1, keys=read_vehicles.columns[1:])
    
    new_data = []

    for i in (range(1, vehicles.index.values[-1][0]+1)):

        distances = []

        for idx in range(len(vehicles.loc[i, 'frame'])):

            fr = vehicles.loc[i, 'frame'][idx]
            dist = []
            for v in vehicles.columns[15:23]:
                v_id = vehicles.loc[i, v][idx]
                if v_id > 0:
                    dist.append((vehicles.loc[v_id][vehicles.loc[v_id]['frame'] == fr]['X'] - vehicles.loc[i, 'X'][idx]).to_list()[0])
                else:
                    dist.append(float('NaN'))
            distances.append(dist)

        if i > 1:
            new_data = np.append(new_data, distances, axis=0)
        else:
            new_data = np.array(distances)       



    new_data = pd.DataFrame(new_data)
    new_data.columns = ['f_d', 'b_d', 'bl_d', 'l_d', 'fl_d', 'br_d', 'r_d', 'fr_d']
    print(new_data.shape)
    vehicles = vehicles.reset_index(level=0)
    vehicles = vehicles.reset_index()
    vehicles = vehicles.drop(labels='index', axis=1)
    print(vehicles.shape)
    test_data = pd.concat([vehicles, new_data], axis=1)
    print(test_data.shape)
    
    np.save('vehicles/vehicle_'+ "{0:0=2d}".format(dataframe_id), np.array(test_data))
    os._exit(os.EX_OK)

vehicles = range(1,60)

pid = []

for i in vehicles:
  
    pid.append(os.fork())
    if pid[-1] == 0:
        worker(i)
        os._exit(os.EX_OK)
    
for i in pid:
    os.waitpid(i, 0)
    print('Work ', i,' done')
