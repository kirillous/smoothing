import sys
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pykalman import KalmanFilter

#reading data from gpx file
def read_gpx(file):
    tree = ET.parse(file)
    root = tree.getroot()
    namespace = "{http://www.topografix.com/GPX/1/0}"

    points = []
    for trkpt in root.iter("{}trkpt".format(namespace)):
        lat = float(trkpt.attrib['lat'])
        lon = float(trkpt.attrib['lon'])
        points.append({"lat": lat,"lon": lon})
    
    df = pd.DataFrame(points)
    return df

#calculating distance from given coordinates
def distance(df):
    lat1 = np.radians(df["lat"])
    lon1 = np.radians(df["lon"])
    lat2 = lat1.shift(-1)
    lon2 = lon1.shift(-1)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    R = 6371000

    dist = R*c
    return dist.sum()

#kalman smoothing of raw df from gpx file
def Kalman_smoothing(df):
    observation_covariance = np.diag([15,15])**2              #GPS error ~15m
    transition_covariance = np.diag([1,1])**2                 #Movement speed is ~1m/s
    transition = np.array([[1,0],[0,1]])                      #No acceleration or external forces

    kf = KalmanFilter(initial_state_mean=df.iloc[0],
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition)
    
    smooth_points, _ = kf.smooth(df.values)
    smooth_df = pd.DataFrame(smooth_points, columns=["lat","lon"])
    return smooth_df

def main():
    filename = "walk1.gpx"
    df = read_gpx(filename)
    dist = distance(df)
    smooth_df = Kalman_smoothing(df)
    smooth_dist = distance(smooth_df)
    print("Unfiltered distance: {}".format(round(dist,2)))
    print("Filtered distance: {}".format(round(smooth_dist, 2)))

if __name__ == "__main__":
    main()