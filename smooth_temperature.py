import sys
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
import numpy as np

filename = "sysinfo.csv"
cpu_data = pd.read_csv(filename, parse_dates=['timestamp'])

#raw data
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label = "Raw data")

#LOESS noise reduction
loess_red = lowess(cpu_data['temperature'], cpu_data['timestamp'], frac=0.025)
plt.plot(cpu_data['timestamp'], loess_red[:, 1], 'r-', label = "LOESS Smoothing")

#kalman noise reduction
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([1, 1, 1, 1]) ** 2
transition_covariance = np.diag([0.01, 0.01, 0.01, 0.01]) ** 2
transition = np.array([[1, 0.25, 0.1, -0.001], [0.1, 0.25, 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
kf = KalmanFilter(initial_state_mean=initial_state, observation_covariance=observation_covariance, transition_covariance=transition_covariance, transition_matrices=transition)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label = "Kalman Smoothing")

#plot
plt.title("CPU temperature data")
plt.legend()
plt.tight_layout()
plt.xlabel("Date")
plt.ylabel("CPU temperature (°C)")
plt.show()


