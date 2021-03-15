# coding: utf-8

# # Experimental Results
# This script parses experimental logs to obtain performance metrics.
# Please note that log_analyzer.py is used from the tools directory.
# Documentation of the usage for the LogAnalyzer class is provided on log_analyzer.py

# Import LogAnalyzer objects
from tools.log_analyzer import *

# and other relevant stuff...
import matplotlib.pyplot as plt

# #### !IMPORTANT: Specify directory and log filenames here 
# Note that the provided names (below) are default names. They do not have to be changes unless you decided to rename files from multiple experiments.
log_dir_path = "./logs/test/sim/"
vehicle_log_file = "vehicle.log"
customer_log_file = "customer.log"
score_log_file = "score.log"
summary_log_file = "summary.log"


# Invoke LogAnalyzer Object
l = LogAnalyzer()


# #### Exploring dataframes of each of the logs
# Loading all the different logs as pandas dataframes
summary_df = l.load_summary_log(log_dir_path)
vehicle_df = l.load_vehicle_log(log_dir_path)
customer_df = l.load_customer_log(log_dir_path)
score_df = l.load_score_log(log_dir_path)

print(summary_df.describe())

print(vehicle_df.describe())

print(customer_df["waiting_time"].describe())

print(score_df.describe())


# #### Exploring the get_customer_status
df = l.get_customer_status(customer_df)
print(df.head())

df = l.get_customer_waiting_time(customer_df)
print(df.head())


# #### Generating plots of summary logs
summary_plots = l.plot_summary([log_dir_path], ["Accept_Reject Rate", "Occupancy Rate"], plt)
summary_plots.savefig("Summary.png", bbox_inches = 'tight')
summary_plots.show()

# #### Generating plots of relevant experiment metrics
plt, df = l.plot_metrics([log_dir_path], ["Revenue", "Working Time", "Cruising Time", "Waiting_Time"], plt)
plt.savefig("Metrics.png", bbox_inches = 'tight')
plt.show()

# #### We may also look at the metrics as a pandas dataframe
print(df.head())
