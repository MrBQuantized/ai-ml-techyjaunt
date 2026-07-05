"""
Lab 2C — Students Dataset EDA
Using the Kaggle Students Performance dataset complete the following tasks:

Task 1: Load the CSV file into a Pandas DataFrame and print its shape and column names
Task 2: Run df.describe() and identify the subject with the highest average score
Task 3: Use df.isnull().sum() to check for any missing values in the dataset
Task 4: Create a histogram of math scores using Matplotlib — add a title and axis labels
Task 5: Use Seaborn to plot a scatter chart of reading vs. writing scores — colour by gender
Task 6: Find the average math score grouped by parental education 
"""


import os
print("current directory: ", os.getcwd())
os.chdir(r"C:\Users\Dell\Documents\MY_WORKSPACE\02_SCHOLARSHIPS\03_AI_ML_TECHYJAUNT\02_module-python-for-ml\lab")
print("current directory: ", os.getcwd())
# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# —— Data Loading and Inspections ————————————————————————————————————————————————
# == 1. Load dataset ==
df = pd.read_csv('student_data.csv')

# Display the first 5 rows
print(df.head)
 
 
# —— Data Cleaning ———————————————————————————————————————————————————————————————
# —— EDA —————————————————————————————————————————————————————————————————————————


