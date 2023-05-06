import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt

class ApplicationTool():
    
    def __init__(self, list=[]):
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.dirname(current_dir)
    
        self.csv_path = self.parent_dir + '/result.csv'
        
        with open(self.csv_path, 'w', newline='') as csvfile:
        
            writer = csv.writer(csvfile)
            writer.writerow(list)

        self.counter1 = 0
        self.counter2 = 0
        

    def save_data(self, data):
  
        
        with open(self.csv_path, 'a', newline='') as csvfile:
        
            writer = csv.writer(csvfile)
            writer.writerow(data)
            
    def save_bar(self, title='Bar Chart', xlabel='gender', x=['man', 'woman'], ylabel="attract times", y=[6, 3]):
        
        plt.bar(x, y)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(self.parent_dir +"/result.png")
        
    
 
        
        
   

    
    


