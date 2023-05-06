import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import math
import os

class ImageClassification():
    
    def __init__(self, model_type):
        
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_type == "emotion":      
            
            print("initial emotion classification model") 
            self.labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]     
            #self.labels = ["ignore", "ignore", "ignore", "attract", "ignore", "attract", "ignore"] 
            filename = "/model/default/emotion.tflite"
            file_path = self.current_dir + filename
                
        elif model_type == "gender":
            
            print("initial gender classification model") 
            self.labels =["man", "woman"]    
            filename = "/model/default/gender.tflite"
            file_path = self.current_dir + filename
                                   
        else:
            self.labels = model_type[1]  
            file_path = model_type[0] 
           
        
        print("load model from:", file_path)
        self.interpreter = tf.lite.Interpreter(file_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1]
        self.input_color = self.input_details[0]['shape'][-1]
                  
        
    def inference(self, face_image):
        
        if self.input_color == 3:
            
            normalize_image = np.array(face_image) / 255.0                           
            input_data = cv2.resize(normalize_image, (self.input_shape,self.input_shape)).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
        elif self.input_color == 1:
            
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            normalize_image = np.array(gray_image) / 255.0
            input_data = cv2.resize(normalize_image, (self.input_shape,self.input_shape)).astype(np.float32)            
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.expand_dims(input_data, axis=-1)
            
        else:
            pass
        
 
        # Set input tensor data
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
     
        # Run inference
        self.interpreter.invoke()
     
        # Get output tensor data
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
  
        # Process output data
        predicted_class = np.argmax(output_data)
        prediction = self.labels[predicted_class]
        
        return prediction
        
def crop_face(image, bbox):
    
    x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
    
    return image[int(y * image.shape[0]):int((y + h) * image.shape[0]), int(x * image.shape[1]):int((x + w) * image.shape[1])]
        
class CoordinateAlgorithm():
    
    def __init__(self):
        
        print("initial coordinate location")
         
        self.left_eye_location = 0
        self.right_eye_location = 1
        self.nose_location = 2
        self.mouth_location = 3
        self.left_ear_location = 4
        self.right_ear_location = 5
        
        
    def face_pose_detection(self, input):
             
        x1,y1 = input[self.left_ear_location].x, input[self.left_ear_location].y
        x2,y2 = input[self.nose_location].x, input[self.nose_location].y
        x3,y3 = input[self.right_ear_location].x, input[self.right_ear_location].y
        
        left_dist = round(self.cal_two_points_distance((y1,x1),(y2,x2)), 2)
        right_dist = round(self.cal_two_points_distance((y2,x2),(y3,x3)), 2)        
 
        ratio = left_dist/right_dist
        
        if ratio>2:
            status = "left"
        elif ratio<0.5:
            status = "right"
        else:
            status = "front"
            
        if x2>=x3:
            status = "left"
        
        if x2<=x1:
            status = "right"
                   
        return status
     
   
    def cal_two_points_distance(self, p1, p2):
        return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))
        


    
    


