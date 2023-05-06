import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from FaceApplicationPipe import model_layer as fap_cl
from FaceApplicationPipe import application_layer as fap_al
import multiprocessing
from multiprocessing import Pool
import time
import datetime
import os
import glob


########################################################################################################

# setting config first

current_dir = os.path.dirname(os.path.abspath(__file__))
index = 1
adv_path = current_dir + "/demo_image/" + str(index)+".jpg"
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

adv_image = cv2.imread(adv_path)
adv_image = cv2.resize(adv_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
sleep_time = 0.1
#########################################################################################################

# initial Mediapipe face detection

print("initial mediapipe face detection")

# mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

##########################################################################################################

# initial FaceApplicationPipe

print("initial FaceApplicationPipe classification layer")

emotion_model = fap_cl.ImageClassification("emotion")
gender_model = fap_cl.ImageClassification("gender") # can input your own tflite model by input a list: [model_path, model_lable]
post_algorithm = fap_cl.CoordinateAlgorithm() # can use inherit to default functions

print("initial FaceApplicationPipe application layer")
tools = fap_al.ApplicationTool(["time", "advertisement", "man attract times", "woman attract times"]) # can use inherit to default functions

##########################################################################################################

# main loop
adv_status = "close"
face_status = "same"
previous_face = 0

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
    t = 0
    while cap.isOpened():
    
        #time.sleep(sleep_time)
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue 
           
#####################################################  MediaPipe detection layer  ######################################################################################        
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)  # mediapipe face detection
        
        
        if results.detections: # detect face !!
            os.system('vcgencmd display_power 1')
            adv_status = "display"
            if t == 25:
                print(adv_path)
                index += 1
                if index > 3:
                    index = 1  

                t = 0
                
            adv_path = current_dir + "/demo_image/" + str(index)+".jpg"
            WINDOW_WIDTH = 1000
            WINDOW_HEIGHT = 1000
            adv_image=cv2.namedWindow('demo',cv2.WINDOW_FULLSCREEN)
            adv_image=cv2.moveWindow('demo',540,0)
            adv_image = cv2.imread(adv_path)
            adv_image = cv2.resize(adv_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            # show advertisement
            cv2.imshow("demo", adv_image) 
            t+=1
        
            if cv2.waitKey(2) & 0xFF == 27:
                print("end program")
                cv2.destroyAllWindows()
                break
                
            num_face = len(results.detections)
            #print(f"detected face number: {num_face}")
           
            for detection in results.detections:
                
                mp_drawing.draw_detection(image, detection)
                
                # get bbox, keypoint
                bbox = detection.location_data.relative_bounding_box
                keypoint = detection.location_data.relative_keypoints        
        
#####################################################  FaceApplicationPipe classification layer  ###################################################################################### 
                
                prediction1 = post_algorithm.face_pose_detection(keypoint)
                
                
                if prediction1 != 'front':
                    print("not looking at the screen, skip this face")
                    continue
                
                face = fap_cl.crop_face(image, bbox)   
                if face.shape[0] == 0 or face.shape[1] == 0:
                    
                    print("crop face error, skip this face")
                    continue
                                         
                prediction2 = emotion_model.inference(face)
                
                print(prediction2)
                if prediction2 == "Happy" or prediction2 == "Surprise":
                    pass
                else:
                    continue
                    
                prediction3 = gender_model.inference(face) 
                print(prediction3)
                
                if prediction3 == "man":
                    tools.counter1 += 1
                   
                else:
                    tools.counter2 += 1
               
                    
#####################################################  FaceApplicationPipe application layer  ######################################################################################
        
        else:
            num_face = 0
            print("no face, close advertisement")
            print(adv_status)
            if cv2.getWindowProperty("demo", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("demo")
            os.system('vcgencmd display_power 0')
            
        if previous_face > num_face:
            face_status = "less"
        else:
            face_status = "same"
        
        print(face_status)
        
        
        if face_status == "less":
            
        
            now = datetime.datetime.now()
                               
            final_statistic = [now, adv_path, int(tools.counter1/previous_face), int(tools.counter2/previous_face)]
            tools.save_data(final_statistic)
            print(tools.counter1,tools.counter2)
                
            adv_status = "close"
            tools.counter1 = 0
            tools.counter2 = 0      
            
        previous_face = num_face
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('face', cv2.flip(image, 1))
        
        if cv2.waitKey(2) & 0xFF == 27:
            print("end program")
            cv2.destroyAllWindows()
            break
                  

cap.release()


