# Real Time Smart Advertising Interest Detection System


## Code introduction step by step

1. environment config:
- tf2 or above
- mediapipe
- opencv

2. demo_image file:
This file represents the advertisement demo part. You can change and put your own advertisement image inside it. 

3. FaceApplicationPipe file:
This file includes the implementation of our own framework: FaceApplicationPipe. You can directly import this file to use the default function inside it.
- model file: include the default model weight path we have used in this project
- application_layer.py: this is the FaceApplicationPipe application layer code. It offers some function such as counter, save data, plot data, etc.
- model_layer.py: this code is the implementation of FaceApplicationPipe model layer. Inside this code, it has two classes. One is for deep learning model image classification development. The other is for bbox post-processing algorithm construction. Again, you can design and put your own model and algorithms into this file.
- tf_to_tflite.ipynb: this code provide the method to convert model into tensorflow lite.
- train_emotion.ipynb: This code provide training an emotion classification model under mobilenetv2.

4. demo.py:
This is the main code of our project: advertisement interest detection. You can directly run it by typing "python demo.py" in command. Remember to change the camera number in 59 line.

5. result.csv:
This is the final output/result after running demo.py. You can get the statistic inside this csv file.