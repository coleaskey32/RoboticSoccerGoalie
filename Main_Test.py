import ctypes
import subprocess
import cv2
import matplotlib.pyplot as plt
import os

# function to show the image output 
def imShow(path):
  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  #plt.show()


# Load the shared library
darknet_lib = ctypes.CDLL('/home/colea/darknet/libdarknet.so')


################## Main Program ##################


# Capturing from webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L)

# Command to execute image detection 
command = ['/home/colea/darknet/darknet', 'detector', 'test', '/home/colea/darknet/data/obj.data', '/home/colea/darknet/cfg/yolov4-tiny-custom.cfg', '/home/colea/darknet/cfg/yolov4.weights', '-ext_output', '/home/colea/S1.jpg']

# Execute the command
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for the process to terminate and get the return code
return_code = process.wait()

# Print return code
print("Return Code:", return_code)

# Check if the command execution was successful
if return_code != 0:
    print("Error: Command execution failed with return code", return_code)
else:
    print("Command executed successfully")
    
    
imShow('predictions.jpg')