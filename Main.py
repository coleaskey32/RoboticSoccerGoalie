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
  
  print(resized_image)
  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  #plt.show()


# Load the shared library
darknet_lib = ctypes.CDLL('/home/colea/darknet/libdarknet.so')

def videoFeed_process(conn):
    cap = cv2.VideoCapture(0)  # Replace 'your_video_file.mp4' with the path to your video file or 0 for webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
            
        conn.send(frame)
        
    cap.release()
    conn.close()


def yolo_process(conn, result_conn):

    # Command to execute YOLO detection
    yolo_command = ['/home/colea/darknet/darknet', 'detector', 'test', '/home/colea/darknet/data/obj.data', '/home/colea/darknet/cfg/yolov4-tiny-custom.cfg', '/home/colea/darknet/cfg/yolov4.weights', '-ext_output', '/home/colea/frame.jpg']

    while True:
        
        # Take Frame from the Pipe with videoFeed 
        frame = conn.recv()
        
        if frame is None:
            break
            
        # Convert frame to JPEG format (YOLO expects image files)
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # Write the encoded image to the correct image path
        with open('/home/colea/frame.jpg', 'wb') as f:
            f.write(img_encoded)
            
        # Execute YOLO detection
        process = subprocess.Popen(yolo_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for the process to terminate and get the return code
        return_code = process.wait()

        # Print return code
        print("Return Code:", return_code)

        # Check if the command execution was successful
        if return_code != 0:
            print("Error: Command execution failed with return code", return_code)
        else:
            print("Command executed successfully")
            
        # Send the detection results through the pipe to the OpticalFlow process
        result_conn.send(stdout)
        
    conn.close()
    result_conn.close()


def opticalFlow_process(conn):

    while True:
    
        result = conn.recv()
        
        if result is None:
            break
            
        # Process the YOLO detection results as needed
        print(result)  # Example: Print the detection results
        
    conn.close()


################## Main Program ##################

if __name__ == '__main__':
    
     sender_parent_conn, sender_child_conn = Pipe()
    result_parent_conn, result_child_conn = Pipe()

    # Start sender process
    sender_process = Process(target=sender, args=(sender_child_conn,))
    sender_process.start()

    # Start YOLO receiver process
    yolo_receiver_process = Process(target=yolo_receiver, args=(sender_parent_conn, result_parent_conn))
    yolo_receiver_process.start()

    # Start result receiver process
    result_receiver_process = Process(target=result_receiver, args=(result_parent_conn,))
    result_receiver_process.start()

    # Wait for sender process to finish
    sender_process.join()

    # Signal YOLO receiver process to finish
    sender_parent_conn.send(None)
    yolo_receiver_process.join()

    # Signal result receiver process to finish
    result_parent_conn.send(None)
    result_receiver_process.join()
