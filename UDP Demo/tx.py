from src import MACGenerator, UDP_TX, SlidingBook
import src.TX as TX

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import socket
import struct
import time
import json
import cv2

import sys

sys.path.append('..')
import utils.utils as utils
import utils.Auth as Auth
import argparse
import json



def tx_param(IP, PORT, param:dict):
# send the parameters to the receiver and the time.time() to synchronize the sender and the receiver
    time.sleep(.1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.connect((IP, PORT))
                break
            except:
                pass
            
        data = json.dumps(param).encode()
        for i in range(len(data)//1000):
            s.send(data[i*1000:(i+1)*1000])
            time.sleep(.01)
        s.send(data[(i+1)*1000:])
        time.sleep(.1)
        s.send(b'END')
        s.close()
    
    time.sleep(.1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.connect((IP, PORT))
                break
            except:
                pass
        data = time.time()
        data = struct.pack('d', data)
        s.send(data)
        s.close()

    time.sleep(.1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.connect((IP, PORT))
                break
            except:
                pass
        
        res = s.recv(1024)
        if res == b'ABORT':
            print("The receiver has aborted the connection")
        elif res == b'RETRY':
            print("The receiver has requested to resend the parameters")
        s.close()
    
    if res == b'ABORT':
        return False
    elif res == b'RETRY':
        time.sleep(2)
        print("retrying to send the parameters!60")
        return tx_param(IP, PORT, param)
    elif res == b'OK':
        return True
    else:
        print("The receiver has sent an unknown message")
        return False



def run_experiment(IP, PORT, param:dict):
    # Capture video from the default camera
    cap = cv2.VideoCapture(0) 
    # Define the compression parameters for Progressive JPEG
    compression_params = [cv2.IMWRITE_JPEG_PROGRESSIVE, 1, cv2.IMWRITE_JPEG_QUALITY, param['QUALITY']]
    ## counting the number of frames per second

    start_time = time.time()
    runTime_cnt = time.time()
    frame_counter = 0
    average_tx_size = 0
    while True:
        ret, frame = cap.read()
        # Encode the image to JPEG format in memory
        success, encoded_frame = cv2.imencode('.jpg', frame, compression_params)
        data = encoded_frame.tobytes()
        average_tx_size+=len(data)//1000

        #############################  TX  ########################################
        buffer = SlidingBook(num_pages=40, page_size=len(param['X']))
        page_processor = MACGenerator(X = param['X'], Y = param['Y'], secret_key=param['KEY'].encode(), digestmod=param['DIGESTMOD'])
        udp_tx = UDP_TX(IP= IP, PORT= PORT, buffer=buffer, page_processor= page_processor, Payload_Size_Byte = param['PAYLOAD_SIZE_BYTE'])
        pages = udp_tx.segment_to_pages(data)
        udp_tx.transmit(pages, param['ATTACK_PROBABILITY'], delay=param['DELAY'])
        ###########################################################################

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

        # print the frame rate
        frame_counter += 1
        if time.time() - start_time >= 10:
            print("frame per second: ", frame_counter/10, "fps ,data rate: ", average_tx_size, "KB/s")
            frame_counter = 0
            start_time = time.time()
            average_tx_size = 0
        if time.time() - runTime_cnt >= param['DURATION']:
            break
    cv2.destroyAllWindows()
    cap.release()
    








if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('my_dict', type=str)
    args = parser.parse_args()

    parameters = json.loads(args.my_dict)

    IP = parameters['IP']
    PORT = parameters['PORT']
    param = parameters['param']

    res = tx_param(IP, PORT, param)
    time.sleep(.2)
    if res:
        run_experiment(IP, PORT, param)
    time.sleep(.2)