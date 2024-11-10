from src import UDP_RX, SlidingBook, MACChecker
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json
import socket
import struct
import pickle
import sys
import argparse
import json


sys.path.append('..')
import utils.utils as utils
import utils.Auth as Auth


# Getting the Parameters from the sender and Calculating the offset
def receive_param(IP, PORT):
    param = b''
    offset = 0
    flag = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP, PORT))
        s.listen()
        conn, addr = s.accept()
        flag = True
        with conn:
            while True:
                data = conn.recv(1024)
                if not data or data == b'END':
                    break
                param += data
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data2 = conn.recv(16)
            offset = time.time() - struct.unpack('d', data2)[0]

            try:
                param = json.loads(param.decode('utf-8'))
                param['OFFSET'] = offset
            except:
                print("Error in decoding the parameters")
                conn.sendall(b'RETRY')
                flag = False
                return False


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            if exist_in_resutls(param):
                print("The parameters already exist in the results")
                conn.sendall(b'ABORT')
                flag = False
            else:
                conn.sendall(b'OK')
                flag = True

    if flag:
        return param
    else:
        return False


def run_experiment(IP, PORT, param):
    total_avg_verification = []
    goodput_total = []
    total_latency = []
    framecnt_total = []

    cnt = 0
    framecnt = 0
    start_frame_cnt = time.time()
    start = time.time()

    while True:

        buffer =  SlidingBook(num_pages=100, page_size=len(param['X']))
        page_processor = MACChecker(X = param['X'], Y = param['Y'], secret_key=param['KEY'].encode(), digestmod=param['DIGESTMOD'])
        udp_rx = UDP_RX(IP= IP, PORT= PORT, buffer=buffer, page_processor= page_processor, Payload_Size_Byte=param['PAYLOAD_SIZE_BYTE'])

        msg, verification_count, latency, goodput = udp_rx.receive()
        total_avg_verification.append(np.average(verification_count))
        total_latency.append(np.average(latency))
        goodput_total.append(goodput)


        if msg is not None and len(msg) > 0:  # Ensure rec is not None and has valid data
            nparr = np.frombuffer(msg, np.uint8)
            
            if nparr is not None and len(nparr) > 0:  # Check if nparr is valid
                try:
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:  # Ensure that the frame is successfully decoded
                        cv2.imshow('Received Video', frame)
                        framecnt += 1
                        if time.time() - start_frame_cnt > 1:
                            framecnt_total.append(framecnt)
                            start_frame_cnt = time.time()
                            print(f"{framecnt} fps")
                            framecnt = 0
                            
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        # print("Failed to decode frame.")
                        pass
                except Exception as e:
                    # print(f"Error decoding frame: {e}")
                    pass
            else:
                # print("Empty or invalid buffer received.")
                pass
        else:
            # print("Received an empty or invalid page.")
            pass

        if cnt % 100 == 0:
            print(f"avg_verification: {np.average(total_avg_verification)}, latency: {np.average(total_latency)}, goodput: {np.average(goodput_total)}")
        cnt += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start > param['DURATION']:
            break

    cv2.destroyAllWindows()

    ## saving the parameters and the results as a json file

    results = {
        "avg_verification": total_avg_verification,
        "latency": total_latency,
        "goodput": goodput_total,
        "frame_rate": framecnt_total,
        "param": param
    }   

    # read a pickle file to get the previous results

    try:
        with open("results.pkl", 'rb') as f:
            file = pickle.load(f)
            experiment_nr = len(file)
            file[experiment_nr]=results
            with open("results.pkl", 'wb') as f:
                pickle.dump(file, f)
    except:
        # create a new file
        print("Creating a new file")
        file = {0:results}
        with open("results.pkl", 'wb') as f:
            pickle.dump(file, f)
    
def exist_in_resutls(param):
    try:
        with open("results.pkl", 'rb') as f:
            file = pickle.load(f)
            for key in file.keys():
                # check all the pram except the offset
                if file[key]['param']['X'] == param['X'] and file[key]['param']['Y'] == param['Y'] and file[key]['param']['QUALITY'] == param['QUALITY'] and file[key]['param']['DIGESTMOD'] == param['DIGESTMOD'] and file[key]['param']['PAYLOAD_SIZE_BYTE'] == param['PAYLOAD_SIZE_BYTE'] and file[key]['param']['ATTACK_PROBABILITY'] == param['ATTACK_PROBABILITY']:
                    return True
            # if file[key]['param'] == param:
        return False
    except:
        return False



if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('my_dict', type=str)
    args = parser.parse_args()

    parameters = json.loads(args.my_dict)

    IP = parameters['IP']
    PORT = parameters['PORT']

    param = receive_param(IP=IP, PORT=PORT)
    if param:
        run_experiment(IP=IP, PORT=PORT, param=param)