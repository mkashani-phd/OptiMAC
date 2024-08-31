import cryptography.hazmat.primitives.padding as padding
import numpy as np
import socket
import hmac
import struct
import time



# define a Buffer with the size of chunck_size_Byte * X.shape[0] * number_of_pages

# Buffer size is in pages where each page is a list of X.shape[0]*chuncks and every chunck is of size chunk_size_Byte
# the buffer is a list of pages
# every page is a dict of chuncks in the format of SN:(msg,mac)
# every chunck is a list of bytes
# the buffer is updated when a new message is received
# the buffer is used to reorder the messages

# page zero keep a timer of the latest time the page is updated
# this is used to remove the old page if a SN comes and the time from the last update is more than timeout
class Buffer:

    def __init__(self, X, Y, BUFFER_SIZE_IN_PAGES = 10, TIMEOUT_SECOND = 1,  warnings = True):
    
        # defining the buffer with maxlen of number_of_pages
        # some chunks are padded
        self.X = X
        self.Y = Y

        self.BUFFER_SIZE_IN_PAGES = BUFFER_SIZE_IN_PAGES
        self.BUFFER = []

        #initialize the buffer with empty page 
        for i in range(0,BUFFER_SIZE_IN_PAGES):
            self.add_page()
 

        self.TIMEOUT_SECOND = TIMEOUT_SECOND
        self.PAGE_ZERO_LAST_UPDATE = time.time()

        self.MIN_SN = 0

        self.warnings = warnings


    def clear_buffer(self):
        self.BUFFER = []
        for i in range(0,self.BUFFER_SIZE_IN_PAGES):
            self.add_page()
        self.MIN_SN = 0
        self.PAGE_ZERO_LAST_UPDATE = time.time()

    def get_min_allowed_SN(self):
        return self.MIN_SN
    def get_max_allowed_SN(self):
        return self.MIN_SN - (self.MIN_SN % self.X.shape[0]) + self.X.shape[0]*len(self.BUFFER) -1
    

    def sort_SN_in_page(self, page):
        return {k: v for k, v in sorted(page.items(), key=lambda item: item[0])}

    def add_page(self):
        if len(self.BUFFER)  < self.BUFFER_SIZE_IN_PAGES:
            self.BUFFER.append({})
            return True
        if self.warnings:
            print("The buffer is full, increase the buffer size or this might be an attack to the buffer.")
        return False
    
    def pop_page(self, page_index):
        if page_index > len(self.BUFFER) or page_index < 0:
            if self.warnings:
                print(f"Page {page_index} is out of range, the buffer size is {len(self.BUFFER)}")
            return None
        if page_index == 0:
            self.PAGE_ZERO_LAST_UPDATE = time.time()
            self.MIN_SN += self.X.shape[0]

        temp = self.sort_SN_in_page(self.BUFFER.pop(page_index))
        self.add_page()
        return temp
    

    def get_page_index_by_SN(self, SN):
        return  ((SN-self.MIN_SN) // self.X.shape[0])%self.BUFFER_SIZE_IN_PAGES
    

    def is_page_full(self, page_index):
        return len(self.BUFFER[page_index]) == self.X.shape[0]

    # three possible return values None, page, (page, None, SN), (page, (page, None, SN), SN)
    def add_msg_to_page(self, SN, time_stamp, msg, mac = b''):
        l, r = self.get_min_allowed_SN(), self.get_max_allowed_SN()
        if  SN < l or SN > r and self.warnings:
            print(f"SN {SN} is out of range [{l,r}] (Buffer full), increase the buffer size or this might be an attack to the buffer.")
            if time.time() - self.PAGE_ZERO_LAST_UPDATE < self.TIMEOUT_SECOND:
                print(f" The message SN: {SN} is dropped. The buffer is full and the page zero will be kept until {self.TIMEOUT_SECOND -time.time() + self.PAGE_ZERO_LAST_UPDATE} more seconds")
                return None
            else:
                min_sn = self.get_min_allowed_SN()
                res = self.pop_page(0)
                temp = self.add_msg_to_page(SN, msg, time_stamp)
                return res ,temp, min_sn 
            

        page_index = self.get_page_index_by_SN(SN)

        if SN in self.BUFFER[page_index] :
            if self.warnings:
                print("Message already exists in the buffer! Replay attack?")
            return None
        
        if page_index == 0: 
            self.PAGE_ZERO_LAST_UPDATE = time.time()
        
        self.BUFFER[page_index][SN] = (msg, mac, time_stamp)
          
        if self.is_page_full(page_index):
            return self.pop_page(page_index)

        return None
    
    def print_buffer(self):
        print(f"{self.MIN_SN} Buffer:", self.BUFFER)
    


from collections import deque, OrderedDict
import numpy as np
import time

class OptimizedBuffer:
    def __init__(self, X, Y, BUFFER_SIZE_IN_PAGES=10, TIMEOUT_SECOND=1, warnings=True):
        self.X = X
        self.Y = Y
        self.BUFFER_SIZE_IN_PAGES = BUFFER_SIZE_IN_PAGES
        self.BUFFER = deque(maxlen=BUFFER_SIZE_IN_PAGES)
        self.TIMEOUT_SECOND = TIMEOUT_SECOND
        self.PAGE_ZERO_LAST_UPDATE = time.time()
        self.MIN_SN = 0
        self.MAX_SN = self.MIN_SN + (self.X.shape[0] * self.BUFFER_SIZE_IN_PAGES) - 1
        self.warnings = warnings

        # Initialize the buffer with empty pages
        for _ in range(BUFFER_SIZE_IN_PAGES):
            self.BUFFER.append(OrderedDict())

    def get_min_allowed_SN(self):
        return self.MIN_SN

    def get_max_allowed_SN(self):
        return self.MAX_SN

    def add_page(self):
        if len(self.BUFFER) < self.BUFFER_SIZE_IN_PAGES:
            self.BUFFER.append(OrderedDict())
            self.MAX_SN = self.MIN_SN + (self.X.shape[0] * len(self.BUFFER)) - 1
            return True
        if self.warnings:
            print("The buffer is full, increase the buffer size or this might be an attack on the buffer.")
        return False

    def pop_page(self, page_index):
        if page_index < 0 or page_index >= len(self.BUFFER):
            if self.warnings:
                print(f"Page {page_index} is out of range, the buffer size is {len(self.BUFFER)}")
            return None
        if page_index == 0:
            self.PAGE_ZERO_LAST_UPDATE = time.time()
            self.MIN_SN += self.X.shape[0]
            self.MAX_SN = self.MIN_SN + (self.X.shape[0] * len(self.BUFFER)) - 1

        temp = self.BUFFER[page_index]
        self.BUFFER.remove(temp)
        self.add_page()
        return temp

    def get_page_index_by_SN(self, SN):
        return ((SN - self.MIN_SN) // self.X.shape[0]) % self.BUFFER_SIZE_IN_PAGES

    def is_page_full(self, page_index):
        return len(self.BUFFER[page_index]) == self.X.shape[0]

    def add_msg_to_page(self, SN, msg, mac=b''):
        if SN < self.MIN_SN or SN > self.MAX_SN:
            if self.warnings:
                print(f"SN {SN} is out of range [{self.MIN_SN},{self.MAX_SN}] (Buffer full), increase the buffer size or this might be an attack on the buffer.")
            if time.time() - self.PAGE_ZERO_LAST_UPDATE < self.TIMEOUT_SECOND:
                print(f"The message SN: {SN} is dropped. The buffer is full, and page zero will be kept for {self.TIMEOUT_SECOND - time.time() + self.PAGE_ZERO_LAST_UPDATE} more seconds.")
                return None
            else:
                min_sn = self.get_min_allowed_SN()
                res = self.pop_page(0)
                temp = self.add_msg_to_page(SN, msg, mac)
                return res, temp, min_sn

        page_index = self.get_page_index_by_SN(SN)

        if SN in self.BUFFER[page_index] and self.warnings:
            return "Message already exists in the buffer! Replay attack?"

        if page_index == 0:
            self.PAGE_ZERO_LAST_UPDATE = time.time()

        self.BUFFER[page_index][SN] = (msg, mac)

        if self.is_page_full(page_index):
            return self.pop_page(page_index)

        return None

    def print_buffer(self):
        print(f"{self.MIN_SN} Buffer:", list(self.BUFFER))

# Running a quick test to verify the changes

if __name__ == "__main__":
    X = np.eye(3)
    Y = np.eye(3)

    buffer = OptimizedBuffer(X, Y, BUFFER_SIZE_IN_PAGES=3, TIMEOUT_SECOND=1, warnings=True)

    print(buffer.get_min_allowed_SN())
    print(buffer.get_max_allowed_SN())

    print(buffer.add_page())

    print(buffer.get_page_index_by_SN(0))
    print(buffer.get_page_index_by_SN(1))

    print(buffer.is_page_full(0))

    print(buffer.add_msg_to_page(7, 'msg7'))
    print(buffer.add_msg_to_page(3, 'msg3'))
    print(buffer.add_msg_to_page(0, 'msg0'))
    print(buffer.add_msg_to_page(1, 'msg1'))
    print(buffer.add_msg_to_page(2, 'msg2'))

    print(buffer.add_msg_to_page(4, 'msg4'))
    print(buffer.add_msg_to_page(5, 'msg5'))

    print(buffer.add_msg_to_page(6, 'msg6'))

    print(buffer.add_msg_to_page(8, 'msg8'))

    print(buffer.add_msg_to_page(-1, 'msg0'))

    print(buffer.add_msg_to_page(100, 'msg0'))
    time.sleep(1.1)
    print(buffer.add_msg_to_page(20, 'msg0'))
    time.sleep(1.1)
    print(buffer.add_msg_to_page(100, 'msg0'))


class UDP_RX:
    def __init__(self,buffer= None, IP:str ='0.0.0.0', PORT:int = 23422, X = np.eye(3), Y = np.eye(3),  chunk_size_Byte=128, KEY=b"key", digestmod='sha384', BUFFER_SIZE_IN_PAGES = 10):
        self.IP = IP
        self.PORT = PORT
        self.X = X
        self.Y = Y
        self.chunk_size_Byte = chunk_size_Byte
        self.KEY = KEY
        self.digestmod = digestmod
        self.HAMC_SIZE = hmac.new(KEY, b'', digestmod=digestmod).digest_size
        self.BUFFER_SIZE_IN_PAGES = BUFFER_SIZE_IN_PAGES

        if buffer is None:
            self.BUFFER = Buffer( X, Y, chunk_size_Byte, BUFFER_SIZE_IN_PAGES)
        else:
            self.BUFFER = buffer

    def unpad(self,data):
        unpadder = padding.PKCS7(self.chunk_size_Byte*8).unpadder()
        return unpadder.update(data) + unpadder.finalize()
        
    def parse_msg(self, data):
        time_stamp = struct.unpack('d', data[4:12])[0]
        SN = int.from_bytes(data[:4], 'big')
        if np.sum(self.Y[SN % self.X.shape[0]]):
            chunk_data = data[12:-self.HAMC_SIZE]
            mac = data[-self.HAMC_SIZE:]
        else:
            chunk_data = data[12:]
            mac = b''
        return SN,time_stamp, chunk_data, mac
    
    # def veify_page(self, page:dict,verified_page:dict, key = None, X=None, Y= None):
    #     if key is None:
    #         key = self.KEY
    #     if X is None:
    #         X = self.X
    #     if Y is None:
    #         Y = self.Y

    #     page_array = np.array(list(page.values()))
    #     try:
    #         for SN in page.keys():
    #             verified_page[SN] = np.array([page_array[SN%X.shape[0]][0],0])
            
    #         for tag_index in range(X.shape[1]):
    #             selected_blocks = page_array[X[:, tag_index] == 1][:,0]
    #             if selected_blocks.size > 0:
    #                 corresponding_data = b''.join(selected_blocks)
    #                 recieved_mac = page_array[np.where(Y[:, tag_index] == 1)[0][0]][1]

    #                 if recieved_mac == hmac.new(self.KEY, corresponding_data, digestmod=self.digestmod).digest():
    #                     # print("Verified", res)
    #                     for SN in np.array(list(page.keys()))[X[:, tag_index] == 1]:
    #                         verified_page[SN][1] = int(verified_page[SN][1]) + 1

    #                 else:
    #                     pass
    #     except:
    #         verified_page = None
    #     return verified_page
    
    def fill_missing_in_page_with_zeros(self, page:dict, SN: int):
        for i in range(SN, SN + self.X.shape[0]):
            if i not in page:
                page[i] = (b'', b'')
        return page

    def receive(self):
        self.BUFFER.clear_buffer()
        total_res = {}
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.IP, self.PORT))
            while True:
                data, addr = sock.recvfrom(4096+48)
                if data == b'END':
                    break
                SN, time_stamp, chunk_data, mac = self.parse_msg(data)
                res = self.BUFFER.add_msg_to_page(SN,time_stamp, chunk_data, mac)
                # 4 stages are possible None, page, (page, None, SN), (page, (page, None, SN), SN)
                total_res = self.process_buffer_respond(res, total_res=total_res)
                
        return total_res
    

    
    def verify_page(self, page:dict,verified_page:dict, key = None, X=None, Y= None):
        if key is None:
            key = self.KEY
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        page_array = np.array(list(page.values()))
        try:
            for SN in page.keys():
                verified_page[SN] = np.array([page_array[SN%X.shape[0]][0],0, page_array[SN%X.shape[0]][2]])
            
            for tag_index in range(X.shape[1]):
                selected_blocks = page_array[X[:, tag_index] == 1][:,0]
                if selected_blocks.size > 0:
                    corresponding_data = b''.join(selected_blocks)
                    recieved_mac = page_array[np.where(Y[:, tag_index] == 1)[0][0]][1]

                    if recieved_mac == hmac.new(self.KEY, corresponding_data, digestmod=self.digestmod).digest():
                        # print("Verified", res)
                        for SN in np.array(list(page.keys()))[X[:, tag_index] == 1]:
                            verified_page[SN][1] = int(verified_page[SN][1]) + 1

                    else:
                        pass
        except:
            verified_page = None
        return verified_page


    # 4 stages are possible
    # Noraml: None, page,
    # Forced poped page0 of the buffer: (page, None, SN), (page, (page, None, SN), SN)
    def process_buffer_respond(self, add_msg_to_page_res, total_res):
        if add_msg_to_page_res is not None:
            if isinstance(add_msg_to_page_res, dict):
                total_res = self.verify_page(add_msg_to_page_res, total_res)

            elif isinstance(add_msg_to_page_res, tuple):  
                page0 = self.fill_missing_in_page_with_zeros(page = add_msg_to_page_res[0], SN = add_msg_to_page_res[2])
                total_res = self.verify_page(page0, total_res)
                if add_msg_to_page_res[1] is not None: 
                    print("add_msg_to_page_res", add_msg_to_page_res[1])
                    total_res = self.process_buffer_respond(add_msg_to_page_res[1], total_res)
        
        return total_res

    
    def process_verified_page(self, verified_page, chunksize=7, print_results=False, latency_penalty_For_Not_Verified = 0.04):
        total_messages = len(verified_page)
        verified_count = 0
        not_verified_count = 0
        verification_attempts = 0
        total_verified_instances = 0
        total_Auth_latency = 0
        result = []

        if verified_page is None:
            if print_results:
                print("No verified page")
            return None, 0, 0, 0, 0
        
        # Sort the dictionary by sequence number
        sorted_keys = sorted(verified_page.keys())
        
        for key in sorted_keys:
            message, verified, time_stamp = verified_page[key]
            verification_attempts += 1
            
            if int(verified) > 0 :
                # try:
                #     message = self.unpad(message)
                # except:
                #     pass
                total_Auth_latency += time.time() - float(time_stamp)
                result.append(message)  # Append the verified message
                verified_count += 1
                total_verified_instances += int(verified)
            else:
                total_Auth_latency += latency_penalty_For_Not_Verified
                result.append(b'\x00' * chunksize)  # Replace with asterisks if not verified
                not_verified_count += 1
        
        # Join the messages to form the complete output
        output_message = b''.join(result)
        
        # Calculate statistics
        average_verifications_per_message = verified_count / total_messages if total_messages > 0 else 0
        missing_messages = not_verified_count
        # Print the results
        if print_results:
            print("Output Message:", output_message)
            print("Average Verifications per Message:", average_verifications_per_message)
            print("Total Messages Not Verified:", missing_messages)
            print("Total Verification Attempts:", verification_attempts)  
            print("Total Authentication Latency:", total_Auth_latency)
        

        return output_message, average_verifications_per_message, missing_messages, total_verified_instances/verified_count if verified_count > 0 else 0, total_Auth_latency/verified_count if verified_count > 0 else 0



## unit test for the page verifier
if __name__ == "__main__":
    # unit test for the buffer
    # X = np.eye(3)
    # Y = np.eye(3)

    # buffer = Buffer(X, Y, BUFFER_SIZE_IN_PAGES = 3, TIMEOUT_SECOND = 1,  warnings = True)

    # print(buffer.get_min_allowed_SN())
    # print(buffer.get_max_allowed_SN())

    # print(buffer.add_page())

    # print(buffer.get_page_index_by_SN(0))
    # print(buffer.get_page_index_by_SN(1))

    # print(buffer.is_page_full(0))

    # print(buffer.add_msg_to_page(7, 'msg7'))
    # print(buffer.add_msg_to_page(3, 'msg3'))
    # print(buffer.add_msg_to_page(0, 'msg0'))
    # print(buffer.add_msg_to_page(1, 'msg1'))
    # print(buffer.add_msg_to_page(2, 'msg2'))

    # print(buffer.add_msg_to_page(4, 'msg4'))
    # print(buffer.add_msg_to_page(5, 'msg5'))

    # print(buffer.add_msg_to_page(6, 'msg6'))

    # print(buffer.add_msg_to_page(8, 'msg8'))

    # print(buffer.add_msg_to_page(-1, 'msg0'))

    # print(buffer.add_msg_to_page(100, 'msg0'))
    # time.sleep(1.1)
    # print(buffer.add_msg_to_page(20, 'msg0'))
    # time.sleep(1.1)
    # print(buffer.add_msg_to_page(100, 'msg0'))
        

    #### parameters that needs to be exhanged between the sender and the receiver #####
    IP = "0.0.0.0"
    PORT = 23422
            #  t1  t2  t3  t4  t5  t6  t7  t8  t9
    X = np.array([[ 1,  0,  0,  0,  0,  0,  1,  0,  0], # m1
                [ 1,  0,  0,  0,  0,  0,  0,  1,  0], # m2
                [ 1,  0,  0,  0,  0,  0,  0,  0,  1], # m3
                [ 0,  1,  0,  0,  0,  0,  1,  0,  0], # m4
                [ 0,  1,  0,  0,  0,  0,  0,  1,  0], # m5
                [ 0,  1,  0,  0,  0,  0,  0,  0,  1], # m6
                [ 0,  0,  1,  0,  0,  0,  1,  0,  0], # m7
                [ 0,  0,  1,  0,  0,  0,  0,  1,  0], # m8
                [ 0,  0,  1,  0,  0,  0,  0,  0,  1], # m9
                [ 0,  0,  0,  1,  0,  0,  1,  0,  0], # m10
                [ 0,  0,  0,  1,  0,  0,  0,  1,  0], # m11
                [ 0,  0,  0,  1,  0,  0,  0,  0,  1], # m12
                [ 0,  0,  0,  0,  1,  0,  1,  0,  0], # m13
                [ 0,  0,  0,  0,  1,  0,  0,  1,  0], # m14
                [ 0,  0,  0,  0,  1,  0,  0,  0,  1], # m15
                [ 0,  0,  0,  0,  0,  1,  1,  0,  0], # m16
                [ 0,  0,  0,  0,  0,  1,  0,  1,  0], # m17
                [ 0,  0,  0,  0,  0,  1,  0,  0,  1]]) # m18
                #  t1  t2  t3  t4  t5  t6  t7  t8  t9
    Y = np.array([[ 0,  0,  0,  0,  0,  0,  1,  0,  0], # m1
                [ 0,  0,  0,  0,  0,  0,  0,  1,  0], # m2
                [ 1,  0,  0,  0,  0,  0,  0,  0,  0], # m3
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m4
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m5
                [ 0,  1,  0,  0,  0,  0,  0,  0,  0], # m6
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m7
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m8
                [ 0,  0,  1,  0,  0,  0,  0,  0,  0], # m9
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m10
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m11
                [ 0,  0,  0,  1,  0,  0,  0,  0,  0], # m12
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m13
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m14
                [ 0,  0,  0,  0,  1,  0,  0,  0,  0], # m15
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m16
                [ 0,  0,  0,  0,  0,  1,  0,  0,  0], # m17
                [ 0,  0,  0,  0,  0,  0,  0,  0,  1]]) # m18
    chunk_size_Byte = 2
    key = b"key"
    digestmod = 'sha384'



    buffer = Buffer(X, Y, BUFFER_SIZE_IN_PAGES = 3, TIMEOUT_SECOND = 0.00001,  warnings = True)
    udp_rx = UDP_RX(buffer= buffer, IP = IP, PORT = PORT, X = X, Y = Y,  chunk_size_Byte=chunk_size_Byte, KEY=key, digestmod=digestmod)
    # verified_page = udp_rx.receive()


    
    x = {0: b'\x00\x00\x00\x00}{\x0cw\x15_\x0fATh\x045\x17o<~\xa1\xb5\x88<\x93\x87gQ'
            b'p\x07\r?\x88\xc8\xef\x17\xdbD\xab\xd5\xdc\xce\x99#\x8c\xcb\x82p'
            b'\x8eA\x07\xc7\xa7C\xad\xf9\xb3M\xe0\x9d6!',
        1: b'\x00\x00\x00\x01\x02k\x14w\x15_\x0fAis5Fl\xab\x18\xa7\xe07G\x10'
            b'\xdf\x94\x08\x9d\xa1\x86O\\\x85\xe4a\x8b\xbe\xc7\r\x81\xe0\x06\x03K'
            b'\xde\x0c\x82\xb5gV\x1e\xe82\x18\xf4\x80\xb0\xa1\xc2@\x8b\x9a',
        2: b'\x00\x00\x00\x02O\xd5\x14w\x15_\x0fA t\xf0I\x83\xd4\x18l\xefsQ\xf1\x17dol'
            b'E\x8a\x97\xc3E\xdfiS\xf8\x03^9M\x88\xb3\xf9\xefzjI\xf5\xdaV\xc1'
            b'm\xce\xa4\x89\x13\xf9\xf2\xa3<\x04',
        3: b'\x00\x00\x00\x03\xa1\x05\x15w\x15_\x0fAes',
        4: b'\x00\x00\x00\x04\x9c?\x15w\x15_\x0fAt ',
        5: b'\x00\x00\x00\x05\x17t\x15w\x15_\x0fAsh7\xb2\xfd\xe9\xb7D\xe8R\xb1\x14'
            b'\xa5\xac3`\xbb\xa7\x90\xfc\x1a\xaa\x13n\x9b\x86\xf8\x13\x8a\xab\x9cw'
            b"m\xe3\xb9\x81'\x80w0\x08\x9f\x99pf\x80@\xdd!\x0f",
        6: b'\x00\x00\x00\x06\xc5\x9e\x15w\x15_\x0fAow',
        7: b'\x00\x00\x00\x07\xb1\xc5\x15w\x15_\x0fAs ',
        8: b'\x00\x00\x00\x08\x87\xe9\x15w\x15_\x0fA2Db\xf8\x81P\xaf*\xdb\x1b\xb9\\'
            b'\x9c\x0c@\x06\xb5\xb7t\xeb\x15;\xc3NrIU\xddm\xec\x13\xa0\xeb2\xb3?'
            b'6j\x19\xfc\xd9\xb5\xe8:jnKt$\xf5',
        9: b'\x00\x00\x00\t\xd8\x0e\x16w\x15_\x0fA i',
        10: b'\x00\x00\x00\n(4\x16w\x15_\x0fAnt',
        11: b'\x00\x00\x00\x0bo^\x16w\x15_\x0fAeg\x04\xed$\x17\xa00\xc9}\xe23'
            b'\x92\x8e\xa0\x11g\xba\x03\x8b\x08(\xd06y7?Z0\x8d\xb2\xc2\xe7\x7f.m'
            b'\xb8{h\xb3\x8b\xae\xa2\x10\x85*Wx\xa6;',
        12: b'\x00\x00\x00\x0c3\x8c\x16w\x15_\x0fAri',
        13: b'\x00\x00\x00\r\x10\xb4\x16w\x15_\x0fAty',
        14: b'\x00\x00\x00\x0e^\xe2\x16w\x15_\x0fA cx\x1e0\x8aU \xfc\xa7\xe8;'
            b'\x0e\xa6(\xa6\xa12S\xf7\xde\xb3\xec\xc5\xccZO-\x1f,\xf8\x98\xdaxQ"'
            b'\xc6.5g\x08-\x9d\x85p\xdcD/\xc2Z',
        15: b'\x00\x00\x00\x0f\xf3\x07\x17w\x15_\x0fAhe',
        16: b'\x00\x00\x00\x10\x88-\x17w\x15_\x0fAck\xaa\xc9\xc5\x98\xe2a\xf2y\xf6\xf1'
            b'i,\xd9eq\x15\xc7X<\xdc\x1fV(\x1b\xd2=\x9c\xce7Og\xb8\xa3.'
            b'\xaf\x04\xb7\x95\x02\xbe\xe2\x85\xa6#H\xdd\xd8\x8c',
        17: b'\x00\x00\x00\x11,R\x17w\x15_\x0fA i+\xc9\xbc\x88b\xcf\xd4*\xa3.'
            b"\x94\x9a\x16fpZ\xa9\x85'\xf3\x1c\xa7\x9f\xb5\x11\xe8@6\xf3\xbd"
            b'Z\xf2\xc7\xa0\xca[\xf3\x94\xb8t\xa6\xeal\xcd|\rp\xe2'}

    for msg in x:
        print(temp:= buffer.add_msg_to_page(*udp_rx.parse_msg(x[msg])))

    verified_page = {}
    verified_page = udp_rx.verify_page(temp, verified_page)

    print(udp_rx.process_verified_page(verified_page, print_results=True))
    page = {0:[]}
    print(udp_rx.verify_page(page,verified_page))
