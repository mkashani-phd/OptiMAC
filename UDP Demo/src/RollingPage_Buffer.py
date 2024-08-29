
# define a Buffer with the size of chunck_size_Byte * X.shape[0] * number_of_pages
from collections import OrderedDict
import numpy as np
import time


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

    # three possible return values None, (page,None), (page, page)
    def add_msg_to_page(self, SN, msg, mac = b''):
        l, r = self.get_min_allowed_SN(), self.get_max_allowed_SN()
        if  SN < l or SN > r and self.warnings:
            print(f"SN {SN} is out of range [{l,r}] (Buffer full), increase the buffer size or this might be an attack to the buffer.")
            if time.time() - self.PAGE_ZERO_LAST_UPDATE < self.TIMEOUT_SECOND:
                print(f" The message SN: {SN} is dropped. The buffer is full and the page zero will be kept until {self.TIMEOUT_SECOND -time.time() + self.PAGE_ZERO_LAST_UPDATE} more seconds")
                return None
            else:
                min_sn = self.get_min_allowed_SN()
                res = self.pop_page(0)
                temp = self.add_msg_to_page(SN, msg)
                return res ,temp, min_sn 
            

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
        print(f"{self.MIN_SN} Buffer:", self.BUFFER)
    

# unit test for the buffer
if __name__ == "__main__":
    X = np.eye(3)
    Y = np.eye(3)

    buffer = Buffer(X, Y, BUFFER_SIZE_IN_PAGES = 3, TIMEOUT_SECOND = 1,  warnings = True)

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