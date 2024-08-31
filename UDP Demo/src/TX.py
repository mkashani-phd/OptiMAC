import cryptography.hazmat.primitives.padding as padding
import numpy as np
import socket
import hmac
import pprint
import time
import struct


class UDP_TX:
    def __init__(self, IP:str ='0.0.0.0', PORT:int = 23422, X = np.eye(10), Y = np.eye(10), chunk_size_Byte=128, KEY=b"key", digestmod='sha384'):
        self.IP = IP
        self.PORT = PORT
        self.X = X
        self.Y = Y
        self.chunk_size_Byte = chunk_size_Byte
        self.KEY = KEY
        self.digestmod = digestmod

    # Padding using PKCS7
    def pad(self,data, chunk_size_Byte = None):
        if chunk_size_Byte is None:
            chunk_size_Byte = self.chunk_size_Byte
        
        padder = padding.PKCS7(chunk_size_Byte*8).padder()
        padded_data = padder.update(data) + padder.finalize()
        return padded_data
    
    def unpad(self,data):
        unpadder = padding.PKCS7(self.chunk_size_Byte*8).unpadder()
        return unpadder.update(data) + unpadder.finalize()


    def divide_Bytes_and_pad(self,Bytes, chunk_size_Byte=None):
        if chunk_size_Byte is None:
            chunk_size_Byte = self.chunk_size_Byte
        
        chunks = [Bytes[i:i + chunk_size_Byte] for i in range(0, len(Bytes), chunk_size_Byte)]
        # if len(chunks[-1]) != chunk_size_Byte:
        #     chunks[-1] = self.pad(chunks[-1], chunk_size_Byte)
        return chunks

    def chucks_to_nD_arrangments(self,chunks,chunk_size_Byte = None, X= None):
        if chunk_size_Byte is None:
            chunk_size_Byte = self.chunk_size_Byte
        if X is None:
            X = self.X

        nD_arrangments = [chunks[i:i + X.shape[0]] for i in range(0, len(chunks), X.shape[0])]
        if len(nD_arrangments[-1]) != X.shape[0]:
            for i in range( X.shape[0] - len(nD_arrangments[-1]))  :
                nD_arrangments[-1].append(b'')
                # nD_arrangments[-1].append(self.pad(b'',chunk_size_Byte=chunk_size_Byte))
        return np.array(nD_arrangments)

    def mac_for_page(self,page, key = None, X = None, Y = None):
        if key is None:
            key = self.KEY
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        
        res = {}
        for msg_index in range(X.shape[0]):
            res[msg_index] = page[msg_index]
        
        for tag_index in range(X.shape[1]):
            selected_blocks = page[X[:, tag_index] == 1]
            if selected_blocks.size > 0:
                data = b''.join(selected_blocks)
                target_index = np.where(Y[:, tag_index] == 1)[0][0]
                res[target_index] = page[target_index] + hmac.new(key, data, digestmod=self.digestmod).digest()
        return res

    # SN: Sequence Number (4Bytes)
    # time: time          (8Bytes)
    # msg: message        (chunk_size_Byte Bytes)  
    def send_msg(self,SN, msg, sock, dest):
        sock.sendto(SN.to_bytes(4, 'big') + struct.pack('d', time.time()) + msg, dest)
        time.sleep(0.000001)


    def transmit(self, Bytes, attack:list=[]):

        SN = 0
        cnt = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            chunks = self.divide_Bytes_and_pad(Bytes = Bytes, chunk_size_Byte= self.chunk_size_Byte)
            pages = self.chucks_to_nD_arrangments(chunks = chunks, chunk_size_Byte= self.chunk_size_Byte, X =  self.X)

            for i in range(len(pages)):
                page = self.mac_for_page(page=pages[i], key= self.KEY, X= self.X, Y= self.Y)
                for msg in page.values():
                    if SN in attack:
                        SN += 1
                        msd = b'ATTACK'+b'0'*(self.chunk_size_Byte-6)
                        continue
                    self.send_msg(SN, msg, sock, (self.IP,  self.PORT))
                    cnt += 1
                    SN += 1
            ### send the end message this is not secure but it is just for testing
            sock.sendto(b'END', ( self.IP,  self.PORT))
        return cnt

# unit test

if __name__ == "__main__":

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

    upd_tx = UDP_TX(IP, PORT, X, Y,  chunk_size_Byte = chunk_size_Byte, KEY=key, digestmod = digestmod)

    data = b'This test shows 2D integrity check is better than blockwise integrity.'

    # testing the divide_Bytes_and_pad function

    print("Dividing the Byte data every to chunk_size_Bytes and pad the last chunck if needed",upd_tx.divide_Bytes_and_pad(data, chunk_size_Byte=chunk_size_Byte))

    # testing the chucks_to_nD_arrangments function witht X.shape[0] and chunk_size
    chunks = upd_tx.divide_Bytes_and_pad(data, chunk_size_Byte=chunk_size_Byte)
    print(upd_tx.chucks_to_nD_arrangments(chunks, chunk_size_Byte, X))

    # testing the mac_for_page function
    pages = upd_tx.chucks_to_nD_arrangments(chunks, chunk_size_Byte, X)
    pprint.pprint(upd_tx.mac_for_page(pages[0], key, X, Y))

    # Emulating the transmitted Packets
    simulation = upd_tx.mac_for_page(pages[0], key, X, Y)
    SN = 0
    for msg in simulation:
        simulation[msg] = SN.to_bytes(4, 'big') + struct.pack('d', time.time()) +  simulation[msg]
        SN += 1
    pprint.pprint(simulation)
    # at this point after calculating the MAC for each page, we can send the data
    # iterate through the chunks (msg) in every page and send it to the destination
    # the function send_msg is responsible for sending the data to the destination
    # the function transmit input bytes and create pages with padded chunks  and 
    # then iterates through the all the chunks in the pages and send them to the destination

    # testing the transmit function
    # print("Number of messages sent: ", upd_tx.transmit(data))
    # testing the 


    print("Number of messages sent under attack: ", upd_tx.transmit(data, attack=[1]))




