from .nD_MAC import MACChecker, SlidingBook, Packet
import numpy as np
import socket





class UDP_RX:

    def __init__(self, buffer: SlidingBook, page_processor:MACChecker,   IP:str ='0.0.0.0', PORT:int = 23422, Payload_Size_Byte=128):
        self.IP = IP
        self.PORT = PORT
        self.Payload_Size_Byte = Payload_Size_Byte
        self.buffer = buffer
        self.page_processor = page_processor
    


    def receive(self):
        self.buffer.clear_all()
        total_message =  {}
        # res = b''
        total_latency = []
        total_verification = []
        len_total_data = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.IP, self.PORT))
            while True:
                data, addr = sock.recvfrom(4096 + 48 + 4 + 8) 
                if data == b'END':
                    break
                len_total_data += len(data) -4 -8
                page = self.buffer.add_packet(Packet.from_bytes(data))
                if page is not None:
                    reconcatenated_message, verification_counts, latencys = self.page_processor.check_page(page)
                    total_message [page.min_SN] = reconcatenated_message
                    # res += reconcatenated_message
                    total_latency.append(np.nanmean(latencys))
                    total_verification.append(np.nanmean(verification_counts))
        # join the messages form low SN to high SN
        res = b''.join([total_message[key] for key in sorted(total_message.keys())])  
        del total_message  
        Goodput = len(res)/len_total_data if len_total_data != 0 else 0
        return res,total_verification, total_latency, Goodput

    


## unit test for the page verifier
if __name__ == "__main__":
    #### parameters that needs to be exhanged between the sender and the receiver #####
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
    
    IP = "0.0.0.0"
    PORT = 23422
    Packet_Payload_Size_Bytes = 18
    key = b"key"
    digestmod = 'sha1'

    buffer = SlidingBook(num_pages=10, page_size=X.shape[0])
    page_processor = MACChecker(X = X, Y = Y, secret_key=key, digestmod=digestmod)

    udp = UDP_RX(IP= IP, PORT= PORT, buffer=buffer, page_processor= page_processor, Payload_Size_Byte=Packet_Payload_Size_Bytes)
    print(udp.receive())


    


