from .nD_MAC import MACGenerator, SlidingBook, Packet
import numpy as np
import socket
import time


# The UDP_TX class is responsible for transmitting data over the network using the UDP protocol.

class UDP_TX:
    def __init__(self, buffer: SlidingBook, page_processor:MACGenerator,   IP:str ='0.0.0.0', PORT:int = 23422, Payload_Size_Byte=128):
        self.IP = IP
        self.PORT = PORT
        self.Payload_Size_Byte = Payload_Size_Byte
        self.buffer = buffer
        self.page_processor = page_processor
    
    def segment_to_pages(self, input_bytes: bytes, payload_size: int = None, modification_attack = None) -> list:
        """
        Segment the input bytes into packets and store them in the SlidingBook.

        :param input_bytes: The byte stream to be segmented.
        :param packet_size: The maximum payload size for each packet.
        """
        if payload_size is None:
            payload_size = self.Payload_Size_Byte
        
        # print(f"payload_size: {payload_size}, len(input_bytes): {len(input_bytes)}, -(-len(input_bytes) // payload_size): {-(-len(input_bytes) // payload_size)}")
        num_packets = -(-len(input_bytes) // payload_size)  # Efficiently calculate the number of packets
        segments = [input_bytes[i*payload_size:(i+1)*payload_size] for i in range(num_packets)]
        packets = [Packet(SN=i, message=segment) for i, segment in enumerate(segments)]
        # Store packets in the book using map to avoid explicit loops
        res =  list(filter(None, map(self.buffer.add_packet, packets)))
        res += list(map(self.buffer.remove_page, self.buffer.get_page_index()))
        # res = res + self.buffer.remove_page(self.buffer.get_page_index())
        list(map(self.page_processor.process_page, res))

        return res

    
    def transmit(self, pages, attack_probability:float = 0 , delay:float = 0.00):
        """
        Transmit all packets stored in the SlidingBook over UDP.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # Use a nested list comprehension to send all packets
            packets_to_send = [packet for page in pages for packet in page.packets if packet is not None]

            # list(map(lambda data: sock.sendto(data, (self.IP, self.PORT)), data_to_send))
            for packet in packets_to_send:
                if np.random.random() > attack_probability:

                    sock.sendto(packet.to_bytes(), (self.IP, self.PORT))
                    # time.sleep(delay)
                else:
                    attack_message = b"ATTACK"
                    sock.sendto(packet.to_bytes()[0:4+8]+attack_message, (self.IP, self.PORT))
                    # time.sleep(delay)

                

            sock.sendto(b'END', (self.IP, self.PORT))
        del self.buffer
    
    def transmit_emulator(self, pages):
        """
        Transmit all packets stored in the SlidingBook over UDP.
        """
        # Use a nested list comprehension to send all packets
        data_to_send = [packet.to_bytes() for page in pages for packet in page.packets if packet is not None]
        # Send all packets at once using map
        list(map(lambda data: print(data), data_to_send))
        # Clear all pages after transmission
        # sock.sendto(b'END', (self.IP, self.PORT))
        del self.buffer

    




if __name__ == "__main__":
                    # t1  t2  t3  t4  t5  t6  t7  t8  t9
    X = np.array(  [[ 1,  0,  0,  0,  0,  0,  1,  0,  0], # m1
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
                    # t1  t2  t3  t4  t5  t6  t7  t8  t9
    Y = np.array(  [[ 0,  0,  0,  0,  0,  0,  1,  0,  0], # m1
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
    KEY = b"key"
    DIGESTMOD = 'sha1' 
    IP = "0.0.0.0"
    PORT = 23422


    buffer = SlidingBook(num_pages=10, page_size=X.shape[0])
    page_processor = MACGenerator(X = X, Y = Y, secret_key=KEY, digestmod=DIGESTMOD)
    udp = UDP_TX(IP= IP, PORT= PORT, buffer=buffer, page_processor= page_processor, Payload_Size_Byte=18)

    pages = udp.segment_to_pages(b'This test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrityThis test shows 2D integrity check is better than blockwise integrity.\nThis test shows 2D integrity check is better than blockwise integrity.\n')
    udp.transmit(pages)
    # udp.transmit_emulator(pages)

















