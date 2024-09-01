import numpy as np
from Book import SlidingBook, Packet, Page
import socket
import hmac

# The UDP_TX class is responsible for transmitting data over the network using the UDP protocol.

class UDP_TX:
    def __init__(self, buffer: SlidingBook,  IP:str ='0.0.0.0', PORT:int = 23422, Payload_Size_Byte=128):
        self.IP = IP
        self.PORT = PORT
        self.Payload_Size_Byte = Payload_Size_Byte
        self.buffer = buffer
    
    def segment_to_pages(self, input_bytes: bytes, payload_size: int = None) -> list:
        """
        Segment the input bytes into packets and store them in the SlidingBook.

        :param input_bytes: The byte stream to be segmented.
        :param packet_size: The maximum payload size for each packet.
        """
        if payload_size is None:
            payload_size = self.Payload_Size_Byte
        
        num_packets = -(-len(input_bytes) // payload_size)  # Efficiently calculate the number of packets
        segments = [input_bytes[i*payload_size:(i+1)*payload_size] for i in range(num_packets)]
        packets = [Packet(SN=i, message=segment) for i, segment in enumerate(segments)]
        # Store packets in the book using map to avoid explicit loops
        return list(map(self.buffer.add_packet, packets))
    
    def transmit(self):
        """
        Transmit all packets stored in the SlidingBook over UDP.
        """
        pages = [self.book.pages[page_index] for page_index in self.book.get_page_index()]
        
        # Use a nested list comprehension to send all packets
        data_to_send = [packet.to_bytes() for page in pages for packet in page.packets if packet is not None]
        
        # Send all packets at once using map
        list(map(lambda data: self.sock.sendto(data, (self.dest_ip, self.dest_port)), data_to_send))
        
        # Clear all pages after transmission
        list(map(self.book.remove_page, self.book.get_page_index()))

    def close(self):
        """Close the UDP socket."""
        self.sock.close()
    

buffer = SlidingBook(num_pages=10, page_size=2)
udp = UDP_TX(buffer=buffer, Payload_Size_Byte=2)

print(udp.segment_to_pages(b'This is a test message'))
















# class UDP_TX:
#     def __init__(self, IP:str ='0.0.0.0', PORT:int = 23422, X = np.eye(10), Y = np.eye(10), Payload_Size_Byte=128, KEY=b"key", digestmod='sha384'):
#         self.IP = IP
#         self.PORT = PORT
#         self.X = X
#         self.Y = Y
#         self.Payload_Size_Byte = Payload_Size_Byte
#         self.KEY = KEY
#         self.digestmod = digestmod



#     def divide_Bytes_and_pad(self,Bytes, chunk_size_Byte=None):
#         if chunk_size_Byte is None:
#             chunk_size_Byte = self.chunk_size_Byte
#         chunks = [Bytes[i:i + chunk_size_Byte] for i in range(0, len(Bytes), chunk_size_Byte)]
#         return chunks

    
#     def chucks_to_nD_arrangments(self, chunks, chunk_size_Byte=None, X=None):
#         if chunk_size_Byte is None:
#             chunk_size_Byte = self.chunk_size_Byte
#         if X is None:
#             X = self.X

#         # Calculate the number of full pages
#         num_full_pages = len(chunks) // X.shape[0]
#         total_pages = num_full_pages + (1 if len(chunks) % X.shape[0] != 0 else 0)
        
#         # Pad the chunks to fit into a full last page if necessary
#         padding_size = total_pages * X.shape[0] - len(chunks)
#         if padding_size > 0:
#             chunks.extend([b''] * padding_size)

#         # Reshape into pages of X.shape[0] chunks each
#         nD_arrangements = np.array(chunks, dtype=object).reshape(total_pages, X.shape[0])
        
#         return nD_arrangements

#     def mac_for_page(self, page, key=None, X=None, Y=None):
#         key = key or self.KEY
#         X = X if X is not None else self.X
#         Y = Y if Y is not None else self.Y

#         # Initialize the result dictionary with the page content
#         res = {i: page[i] for i in range(X.shape[0])}

#         # Transpose X and Y for more efficient column operations
#         X_transposed = X.T
#         Y_transposed = Y.T

#         # Iterate over tag indices
#         for tag_index in range(X.shape[1]):
#             selected_blocks_indices = np.where(X_transposed[tag_index] == 1)[0]
#             if len(selected_blocks_indices) > 0:
#                 # Join the selected blocks' data
#                 data = b''.join(page[i] for i in selected_blocks_indices)
#                 target_index = np.where(Y_transposed[tag_index] == 1)[0][0]
#                 res[target_index] = page[target_index] + hmac.new(key, data, digestmod=self.digestmod).digest()

#         return res

#     # SN: Sequence Number (4Bytes)
#     # time: time          (8Bytes)
#     # msg: message        (chunk_size_Byte Bytes)  
#     def send_msg(self,SN, msg, sock, dest):
#         sock.sendto(SN.to_bytes(4, 'big') + struct.pack('d', time.time()) + msg, dest)
#         time.sleep(0.000001)


#     def transmit(self, Bytes, attack=None):
#         if attack is None:
#             attack = []

#         SN = 0
#         cnt = 0
#         with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
#             # Divide Bytes into chunks and arrange them into pages
#             chunks = self.divide_Bytes_and_pad(Bytes, chunk_size_Byte=self.chunk_size_Byte)
#             pages = self.chucks_to_nD_arrangments(chunks, chunk_size_Byte=self.chunk_size_Byte, X=self.X)

#             # Flatten the pages and their corresponding messages
#             for page in pages:
#                 mac_page = self.mac_for_page(page=page, key=self.KEY, X=self.X, Y=self.Y)
#                 for msg_index, msg in mac_page.items():
#                     if SN in attack:
#                         SN += 1
#                         continue
#                     self.send_msg(SN, msg, sock, (self.IP, self.PORT))
#                     cnt += 1
#                     SN += 1
            
#             # Send the end message (for testing purposes)
#             sock.sendto(b'END', (self.IP, self.PORT))

#         return cnt

# # unit test

# if __name__ == "__main__":

#     #### parameters that needs to be exhanged between the sender and the receiver #####
#     IP = "0.0.0.0"
#     PORT = 23422
#             #  t1  t2  t3  t4  t5  t6  t7  t8  t9
#     X = np.array([[ 1,  0,  0,  0,  0,  0,  1,  0,  0], # m1
#                 [ 1,  0,  0,  0,  0,  0,  0,  1,  0], # m2
#                 [ 1,  0,  0,  0,  0,  0,  0,  0,  1], # m3
#                 [ 0,  1,  0,  0,  0,  0,  1,  0,  0], # m4
#                 [ 0,  1,  0,  0,  0,  0,  0,  1,  0], # m5
#                 [ 0,  1,  0,  0,  0,  0,  0,  0,  1], # m6
#                 [ 0,  0,  1,  0,  0,  0,  1,  0,  0], # m7
#                 [ 0,  0,  1,  0,  0,  0,  0,  1,  0], # m8
#                 [ 0,  0,  1,  0,  0,  0,  0,  0,  1], # m9
#                 [ 0,  0,  0,  1,  0,  0,  1,  0,  0], # m10
#                 [ 0,  0,  0,  1,  0,  0,  0,  1,  0], # m11
#                 [ 0,  0,  0,  1,  0,  0,  0,  0,  1], # m12
#                 [ 0,  0,  0,  0,  1,  0,  1,  0,  0], # m13
#                 [ 0,  0,  0,  0,  1,  0,  0,  1,  0], # m14
#                 [ 0,  0,  0,  0,  1,  0,  0,  0,  1], # m15
#                 [ 0,  0,  0,  0,  0,  1,  1,  0,  0], # m16
#                 [ 0,  0,  0,  0,  0,  1,  0,  1,  0], # m17
#                 [ 0,  0,  0,  0,  0,  1,  0,  0,  1]]) # m18
#                 #  t1  t2  t3  t4  t5  t6  t7  t8  t9
#     Y = np.array([[ 0,  0,  0,  0,  0,  0,  1,  0,  0], # m1
#                 [ 0,  0,  0,  0,  0,  0,  0,  1,  0], # m2
#                 [ 1,  0,  0,  0,  0,  0,  0,  0,  0], # m3
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m4
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m5
#                 [ 0,  1,  0,  0,  0,  0,  0,  0,  0], # m6
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m7
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m8
#                 [ 0,  0,  1,  0,  0,  0,  0,  0,  0], # m9
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m10
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m11
#                 [ 0,  0,  0,  1,  0,  0,  0,  0,  0], # m12
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m13
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m14
#                 [ 0,  0,  0,  0,  1,  0,  0,  0,  0], # m15
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # m16
#                 [ 0,  0,  0,  0,  0,  1,  0,  0,  0], # m17
#                 [ 0,  0,  0,  0,  0,  0,  0,  0,  1]]) # m18
#     chunk_size_Byte = 2 
#     key = b"key"
#     digestmod = 'sha384'

#     upd_tx = UDP_TX(IP, PORT, X, Y,  chunk_size_Byte = chunk_size_Byte, KEY=key, digestmod = digestmod)

#     data = b'This test shows 2D integrity check is better than blockwise integrity.'

#     # testing the divide_Bytes_and_pad function

    # print("Dividing the Byte data every to chunk_size_Bytes and pad the last chunck if needed",upd_tx.divide_Bytes_and_pad(data, chunk_size_Byte=chunk_size_Byte))

    # # testing the chucks_to_nD_arrangments function witht X.shape[0] and chunk_size
    # chunks = upd_tx.divide_Bytes_and_pad(data, chunk_size_Byte=chunk_size_Byte)
    # print(upd_tx.chucks_to_nD_arrangments(chunks, chunk_size_Byte, X))

    # # testing the mac_for_page function
    # pages = upd_tx.chucks_to_nD_arrangments(chunks, chunk_size_Byte, X)
    # pprint.pprint(upd_tx.mac_for_page(pages[0], key, X, Y))

    # # Emulating the transmitted Packets
    # simulation = upd_tx.mac_for_page(pages[0], key, X, Y)
    # SN = 0
    # for msg in simulation:
    #     simulation[msg] = SN.to_bytes(4, 'big') + struct.pack('d', time.time()) +  simulation[msg]
    #     SN += 1
    # pprint.pprint(simulation)
    # # at this point after calculating the MAC for each page, we can send the data
    # # iterate through the chunks (msg) in every page and send it to the destination
    # # the function send_msg is responsible for sending the data to the destination
    # # the function transmit input bytes and create pages with padded chunks  and 
    # # then iterates through the all the chunks in the pages and send them to the destination

    # # testing the transmit function
    # # print("Number of messages sent: ", upd_tx.transmit(data))
    # # testing the 


    # print("Number of messages sent under attack: ", upd_tx.transmit(data, attack=[1]))




