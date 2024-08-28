# %%
import socket
import hmac
import numpy as np

ip = '0.0.0.0'
port = 23422

# %%

def data_parser(data, X, Y):
    chunk_index = int.from_bytes(data[:4], 'big')
    if(np.sum(Y[chunk_index%X.shape[0]-1])):
        chunk_data = data[4:-48]
        mac = data[-48:]
    else:
        chunk_data = data[4:]
        mac = b''
    return chunk_index, chunk_data, mac

def fill_buffer_mac(chunk_index, chunk_data, mac, X, Y):
    global Buffer, macs
    Buffer[chunk_index%X.shape[0]-1] = chunk_data
    try:
        macs[np.where(Y[chunk_index%X.shape[0]-1] == 1)[0][0]] = mac
    except:
        pass
def reset_buffer_mac():
    global Buffer, macs, X
    Buffer = [b'']*X.shape[0]
    macs = [b'']*X.shape[1]

def check_mac_for_buffer(Buffer, X, Y, macs, excepted_chunk_index):
    global res, total_Verified_Data_tag_Length
    Buffer = np.array(Buffer)
    Buffer_verified = [b'*'*2]*X.shape[0]

    for tags in range(X.shape[1]):
        data = b''.join(Buffer[np.where(X[:,tags] == 1)])
        mac = hmac.new(b'key', data, 'sha384').digest()
        if(mac == macs[tags]):
            for x in np.where(X[:,tags] == 1)[0]:
                Buffer_verified[x] = Buffer[x] 
        else:
            pass
            # print("Data is corrupted")
        cnt = (excepted_chunk_index-1) - (excepted_chunk_index-1)%X.shape[0]
        for x in Buffer_verified:
            res[cnt] = x
            # total_Verified_Data_tag_Length += len(macs[tags])
            cnt += 1
        

    reset_buffer_mac()

def receive_data(sock, X, Y):
    global total_MAC_Length, total_Data_Length, total_Verified_Data_Length 
    global res, excepted_chunk_index
    global Buffer, macs
    while True:
        data, addr = sock.recvfrom(4096)  # 8 bytes for index + 1024 bytes of data
        if data == b'END':
            # print("End of transmission\n")

            received_data = b''.join([res[x] for x in res])
            print('\033[1m'+ received_data.decode() + '\033[0m')
            for x in res:
                total_Data_Length += len(res[x])
                if b'*' not in res[x]:
                    total_Verified_Data_Length += len(res[x])
            print(f"Goodput: {np.round(total_Verified_Data_Length/(total_Data_Length+total_MAC_Length),5)}")
            print("")

            # print("\n*******************************")



            # print(f"\nStats --------------\n\nTotal tag Length: {total_MAC_Length*8}")

            # print(f"Total Data Length: {total_Data_Length*8}")

            # print(f"Total Verified Data Length: {total_Verified_Data_Length*8}")
            # print(f"Goodput: {np.round(total_Verified_Data_Length/(total_Data_Length+total_MAC_Length),2)}")
            # print(f"Average tag bits per message: {total_Verified_Data_tag_Length*8/len(res)}")
            sock.close()
            break
        
        # if there is a tag with the message
        chunk_index, chunk_data, mac = data_parser(data, X, Y)

        total_MAC_Length += len(mac)

        # if the chunk index is good fill the buffer
        # else check is the time to verify the macs
        if excepted_chunk_index == chunk_index:
            fill_buffer_mac(chunk_index, chunk_data, mac, X, Y)
            excepted_chunk_index += 1
            
        else:
            left_to_verify = X.shape[0] - excepted_chunk_index%X.shape[0]
            # if the chunk index is too far from the excepted chunk index
            if excepted_chunk_index%X.shape[0] == 0 or ((chunk_index - excepted_chunk_index) > left_to_verify):
                check_mac_for_buffer(Buffer, X, Y, macs, excepted_chunk_index)

            fill_buffer_mac(chunk_index, chunk_data, mac, X, Y)
            excepted_chunk_index = chunk_index + 1
        
        # decide if it is the time to check the MACs
        if chunk_index%X.shape[0] == 0:
            check_mac_for_buffer(Buffer, X, Y, macs, excepted_chunk_index)






total_MAC_Length = 0
total_Data_Length = 0
total_Verified_Data_Length = 0
total_Verified_Data_tag_Length = 0
res = {}
excepted_chunk_index = 1

X = np.array([[1],
              [1],
              [1]]) # tag generation matrix

Y = np.array([[0],
              [0],
              [1]]) # tag assignment matrix

Buffer = [b'']*X.shape[0]
macs = [b'']*X.shape[1]


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))
# print(f"Listening on port {port} ...")
text = """This test shows  2D integrity check is better than  blockwise integrity."""
print(f"Expected text:\n\t\033[1m{text}\033[0m")

print("\n------------------------Receiving in the 1D mode-------------------------\n")

receive_data(sock, X, Y)
# %%

total_MAC_Length = 0
total_Data_Length = 0
total_Verified_Data_Length = 0
total_Verified_Data_tag_Length = 0
res = {}
excepted_chunk_index = 1

X = np.array([[1,0],
              [0,1]]) # tag generation matrix
Y = np.array([[1,0],
              [0,1]]) # tag assignment matrix
Buffer = [b'']*X.shape[0]
macs = [b'']*X.shape[1]


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))
# print(f"Listening on port {port} ...")

print("------------------------Receiving in the blockwise mode------------------------\n")

receive_data(sock, X, Y)



# %%
total_MAC_Length = 0
total_Data_Length = 0
total_Verified_Data_Length = 0
total_Verified_Data_tag_Length = 0
res = {}
excepted_chunk_index = 1


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
Buffer = [b'']*X.shape[0]
macs = [b'']*X.shape[1]


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))
# print(f"Listening on port {port} ...\n")
# pritn in red in terminal


print("------------------------Receiving in the 2D mode------------------------\n")
receive_data(sock, X, Y)



