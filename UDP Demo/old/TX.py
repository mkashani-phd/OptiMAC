# %%
import socket
import time
import numpy as np
import hmac
import time

# %%

Number_of_attacks = 2
ip = '0.0.0.0'
port = 23422



def divide_text(text, block_size_bits = 16, X= None):
    text = text.encode('utf-8') 
    blocks = np.array([])
    for i in range(0, len(text), block_size_bits//8):
       blocks = np.append(blocks,text[i:i+block_size_bits//8])
    #append b'' to block make the len of blocks to be a multiple of xsahpe
    if X.shape[0] is not None and len(blocks)%X.shape[0]!=0:
        for i in range(X.shape[0] - len(blocks)%X.shape[0]):
            blocks = np.append(blocks,b'')
    return blocks

def mac_for_block(blocks:np.array, key:str, X:np.array, Y:np.array):
    res = {}
    for msg in range(X.shape[0]):
        res[msg] = blocks[msg].tobytes()
    for tags in range(X.shape[1]):
        data = b''.join(blocks[np.where(X[:,tags] == 1)]) 
        res[np.where(Y[:,tags] == 1)[0][0]] =  blocks[np.where(Y[:,tags] == 1)][0] + hmac.new(key, data, digestmod='sha384').digest()
    return res

def send_msg(FN, msg, sock, dest):
    sock.sendto(FN.to_bytes(4, 'big')+ msg, dest)
    # print(FN.to_bytes(4, 'big')+ msg)


def tx(text, sock, ip, port, X, Y, block_size_bits = 128, attack = []):
    blocks= divide_text(text, block_size_bits=block_size_bits, X = X)
    FN = 1
    for i in range(len(blocks)//X.shape[0]):
        block = mac_for_block(blocks[i*X.shape[0]:i*X.shape[0]+X.shape[0]], key = b"key", X = X, Y = Y)
        for msg in block.values():
            if FN in attack:
                # print("Dropped")
                FN += 1
                continue
            else:
                send_msg(FN, msg, sock, (ip,port) )
                FN += 1 
                time.sleep(0.01)

    sock.sendto(b'END', (ip,port))
    sock.close()
    time.sleep(1)

# mac_for_block(divide_text(text), b"key")



# %%


# randomly selecting a message of 18 messages to drop

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



text = """This test shows  2D integrity check is better than  blockwise integrity."""
attack = np.random.randint(1,36,Number_of_attacks)
# print bold the text
print(f'\nTransmitting text:\n\n\t\033[1m{text}\033[0m\n\n\nAttacking on packet numbers {sorted(list(attack.tolist()))}')
fir= min(attack[0],attack[1])-1
sec = max(attack[1],attack[0])-1
modified_text =  text[0:fir*2]  + '\033[1m\033[31m'+text[fir*2:fir*2+2]+ '\033[0m'  + text[fir*2+2:sec*2]   + '\033[1m\033[31m'+text[sec*2:sec*2+2]+'\033[0m'  +  text[sec*2+2:]
print(f'Modified text:\n\n\t{modified_text}\n\n')




X = np.array([[1],
              [1],
              [1]]) # tag generation matrix

Y = np.array([[0],
              [0],
              [1]]) # tag assignment matrix

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


print('transmitting in 1D mode ...')
tx(text, sock, ip, port, X, Y,block_size_bits = 16, attack = attack)




X = np.array([[1,0],
              [0,1]]) # tag generation matrix

Y = np.array([[1,0],
              [0,1]]) # tag assignment matrix

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


print('transmitting in blockwise tag mode ...')
tx(text, sock, ip, port, X, Y,block_size_bits = 16, attack = attack)



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


print('transmitting in 2D mode ...\n')
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tx(text, sock, ip, port, X, Y,block_size_bits = 16, attack = attack)




