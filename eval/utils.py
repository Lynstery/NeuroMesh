import os
import sys
import socket
import struct
import time
import pickle
import lzma
from collections import deque
from threading import Lock

def connect_with_retry(sock, host, port, delay=0.5):
    while True:
        try:
            sock.connect((host, port))
            return sock
        except (socket.timeout, socket.error) as e:
            time.sleep(delay)

def recv_msg(sock):
    fmt = 'ii'
    meta_info = recv_exact(sock, struct.calcsize(fmt))
    msg_type, data_size = struct.unpack(fmt, meta_info)
    data = recv_exact(sock, data_size)
    #data = pickle.loads(lzma.decompress(data))
    data = pickle.loads(data)
    return msg_type, data, data_size

def send_msg(sock, msg_type, data):
    fmt = 'ii'
    #data = lzma.compress(pickle.dumps(data))
    data = pickle.dumps(data)
    data_size = len(data)
    meta_info = struct.pack(fmt, msg_type, data_size)
    sock.sendall(meta_info)
    sock.sendall(data)

def recv_exact(sock, target_size, buf_size=4096):
    data = b'' 
    recv_size = 0
    while recv_size < target_size:
        chunk = sock.recv(min(buf_size, target_size - recv_size)) 
        if not chunk:
            raise RuntimeError("Connection closed before receiving the requested data.")
        data += chunk
        recv_size += len(chunk)
    return data

def check_work_dir():
    '''
    Switch working directory to directory where py file (to be run) is located
    Calling check_work_dir function at beginning of py file running as main function
      corrects all relative paths in it
    Function is invalid if py file is called as a module by another main function
    In this case you should use absolute path 
      (configure project directory with command ./configure.sh 1)
    '''
    curr_dir = os.getcwd()
    file_path = sys.argv[0]
    # file_name = os.path.basename(file_path)
    # file_name = file_path.split('/')[-1]
    if file_path[0] == '/':
        work_dir = file_path
    else:
        if file_path[0] == '.' and file_path[1] == '/':
            work_dir = os.path.join(curr_dir, file_path[1:])
            # work_dir = curr_dir+file_path[1:]
        else:
            work_dir = os.path.join(curr_dir, file_path)
            # work_dir = curr_dir+'/'+file_path
    work_dir = os.path.dirname(work_dir)
    # work_dir = work_dir[:-(len(file_name))]
    if os.path.exists(work_dir):
        os.chdir(work_dir)


class MovingWindow:
    def __init__(self, time_window=None, max_length=None):
        self.time_window = time_window
        self.store = deque(maxlen=max_length)
        self.lock = Lock()

    def update(self, x):
        self.store.append((x, time.time()))

    def reset(self):
        with self.lock:
            self.store.clear()

    def _start_timeout_loop(self):
        if self.time_window is None:
            return

        while True:
            time.sleep(self.time_window / 10)
            with self.lock:
                t = time.time()
                while len(self.store) > 0 and t - self.store[0][1] > self.time_window:
                    self.store.popleft()

class MovingSum(MovingWindow):
    def __init__(self, window_length, timeout=None):
        super().__init__(time_window=timeout, max_length=window_length)

    def value(self):
        with self.lock:
            return sum([x for x, _ in self.store])

class MovingAvg(MovingWindow):
    def __init__(self, window_length, timeout=None):
        super().__init__(time_window=timeout, max_length=window_length)

    def value(self):
        with self.lock:
            if len(self.store) == 0:
                return 0
            return sum([x for x, _ in self.store]) / len(self.store)