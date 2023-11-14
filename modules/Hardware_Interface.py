import serial
import numpy as np 

class Hardware_Interface:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM0', 115200, 0.1) 

        self.starting_bit = 'a'
        self.ending_bit = '\n'
        
        self.recv_data = None
    
    def recv(self):
        pass

    def transmit(self, data):
        pass

    def close(self):
        pass

    def run(self):
        pass

if __name__ == "__main__":
    HI = Hardware_Interface()
    HI.run()
