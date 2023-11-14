from multiprocessing import Process, Pipe
from datetime import datetime
import numpy as np
import logging
import socket
import cv2

from modules.Controller_Module import Controller_Module
from modules.Networking_Package import Networking_Package
from modules.Logger_Module import Logger_Module

class surface:
    def __init__(self):
        self.orin_ip = '192.168.1.192'
        self.orin_port = 9999

        self.CM_Parent, CM_Child = Pipe()
        self.NP_Parent, NP_Child = Pipe()

        self.CM = Process(target=self.run_Controller_Module, args=(CM_Child,))
        self.NP = Process(target=self.run_Networking_Package, args=(NP_Child,))

        self.logger = Logger_Module('Surface')

    def run_Controller_Module(self, pipe):
        CM = Controller_Module()
        self.logger.info("Controller Module started")
        while True:
            pipe.send(CM.get_data())

    def run_Networking_Package(self, pipe):
        NP = Networking_Package(socket.AF_INET, socket.SOCK_STREAM)
        NP.connect((self.orin_ip, self.orin_port)))
        self.logger.info("Networking Package started")
        while True:
            NP.send_string_as_bytes(pipe.recv())
            pipe.send(NP.recv())

    def prep_configs(self):
        pass

    def send_configs(self):
        pass

    def run(self):
        self.CM.start()
        self.NP.start()

        while True:
            controller_data = self.CM_Parent.recv()
            self.logger.info(f'Data sent: {controller_data}')
            self.NP_Parent.send(controller_data)

            sub_data = self.NP_Parent.recv()
            self.logger.info(f'Data received: {sub_data[0]}')
            cv2.imshow("Surface", sub_data[1])
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main = surface()
    main.run()
