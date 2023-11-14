#!/usr/bin/env python3

from io import BytesIO
import logging
import socket
from typing import Any

import numpy as np


class Networking_Package(socket.socket):
    """
    ## Networking_Package class
    Inherits from `socket.socket` and provides specialized methods to send and receive numpy arrays and strings as bytes.
    """

    def sendall(self, frame: np.ndarray) -> None:
        """
        ## Send a numpy frame over the socket.
        @param frame: The numpy array frame to send.
        """
        out = self.__pack_frame(frame)
        super_socket = super()
        super_socket.sendall(out)
        logging.debug("frame sent")

    def send_string_as_bytes(self, string: str) -> None:
        """
        ## Send a string as a byte array over the socket.
        @param string: The string to send.
        """
        byte_array = string.encode('utf-8')
        super().sendall(byte_array)
        logging.debug(f"String '{string}' sent as bytes")

    def recv(self, bufsize: int = 1024) -> np.ndarray:
        """
        ## Receive a numpy frame over the socket.
        @param bufsize: The size of the buffer to use for receiving data. Defaults to 1024.
        @return: The received numpy array.
        """
        length = None
        frame_buffer = bytearray()
        while True:
            data = super().recv(bufsize)
            if len(data) == 0:
                return np.array([])
            frame_buffer += data

            while True:
                if length is None:
                    if b":" not in frame_buffer:
                        break
                    length_str, _, frame_buffer = frame_buffer.partition(b":")
                    length = int(length_str)

                if len(frame_buffer) < length:
                    break

                frame_data = frame_buffer[:length]
                frame_buffer = frame_buffer[length:]
                frame = np.load(BytesIO(frame_data), allow_pickle=True)["frame"]
                logging.debug("frame received")
                return frame

    def recv_string_as_bytes(self, bufsize: int = 1024) -> str:
        """
        ## Receive string data in the form of a byte array and returns the string.
        @param bufsize: The size of the buffer to use for receiving data. Defaults to 1024.
        @return: The received string.
        """
        data = super().recv(bufsize)
        if len(data) == 0:
            return ""
        decoded_string = data.decode('utf-8')
        logging.debug(f"Received string as bytes: {decoded_string}")
        return decoded_string

    def accept(self) -> tuple["Networking_Package", tuple[str, int] | tuple[Any, ...]]:
        """
        ## Accept a connection.
        Overrides the base class method to return an object of this class instead of `socket.socket`.
        @return: Tuple containing a new Networking_Package object and the address of the client.
        """
        super_socket = super()
        fd, addr = super_socket._accept()
        sock = Networking_Package(super_socket.family, super_socket.type, super_socket.proto, fileno=fd)

        if socket.getdefaulttimeout() is None and super_socket.gettimeout():
            sock.setblocking(True)
        return sock, addr

    @staticmethod
    def __pack_frame(frame: np.ndarray) -> bytearray:
        """
        ## Pack a numpy frame into a byte array with a header indicating its size.
        @param frame: The numpy array frame to pack.
        @return: The packed byte array.
        """
        f = BytesIO()
        np.savez(f, frame=frame)

        packet_size = len(f.getvalue())
        header = f"{packet_size}:"
        header_bytes = bytes(header.encode())

        out = bytearray(header_bytes)
        f.seek(0)
        out += f.read()

        return out
