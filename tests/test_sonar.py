#!/usr/bin/env python3

# Import necessary modules
from brping import Ping360
import argparse
import time

parser = argparse.ArgumentParser(description="Ping python library example.")
parser.add_argument('--device', action="store", required=False, type=str, help="Ping device port. E.g: /dev/ttyUSB0")
parser.add_argument('--baudrate', action="store", type=int, default=115200, help="Ping device baudrate. E.g: 115200")
parser.add_argument('--udp', action="store", required=False, type=str, help="Ping UDP server. E.g: 192.168.2.2:9092")
args = parser.parse_args()
if args.device is None and args.udp is None:
    parser.print_help()
    exit(1)

p = Ping360()
if args.device is not None:
    p.connect_serial(args.device, args.baudrate)
elif args.udp is not None:
    (host, port) = args.udp.split(':')
    p.connect_udp(host, int(port))

print("Initialized: %s" % p.initialize())

p.set_transmit_frequency(800)
p.set_sample_period(80)
p.set_number_of_samples(200)
p.set_gain_setting(3)

tstart_s = time.time()
for x in range(400):
    p.transmitAngle(x)
tend_s = time.time()

print(p._data)

import matplotlib.pyplot as plt
import numpy as np

data = p._data

# Normalize data (optional, depending on your analysis needs)
normalized_data = [x / 255 for x in data]  # Normalize to range 0-1

# Visualization (simple line plot for demonstration)
plt.plot(normalized_data)
plt.xlabel('Sample')
plt.ylabel('Echo Strength')
plt.title('Sonar Echo Strength')
plt.show()

print("full scan in %dms, %dHz" % (1000*(tend_s - tstart_s), 400/(tend_s - tstart_s)))

p.control_reset(0, 0)