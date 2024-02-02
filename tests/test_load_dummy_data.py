import requests
from datetime import datetime
import random

sensor_url = 'http://localhost:5000/post-sensors'
output_url = 'http://localhost:5000/post-output'

sensor_data = {
    'Date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
    'OTemp': random.randint(0, 100),
    'TTemp': random.randint(0, 100),
    'Humidity': random.randint(0, 100),
    'Pressure': random.randint(0, 100),
    'AvgVoltage': random.randint(0, 100),
    'AvgCurrent': random.randint(0, 100),
    'B1Voltage': random.randint(0, 100),
    'B1Current': random.randint(0, 100),
    'B2Voltage': random.randint(0, 100),
    'B2Current': random.randint(0, 100),
    'B3Voltage': random.randint(0, 100),
    'B3Current': random.randint(0, 100),
    'X': random.randint(0, 100),
    'Y': random.randint(0, 100),
    'Z': random.randint(0, 100),
    'Pitch': random.randint(0, 100),
    'Roll': random.randint(0, 100),
    'Yaw': random.randint(0, 100),
}

output_data = {
    'Date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
    'M1': random.randint(0, 255),
    'M2': random.randint(0, 255),
    'M3': random.randint(0, 255),
    'M4': random.randint(0, 255),
    'M5': random.randint(0, 255),
    'M6': random.randint(0, 255),
    'M7': random.randint(0, 255),
    'M8': random.randint(0, 255),
    'Claw': random.randint(0, 255),
    'KS': random.randint(0, 255)
}

response = requests.post(sensor_url, json=sensor_data)
print(response.status_code)
print(response.content)

response = requests.post(output_url, json=output_data)
print(response.status_code)
print(response.content)