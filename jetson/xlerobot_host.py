#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import time

import cv2
import zmq

try:
    import smbus2 as _smbus2
    _imu_bus = _smbus2.SMBus(7)
    _imu_bus.write_byte_data(0x68, 0x6B, 0)
    IMU_OK = True
except Exception:
    IMU_OK = False

def _read_imu():
    if not IMU_OK:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        d = _imu_bus.read_i2c_block_data(0x68, 0x3B, 14)
        def s16(hi, lo):
            v = (hi << 8) | lo
            return (v - 65536) if v > 32767 else v
        ax = s16(d[0],  d[1])  / 16384.0
        ay = s16(d[2],  d[3])  / 16384.0
        az = s16(d[4],  d[5])  / 16384.0
        gx = s16(d[8],  d[9])  / 131.0
        gy = s16(d[10], d[11]) / 131.0
        gz = s16(d[12], d[13]) / 131.0
        return ax, ay, az, gx, gy, gz
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

from .xlerobot import XLerobot
from .config_xlerobot import XLerobotConfig, XLerobotHostConfig


class XLerobotHost:
    def __init__(self, config: XLerobotHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main():
    logging.info("Configuring Xlerobot")
    robot_config = XLerobotConfig(id=None, port1="/dev/ttyACM1", port2="/dev/ttyACM0")
    robot = XLerobot(robot_config)

    logging.info("Connecting Xlerobot")
    robot.connect()

    logging.info("Starting HostAgent")
    host_config = XLerobotHostConfig()
    host = XLerobotHost(host_config)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                if last_observation.get(cam_key) is None:
                    last_observation[cam_key] = ""
                    continue
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            ax, ay, az, gx, gy, gz = _read_imu()
            last_observation["imu_ax"] = ax
            last_observation["imu_ay"] = ay
            last_observation["imu_az"] = az
            last_observation["imu_gx"] = gx
            last_observation["imu_gy"] = gy
            last_observation["imu_gz"] = gz

            # Send the observation to the remote agent
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
