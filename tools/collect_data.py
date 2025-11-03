#!/usr/bin/env python3
"""
collect_data.py

Connects to serial (USB-serial from V5 via PROS serial prints) and logs CSV rows.
Usage: python tools/collect_data.py /dev/ttyUSB0 115200 out.csv

Robot should print CSV lines like:
feat1,feat2,...,label1,label2
"""
import sys
import serial
import time

def main():
    if len(sys.argv) < 4:
        print("Usage: collect_data.py <serial_port> <baud> <out.csv>")
        return
    port = sys.argv[1]; baud = int(sys.argv[2]); out = sys.argv[3]
    ser = serial.Serial(port, baud, timeout=1)
    with open(out, "w") as f:
        print("Logging to", out)
        try:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                # basic filtering
                if any(ch.isalpha() for ch in line): 
                    # skip logs with letters unless CSV
                    pass
                print(line)
                f.write(line + "\n")
                f.flush()
        except KeyboardInterrupt:
            print("Stopped")
            ser.close()

if __name__ == "__main__":
    main()
