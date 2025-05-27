import serial
import time

PORT = "/dev/ttyACM1"  # Change this to match the STM32 USB CDC device
BAUD_RATE = 115200  # Required by PySerial but ignored for USB CDC
OUTPUT_FILE = "received_data.txt"

def receive_data():
    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=1) as ser, open(OUTPUT_FILE, "wb") as file:
            print(f"Listening on {PORT} ...")
            
            received_bytes = 0
            start_time = time.time()  # Start timing before first read

            while True:
                data = ser.read(4096)  # Read USB FS max packet size (64 bytes)
                
                if data:
                    file.write(data)
                    file.flush()
                    received_bytes += len(data)
                    print(f"Received {len(data)} bytes, Total: {received_bytes} bytes")
                else:
                    # No data received (timeout), exit the loop
                    if received_bytes > 0:
                        print("No more data received, exiting...")
                        break

            end_time = time.time()  # End timing after the transmission
            elapsed_time = end_time - start_time
            print(f"Transmission Completed. Total bytes received: {received_bytes}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Measured Speed: {received_bytes / elapsed_time:.2f} B/s")

    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    receive_data()


