import serial
import csv
import time

#pip install pyserial

def read_serial_data(port,baud_rate,num_readings,output_file):
    try:
        #open serial connection
        with serial.Serial(port,baud_rate,timeout=1) as ser:
            print(f"Connected to {port} at {baud_rate} baud rate")
        
        #open csv file for writing
        with open(output_file,mode='w',newline='') as file:
            writer= csv.writer(file)
            writer.writerow(["Reading Number", "Date"])#write Header

            time.sleep(2)#Delay of 2 secs between readings
            #read specified number of data entries
            for i in range(num_readings):
                data=ser.readline().decode('utf-8').strip()

                if data:
                    print(f"Reading {i+1}:{data}")
                    writer.writerow([i+1,data])
                else:
                    print(f"Reading {i+1}: No data Recieved")
                    writer.writerow([i+1,"No data recieved"]) 
                time.sleep(2) #Delay of 2 secs between readings
    except serial.SerialException as e:
        print(f"Error : {e}")
    except Exception as e :
        print(f"Unexpected Error :{e}")


if __name__ == "__main__":
    #replace 'COMB  with your deviced port and set the appropriate  baud rate
    port='COM12'# Example :dev/ttyUSB0 for linux machines
    baud_rate=115200 #adjust as per your devices configuration
    num_readings=10 
    output_file='Serial_data.csv'   

    read_serial_data(port,baud_rate,num_readings,output_file)    