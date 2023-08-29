import csv
from datetime import datetime
import time

def get_employee_id(name, id_file):
    """
    Get the employee ID from the CSV file based on the employee name.
    """
    with open(id_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                return row[1]
    return None

def store_attendance(name, employee_id, attendance_file):
    """
    Store the attendance entry in the CSV file.
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    time_int = datetime.now().time()
    time_abs = (time_int.hour * 3600) + (time_int.minute * 60) + (time_int.second)
    
    # Check for duplicate entry
    if check_duplicate(attendance_file, name, current_date, time_abs):
        print("Duplicate entry found. Skipping.")
        return



    # Store attendance entry
    with open(attendance_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, employee_id, current_date, current_time])
        print("Attendance entry stored successfully.")

def check_duplicate(file, name, cur_date, cur_time):
    """
    Check if a name already exists in the CSV file.
    """
    with open(file, mode='r') as file:
        reader = csv.reader(file)
        check = False
        for idx, row in enumerate(reader):
            print(idx)
            print(row)
            if idx>0:
                if row[0] == name:
                    if row [2] == cur_date:
                        if row [3] - cur_time < 300:
                            check = True
                        else :
                            check = False
                    
    return check

def get_time_int():
    time_int = datetime.now().time()
    time_abs = (time_int.hour * 3600) + (time_int.minute * 60) + (time_int.second)
    return time_abs




# Example usage
employee_name = "Shafiq"  # Replace with the employee name
id_csv_file = "employee_ids.csv"  # Replace with the employee IDs CSV file name
attendance_csv_file = "att.csv"  # Replace with the attendance CSV file name

employee_id = get_employee_id(employee_name, id_csv_file)

if employee_id:
    store_attendance(employee_name, employee_id, attendance_csv_file)
else:
    print("Employee ID not found for the given name.")



# current_time = datetime.now().time()
# time1 = (current_time.hour * 3600) + (current_time.minute * 60) + (current_time.second)
# time.sleep(10)
# current_time2 = datetime.now().time()
# time2 = (current_time2.hour * 3600) + (current_time2.minute * 60) + (current_time2.second)
# print (time2 - time1)