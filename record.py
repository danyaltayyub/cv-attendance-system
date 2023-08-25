import csv
from datetime import datetime

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
    
    # Check for duplicate entry
    if check_duplicate(attendance_file, name):
        print("Duplicate entry found. Skipping.")
        return
    
    # Store attendance entry
    with open(attendance_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, employee_id, current_date, current_time])
        print("Attendance entry stored successfully.")

def check_duplicate(file, name):
    """
    Check if a name already exists in the CSV file.
    """
    with open(file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == name:
                return True
    return False

# Example usage
employee_name = "John Doe"  # Replace with the employee name
id_csv_file = "employee_ids.csv"  # Replace with the employee IDs CSV file name
attendance_csv_file = "att.csv"  # Replace with the attendance CSV file name

employee_id = get_employee_id(employee_name, id_csv_file)

if employee_id:
    store_attendance(employee_name, employee_id, attendance_csv_file)
else:
    print("Employee ID not found for the given name.")
