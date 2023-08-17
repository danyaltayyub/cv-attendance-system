import csv
import datetime
import time


def initiate():
    header = ['Name', 'ID', 'Date', 'Time in']
    f = open('att.csv', "w+")
    writer = csv.writer(f)
    writer.writerow(header)
    return f


# def get_datetime():
#     dt = datetime.timedelta()
def conv_time(t):
    return str(t.tm_hour) + ":" + str(t.tm_min)+ ":" + str(t.tm_sec)

def search_entry(entry , file):
    reader = csv.reader(file)
    found = False
    for line in reader:      #Iterates through the rows of your csv
        if entry in line:      #If the string you want to search is in the row
            found = True
        break
    return found

def record(name, file):
    row = [name , 'id', str(datetime.date.today()), conv_time(time.localtime())]
    # f = open('att.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(row)

def close_file(fil):
    fil.close()
# record('Danyal')
