import csv
import os

features = []
with open('features.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    features = list(csv_reader)
 
n = 0
MAX_FILE_ID = 10000
data_dir = "BERT3"

with open("features_and_BERT3.csv", 'w', newline="") as file:
    writer = csv.writer(file)
    for i in range(MAX_FILE_ID): # Explore exhaustively
        file_name = data_dir + '/' + str(i + 1)
        n += 1
        file = open(file_name, "r")
        BERT = file.read()
        BERT = BERT.split(", ")
        BERT = [float(s) for s in BERT]
        writer.writerow(features[n][:10] + BERT)