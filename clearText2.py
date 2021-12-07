import os
import re

MAX_FILE_ID = 10000

log = "log.txt"

data_dir = "text9999" 
save_dir = "clean_text" 

def cleanData(data):
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.replace('*', '')
    data = data.replace('<', ' ')
    data = data.replace('>', ' ')
    data = data.replace('|', ' ')
    data = data.replace("'''", '')
    data = data.replace("''", '')
    data = data.replace('"', '')
    data = data.replace('{', '')
    data = data.replace('}', '')
    data = data.replace('===', '')
    data = data.replace('==', '')
    data = data.replace('(', '')
    data = data.replace(')', '')

    return data


with open(log, mode='r') as f:
    indices = f.read()
    indices = indices.replace("[", "")
    indices = indices.replace("]", "")
    indices = indices.replace("\n", "")
    indices = indices.split(", ")

    indices = [int(s) for s in indices]

# print(indices)
print(len(indices))

j = 0

for i in indices:                # Explore exhaustively
    file_name = data_dir + '/' + str(i)
    save_name = save_dir + '/' + str(i) 

    if os.path.isfile(file_name):        
        with open(file_name) as f:
            text = f.read()
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"↩︎", "", text)
            text = cleanData(text)
            with open(save_name, mode='w') as f:
                f.write(text)
        # print("Processed {}".format(file_name))
        j += 1
print(j)




