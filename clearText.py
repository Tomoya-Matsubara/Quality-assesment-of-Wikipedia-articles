import os
import re

MAX_FILE_ID = 10000

data_dir = "HTML" 
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

unprocessed = []

for i in range(MAX_FILE_ID):                # Explore exhaustively
    file_name = data_dir + '/' + str(i + 1) + '.html'
    save_name = save_dir + '/' + str(i + 1) 

    if os.path.isfile(file_name):        
        with open(file_name) as f:
            text = f.read()
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"↩︎", "", text)
            with open(save_name, mode='w') as f:
                f.write(text)
    else:
        unprocessed.append(i+1)

with open("log.txt", mode='w') as f:
    print(unprocessed, file=f)

