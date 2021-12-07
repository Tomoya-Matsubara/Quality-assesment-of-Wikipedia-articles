import os


MAX_FILE_ID = 10000

data_dir = "clean_text" 
save_dir = "Cleaned"

for i in range(MAX_FILE_ID):                # Explore exhaustively
    file_name = data_dir + '/' + str(i + 1)
    save_name = save_dir + '/' + str(i + 1) 

    if os.path.isfile(file_name):        
        with open(file_name) as file:
            with open(save_name, mode='w') as save:
                for line in file:
                    if not line.isspace():
                        print(line, file=save, end="")
