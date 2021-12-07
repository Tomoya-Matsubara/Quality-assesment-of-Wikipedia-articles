from sklearn.neural_network import MLPClassifier
import csv
from joblib import dump, load

DATA_SIZE = 3000

features = []

with open('L:\\Documents\\Universidade\\CentraleSupélec\\Matières\\2A\\ST7\\Final Project\\quality-assesment-of-wikipedia-articles\\features_and_BERT.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        features.append([float(f) for f in row])
        if len(features) == DATA_SIZE:
          break

classes = {"stub":1, "start":2, "c":3, "b":4, "ga":5, "fa":6}
# function to return key for any value
def get_key(my_dict,val):
    for key, value in my_dict.items():
         if val == value:
             return key

l = open('L:\\Documents\\Universidade\\CentraleSupélec\\Matières\\2A\\ST7\\Final Project\\quality-assesment-of-wikipedia-articles\\enwikilabel3k', "r")
string_labels = l.read().splitlines()
labels = [classes[label] for label in string_labels]
l.close()

labels = labels[:DATA_SIZE]

TRAINING_SIZE = int(0.8*DATA_SIZE)

training_data = (features[0:TRAINING_SIZE],labels[0:TRAINING_SIZE])
validation_data = (features[TRAINING_SIZE:],labels[TRAINING_SIZE:])

VALIDATION_SIZE = len(validation_data[0])

#print(len(training_data[0]))
#print(len(validating_data[0]))

clf = MLPClassifier(solver='adam', alpha=0.002, hidden_layer_sizes=(512,1024,256), random_state=1)

clf.fit(training_data[0], training_data[1])

n_rights = 0
n_wrongs = 0

result = []
for i in range(VALIDATION_SIZE):
  r = clf.predict([validation_data[0][i]])
  result.append(r)
  if r == validation_data[1][i]:
    n_rights+=1
  else:
    print("prediction: "+get_key(classes,r[0])+" ; label: "+get_key(classes,validation_data[1][i]))
    n_wrongs+=1

print(n_rights)
print(n_wrongs)

print(100*n_rights/(n_rights+n_wrongs))

dump(clf, 'L:\\Documents\\Universidade\\CentraleSupélec\\Matières\\2A\\ST7\\Final Project\\quality-assesment-of-wikipedia-articles\\model.joblib') 