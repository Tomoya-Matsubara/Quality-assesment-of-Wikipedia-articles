# Load necessary packages
import csv
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

DATA_SIZE = 9999 # Total size of data that will be used for training and validation

classes = {"stub":1, "start":2, "c":3, "b":4, "ga":5, "fa":6} # The classes that will be considered in the classification

# Function to return the key of a dictionary corresponding to a value
def get_key(my_dict,val):
    for key, value in my_dict.items():
         if val == value:
             return key

# Load the features of the dataset
features = []
with open('L:\\Documents\\Universidade\\CentraleSupélec\\Matières\\2A\\ST7\\Final Project\\Test area\\features_and_BERT.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        features.append([float(f) for f in row])
        if len(features) == DATA_SIZE:
          break

# Load the labels of the dataset
l = open('L:\\Documents\\Universidade\\CentraleSupélec\\Matières\\2A\\ST7\\Final Project\\quality-assesment-of-wikipedia-articles\\enwikilabel9k', "r")
string_labels = l.read().splitlines()
labels = [classes[label] for label in string_labels]
l.close()
labels = labels[:DATA_SIZE]

# Normalize the data in order to apply PCA
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)

# Apply PCA to take the 95% most significant components
pca_095 = PCA(n_components = 0.95, random_state = 2020)
pca_095.fit(scaled_features)
features_pca_095 = pca_095.transform(scaled_features)
print("Number of components selected by PCA =",len(features_pca_095[0]),"=",100*len(features_pca_095[0])/len(features[0]),"% of the total number of features.")

# Separate the dataset
TRAINING_SIZE = int(0.8*DATA_SIZE)
training_data = (features_pca_095[0:TRAINING_SIZE],labels[0:TRAINING_SIZE])
validation_data = (features_pca_095[TRAINING_SIZE:],labels[TRAINING_SIZE:])
VALIDATION_SIZE = len(validation_data[0])
print("Size of the training dataset =",len(training_data[0]))
print("Size of the validation dataset =",len(validation_data[0]))

# Create the FFNN classifier
clf = MLPClassifier(solver='adam', alpha=0.002, hidden_layer_sizes=(512,1024,256), random_state=1)

# Train the FFNN classifier
clf.fit(training_data[0], training_data[1])

# Apply the model to the validation dataset
n_rights = 0
n_wrongs = 0
result = []
for i in range(VALIDATION_SIZE):
  r = clf.predict([validation_data[0][i]])
  result.append(r)
  if r == validation_data[1][i]:
    n_rights+=1
  else:
    # Uncomment the following line to plot the cases where the algorithm is mistaken
    #print("Prediction: "+get_key(classes,r[0])+" ; Label: "+get_key(classes,validation_data[1][i]))
    n_wrongs+=1

# Calculate and print the accuracy of the architecture
print("Number of right predictions =",n_rights)
print("Number of wrong predictions =",n_wrongs)
print("Accuracy (%) =",100*n_rights/(n_rights+n_wrongs))

# Uncomment the following line to save the FFNN model
#dump(clf, 'L:\\Documents\\Universidade\\CentraleSupélec\\Matières\\2A\\ST7\\Final Project\\quality-assesment-of-wikipedia-articles\\model.joblib') 