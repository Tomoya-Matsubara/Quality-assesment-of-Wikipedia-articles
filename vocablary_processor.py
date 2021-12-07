from tensorflow.keras.preprocessing import text, sequence
import numpy as np

data_dir = "text"   # directory contains text documents
model_size = 11     # length of output vectors

num_words = 2000    # Maximum number of words
oov_token = '<UNK>' # Encoding for unknown words
pad_type = 'post'
trunc_type = 'pre'

maxlen = 2000       # Maximum number of vector dimension


file_name = "text1/new1"
with open(file_name) as f:
    l = f.readlines()
    data = ".".join(l)

f.close()



print(len(data))
# data = data.split(".")
# print(data)
data = [data]


# X_train = [
#   "I enjoy coffee.",
#   "I enjoy tea.",
#   "I dislike milk.",
#   "I am going to the supermarket later this morning for some coffee."
# ]

X_train = data


# X_train = [s.replace(" ", "").replace("", " ") for s in X_train]

print("\033[31m" + "== Raw Training Data ==")
print(X_train)
print("\033[0m")



# vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length=model_size, min_frequency=0)
# X_train = np.array(list(vocab_processor.fit_transform(X_train)))
# X_test = np.array(list(vocab_processor.fit_transform(X_test)))

# X_train = pad_sequences(X_train, maxlen=model_size, value=0.)
# X_test = pad_sequences(X_test, maxlen=model_size, value=0.)

# n_words = len(vocab_processor.vocabulary_)
# print('Total words: %d' % n_words)

tokenizer = text.Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

print("\033[32m" + "== Word Index ==")
print(word_index)
print("\033[0m")

# Y_train = np.array(list(tokenizer.fit_on_texts(X_train)))
# print(Y_train)

X_train = tokenizer.texts_to_sequences(X_train)
print("\033[33m" + "== Training Data Sequences ==")
print(X_train)
print("\033[0m")

# maxlen = max([len(x) for x in X_train])

# maxlen = max(maxlen, 2000)


X_train = sequence.pad_sequences(X_train, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
print("\033[34m" + "== Padded Training Sequences ==")
print(X_train)
print("\033[0m")

# X_test = tokenizer.texts_to_sequences(X_test)
# X_test = sequence.pad_sequences(X_test, maxlen=model_size)