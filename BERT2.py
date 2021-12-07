import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

BERT_LIMIT = 512

# Open a file
file_name = "text/1"

def BERT(file_name):
  with open(file_name) as f:
      l = f.readlines()     # Read each line
      text = ".".join(l)    # '\n' is replaced by '.'

  f.close()

  # Simple example
  # text = "This is a pen."

  # Show the loaded text
  # print("\033[31m"); print(text); print("\033[0m")

  # Add the start token
  marked_text = "[CLS] " + text

  # Tokenize the text (break down sentences into words)
  tokenized_text = tokenizer.tokenize(marked_text)


  tokenized_text_split =[]
  segments_ids = []
  segment = 1

  # Add the end token at each end of sentences
  # and make flags to differentiate 2 sentences
  for i in tokenized_text:
    tokenized_text_split.append(i) 
    segments_ids.append(segment)

    if i in {"."}:
      tokenized_text_split.append("[SEP]")
      segments_ids.append(segment)

      if segment == 1: 
        segment = 0
      else:
        segment = 1

  # print("\033[32m"); print(tokenized_text_split); print("\033[0m")

  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_split)

  # If you want to see the correspondance of works and indices, remove the comment
  # for tup in zip(tokenized_text_split, indexed_tokens, segments_ids):
  #     print('{:<12} {:>6,} {:1}'.format(tup[0], tup[1], tup[2]))


  Indexed_tokens = []
  Segments_ids = []

  tmp1 = indexed_tokens
  tmp2 = segments_ids

  # BERT can support no more than 512 words
  for i in range(len(tmp1)):
    if len(tmp1) > BERT_LIMIT:
      Indexed_tokens.append(tmp1[:BERT_LIMIT])
      Segments_ids.append(tmp2[:BERT_LIMIT])
      tmp1 = tmp1[BERT_LIMIT:]
      tmp2 = tmp2[BERT_LIMIT:]
    else:
      Indexed_tokens.append(tmp1)
      Segments_ids.append(tmp2)
      break


  import pprint 
  print("INDEXED_TOKENS")
  pprint.pprint(Indexed_tokens)

  # print(indexed_tokens)

  # print(len(tokenized_text_split), len(indexed_tokens), len(segments_ids))



  Tokens_tensor = []
  Segments_tensors = []

  # Make tensors corresponding to words of the text
  for i, s in zip(Indexed_tokens, Segments_ids):
    Tokens_tensor.append(torch.tensor([i]))
    Segments_tensors.append(torch.tensor([s]))

  print("\033[31m"); print(Tokens_tensor); print("\033[0m")

  Embeddings = []


  for i in range(len(Tokens_tensor)):
    with torch.no_grad():
      # Apply the BERT model
      outputs = model(Tokens_tensor[i], Segments_tensors[i])
      hidden_states = outputs[2]

    # Build embeddings
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1
    token_embeddings = token_embeddings.permute(1,0,2)

    # print(token_embeddings.size())


    # Stores the token vectors
    token_vecs_sum = []

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


    # `token_vecs` is a tensor
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    print("\033[33m", end="")
    print ("Final sentence embedding vector of shape:", sentence_embedding.size())
    print("\033[0m", end="")
    Embeddings.append(sentence_embedding)
  
  return Embeddings

se = BERT(file_name)

SAVE = "../AIUEO.txt"
# for i in range(len(se)):
#   print("Vector {}: {}".format(i, se[i][:5]))


print("\033[31mWrite {}\033[0m".format(SAVE))
with open(SAVE, mode='w') as f:
  for i in range(len(se)):
    for j in range(len(se[i])):
      if j == len(se[i]) - 1:
        print("{}".format(se[i][j]), file=f)
      else:
        print("{}, ".format(se[i][j]), end="", file=f)
f.close()