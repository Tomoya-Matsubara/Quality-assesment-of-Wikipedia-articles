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
  print("\033[31m"); print(text); print("\033[0m")

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

  print("\033[32m"); print(tokenized_text_split); print("\033[0m")

  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_split)

  # If you want to see the correspondance of works and indices, remove the comment
  # for tup in zip(tokenized_text_split, indexed_tokens, segments_ids):
  #     print('{:<12} {:>6,} {:1}'.format(tup[0], tup[1], tup[2]))

  # BERT can support no more than 512 words
  if len(indexed_tokens) > BERT_LIMIT :
    indexed_tokens = indexed_tokens[:BERT_LIMIT]
    segments_ids = segments_ids[:BERT_LIMIT]


  # print(len(tokenized_text_split), len(indexed_tokens), len(segments_ids))

  # Make tensors corresponding to words of the text
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  # print(tokens_tensor)


  with torch.no_grad():
    # Apply the BERT model
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]

  # Build embeddings
  token_embeddings = torch.stack(hidden_states, dim=0)

  # Remove dimension 1
  token_embeddings = torch.squeeze(token_embeddings, dim=1)

  # Swap dimensions 0 and 1
  token_embeddings = token_embeddings.permute(1,0,2)

  print(token_embeddings.size())


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

  print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
  return sentence_embedding

se = BERT(file_name)
print(se)

SAVE = "../ABC.txt"
with open(SAVE, mode='w') as f:
  for i in range(len(se)):
    if i == len(se) -1:
      print("{}".format(se[i]), file=f)
    else:
        print("{}, ".format(se[i]), end="", file=f)
f.close()