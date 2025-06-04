# optional
# need to download media by using URL attached in the Twitter dataset (according to media key)
# apply to jpg, png, jpeg

import importlib.util
import torch


globalML_device = 'cpu'
if torch.cuda.is_available():
      globalML_device = "cuda:0"
elif torch.backends.mps.is_available():
      globalML_device = 'mps'
print(f'Running on {globalML_device}')

def initial_sentenceTransformer():

    package_name = 'sentence_transformers'

    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(package_name +" is not installed")
        %pip install -U sentence-transformers

    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer('clip-ViT-B-32', device = globalML_device)

    return model

model = initial_sentenceTransformer()
tokenizer = model._first_module().processor.tokenizer

def get_img_embeddings(imgs_path):
    img_emb = model.encode([Image.open(filepath) for filepath in imgs_path], batch_size=128, convert_to_tensor=True, show_progress_bar=False, device = globalML_device)
    return img_emb.cpu().numpy().tolist()

def get_text_embedding(text_chunk, text_batch_size = 50):

    text_embeddings = []

    def generate_batches(lst, batch_size):
      for i in range(0, len(lst), batch_size):
          yield lst[i:i + batch_size]

    def truncate_document(document, tokenizer):
      tokens = tokenizer.encode(document)
      if len(tokens) > 77:
        truncated_tokens = tokens[1:76]
        document = tokenizer.decode(truncated_tokens)
        return truncate_document(document, tokenizer)
      else:
        return document


    batches = generate_batches(text_chunk, text_batch_size)

    def text_embed_func(inputs, batch_size = 50):
        # get text embedding
        not_empty_lst = []
        res = []
        for idx, t in enumerate(inputs):
          if pd.notna(t):
            if len(t.split(' ')) > 0:
              not_empty_lst.append(idx)
              res.append(t)
        # embedding valid text
        encoded_text = [truncate_document(doc, tokenizer) for doc in res]
        text_embed = model.encode(encoded_text, convert_to_tensor=True, show_progress_bar=False).cpu().numpy().tolist()

        # reallocate the list to store embeddings
        embed = [None for i in range(len(inputs))]
        index = 0
        for idx in not_empty_lst:
          embed[idx] = text_embed[index]
          index += 1
        return embed

    for batch in batches:
        text_embeddings.append(text_embed_func(batch, text_batch_size))
    return [item for sublist in text_embeddings for item in sublist]

img_embed = get_text_embedding(imgs_path)
text_embed = get_text_embedding(text_chunk)

    