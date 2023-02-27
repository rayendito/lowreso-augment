import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.linalg import norm
from loaders.load_model_and_tokenizer import load_model_and_tokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences

def _get_torch_tensor_sim(a, b):
  return torch.dot(a/norm(a), b/norm(b)).item()

def _get_BERT_sentence_representation(sentence, model, tokenizer):
  encoded_input = tokenizer(sentence, return_tensors='pt')
  output = model(**encoded_input)
  return output['last_hidden_state'][0][0]

def _get_representation_dict(corpus, model, tokenizer):
  dic = {}
  for sentence in corpus:
    dic[sentence] = _get_BERT_sentence_representation(sentence, model, tokenizer)
  return dic
  
def _get_closest_from_a_sentence(sentence, model, tokenizer, representation_dict):
  sent_rep = _get_BERT_sentence_representation(sentence, model, tokenizer)
  closest_sentence= ''
  highest_similarity = 0

  for sent in representation_dict:
    dist = _get_torch_tensor_sim(sent_rep, representation_dict[sent])
    if(dist >= highest_similarity):
      highest_similarity = dist
      closest_sentence = sent
    
  return closest_sentence

def _insert_a_sentence_to_a_sentence(splitter, tgt_sentence, inserted_sentence):
  return 'wleeeeeee'

def _translate_a_sentence():
  return 'wleeeeeeee'

def augment_parallel_with_insertion(target_corpus, mono_corpus, dataset_name, src_lang, tgt_lang, pretrained_path):
  tgt_path = "./data/augment_embeddingsp/"+"insertion_augmented_{}_{}-{}.csv".format(dataset_name, src_lang, tgt_lang)
  print("Creating new training instances with insertion...")
  print("Loading pretrained model...")
  model, tokenizer = load_model_and_tokenizer(pretrained_path)
  splitter = SentenceSplitter(language='en')

  print("Getting vector representation for sentences...")
  representation_dict = _get_representation_dict(mono_corpus, model, tokenizer)

  for thing in representation_dict:
    print(thing)
    print(representation_dict[thing])
    break
  # for sent in tqdm(target_corpus):
  #   closest_sentence = _get_closest_from_a_sentence(sent, model, tokenizer, representation_dict)
  #   new_tgt_sentence = _insert_a_sentence_to_a_sentence(splitter, sent, closest_sentence)
  #   new_src_sentence = _translate_a_sentence()

  #   df_augmented = pd.DataFrame([(new_src_sentence, new_tgt_sentence)], columns =[src_lang, tgt_lang])
  #   if(os.path.isfile(tgt_path)):
  #     df_augmented.to_csv(tgt_path, mode='a', index=False, header=False)
  #   else:
  #     df_augmented.to_csv(tgt_path, index=False)

  # print("Created new training instances in {}".format(tgt_path))