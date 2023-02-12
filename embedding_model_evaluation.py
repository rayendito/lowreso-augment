import sys        
sys.path.append('./src')
from embd_evaluator.embeddingsp_eval import evaluate_compactness_score
import fasttext
import pickle


import pickle

f = open('./data/augment_embeddingsp/embedding_model_evaluation/compactness_test.pkl', 'rb')   # 'rb' for reading binary file
refs = list(pickle.load(f).values())
f.close()

model = fasttext.load_model('./data/augment_embeddingsp/embedding_models/jv.cbow.20.300.bin')
compactness_scores = evaluate_compactness_score(model, refs)

print("===========================COMPACTNESS SCORE===========================")
print(compactness_scores)