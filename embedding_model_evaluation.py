import sys        
sys.path.append('./src')

import fasttext
import pickle
from embd_evaluator.embeddingsp_eval import evaluate_compactness_score
from augmentators.augmentator_embeddingsp import train_fasttext_model_return_path

f = open('./data/augment_embeddingsp/embedding_model_evaluation/compactness_test.pkl', 'rb')   # 'rb' for reading binary file
refs = list(pickle.load(f).values())
f.close()

#===========================COMPACTNESS SCORES===========================
method = "skipgram"
scores = []
for epoch in range(70,80):
    model_path = train_fasttext_model_return_path('./data/augment_embeddingsp/javanese_mono/java_mono_clean',
                                        method = method,
                                        minn = 2,
                                        maxn = 4,
                                        epoch = epoch,
                                        lr = 0.1)
    model = fasttext.load_model('./data/augment_embeddingsp/embedding_models/{}/jv.{}.{}.300.bin'.format(method, method, epoch))
    compactness_scores = evaluate_compactness_score(model, refs)
    print("method {} epoch {} : compactness {}".format(method, epoch, compactness_scores))
    scores.append(compactness_scores)

print("ORDERED RESULTS, COPYABLE")
for score in scores:
    print(score)