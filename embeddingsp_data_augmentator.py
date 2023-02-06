import sys        
sys.path.append('./src')
import re
from augmentators.augment_embeddingsp import substitute_with_embedding, train_fasttext_model

if __name__ == "__main__":
    example = "Kupat tahu gempol iku salah siji kupat tahu legendaris. Panggone ora tek gedhe nanging cukup nyolok karo werna abange ning protelon pasar gempol. Porsine pas banget dingge ngilangne luwe. Ciri khas saka kupat tahu iki yaiku tahune lembut lan kenyel lan duduh kacange pas ning ilat, dicampur karo lontong, capar lan krupuk abang. Enak banget."
    
    model_path = train_fasttext_model('./data/augment_embeddingsp/javanese_mono/java_mono_clean')
    substitute_with_embedding([example], model_path)
    # print("===============================================")
    # example = re.sub(r'[^A-Za-z0-9 -]+', '', example).lower()
    # print(example)
    # for token in example.split():
    #     print(token)
    #     nearest_neighbors = model.get_nearest_neighbors(token)
    #     for i in range(7):
    #         print(nearest_neighbors[i])
    # print("===============================================")
    