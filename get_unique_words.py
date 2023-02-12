import re

splittedWords = open('./data/augment_embeddingsp/javanese_mono/raw_separate/nusax_train_cleaned').read().lower().split()
# splittedWords = re.sub(r'[^A-Za-z0-9 -]+', '', splittedWords).lower().split()
uniqueValues = set(splittedWords)

print(len(uniqueValues))
with open(r'nusax_train_unique_tokensaaaaaa', 'w') as fp:
    for item in list(uniqueValues):
        fp.write("%s\n" % item)
    print('Done')