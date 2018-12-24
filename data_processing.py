import numpy as np
from sklearn.linear_model import LogisticRegression


def process_data():
  """
  Function that reads the 'uk_iu-ud-train.conllu' data file and extracts
  sentences, words, tags, features.
  returns : data_set list that contains: [data_entry1,data_entry2, ...]
  where data entry contains: [sentence, [words],[tags],[features]]
  Words list contains each word of sentence in right order.
  Tags list contains tag for each word of the sentence in right order.
  Features list contains features for each word of the sentence in right order.
  tags contain :
  {'CCONJ', 'SCONJ', 'NOUN', 'PUNCT', 'PRON', 'ADP', 'ADJ', 'VERB', 'ADV', 'PART', 'X', 'SYM', 'NUM', 'AUX', 'PROPN', 'DET', '_', 'INTJ'}
  where X tag is a word from foreign language.
  """
  
  file = open("data/uk_iu-ud.conllu","r", encoding="utf-8")
  #initializing lists
  words = []
  tags = []
  features = []
  data_entry = []
  data_set = []

  for line in file:
    if line.startswith("# text = "):
      #adding sentance to data_entry
      data_entry.append(line.strip("# text = ").strip("\n"))
      continue
    if(line[0].isdigit()):
      #generating words, tags and features
      word_tag_feature = line.split("\t")
      words.append(word_tag_feature[2])
      tags.append(word_tag_feature[3])
      features.append(" ".join(word_tag_feature[4:]).strip("\n"))
      continue
    else:
      if(len(words)>0):
        # End of 1 sentance, apending words, tags, features to data entry
        # and appending data_entry to data_set. After that cleaning data_
        # entry for next sentence.
        data_entry.append(words)
        data_entry.append(tags)
        data_entry.append(features)
        data_set.append(data_entry)
        data_entry = []
        words = []
        tags = []
        features = []
  file.close()
  return data_set


def create_language_voc(data_set):
    # Function takes data set as parameter and returns unique set of words in data set.
    word_set =  []
    for lst in data_set:
        for word in lst[1]:
            if word not in word_set:
                word_set.append(word)
    return word_set


def extract_morph_tags(data_set):
    # Function takes data set as parameter and returns unique set of tags in data set.
    tags_set = []
    for lst in data_set:
        for tag in lst[2]:
            if tag not in tags_set:
                tags_set.append(tag)
    return tags_set


def extract_features(sentence, voc, tags_voc):
    """
    Extracts features for each word in the sentence and morph tag as a label
    :param sentence: sentence, [words],[tags],[features]
    :return: features_list, labels_list
    """
    features_list = []
    labels_list = []
    if not "START_SENTENCE" in tags_voc:
        tags_voc.append("START_SENTENCE")
    last_tag = "START_SENTENCE"

    for i in range(len(sentence[1])):
        word = sentence[1][i]
        tag = sentence[2][i]

        labels_list.append(tag)

        # vector representation of word in the vocabulary
        word_to_vec = np.zeros(len(voc))
        word_to_vec[voc.index(word)] = 1

        # vector representation of the previous word morph tag
        last_tag_to_vec = np.zeros(len(tags_voc))
        last_tag_to_vec[tags_voc.index(last_tag)] = 1
        last_tag = tag

        # if the word is the last in the sentence
        is_end = np.zeros(1) if i < len(sentence[1])-1 else np.ones(1)

        feature_vector = np.hstack((word_to_vec, last_tag_to_vec, is_end))
        features_list.append(feature_vector)

    features_list = np.array(np.vstack(features_list), dtype=np.int8)
    return features_list, labels_list


def get_dataset(prepared_data, voc, tags_voc):
    X, y = [], []

    for i, sentence in enumerate(prepared_data):
        print(i)
        ftrs, lbls = extract_features(sentence, voc, tags_voc)
        X.extend(ftrs)
        y.extend(lbls)

    return X, y


if __name__ == "__main__":
    processed_data = process_data()
    # print(processed_data)
    words = create_language_voc(processed_data)
    tags = extract_morph_tags(processed_data)
    X, y = get_dataset(processed_data, words, tags)
    del processed_data
    X = np.array(X, dtype=np.int8)
    model = LogisticRegression()
    model.fit(X,y)
    print(model.score(X,y))