def process_data():
  """
  Function that reads the 'uk_iu-ud-train.conllu' data file and extracts
  sentances, words, tags, features.
  returns : data_set list that contains: [data_entry1,data_entry2, ...]
  where data entry contains: [sentance, [words],[tags],[features]]
  Words list contains each word of sentance in right order.
  Tags list contains tag for each word of the sentance in right order.
  Features list contains features for each word of the sentance in right order.
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
        # entry for next sentance. 
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
    word_set = set()
    for lst in data_set:
        for word in lst[1]:
            word_set.add(word)
    return word_set


def extract_morph_tags(data_set):
    # Function takes data set as parameter and returns unique set of tags in data set.
    tags_set = set()
    for lst in data_set:
        for tag in lst[2]:
            tags_set.add(tag)
    return tags_set



if __name__ == "__main__":
    procesed_data = process_data()
    print(procesed_data)
    words = create_language_voc(procesed_data)
    tags = extract_morph_tags(procesed_data)