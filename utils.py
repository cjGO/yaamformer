from ast import literal_eval
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset


def get_dataset(file_location):# Using readline()
    file1 = open(file_location, 'r')
    count = 0

    Seqs = []
    Labels = []

    while True:
        count += 1
        #print(count)
        # Get next line from file
        line = file1.readline()

        if not line:
            break

        labels = literal_eval(line.split('\t')[2])
        seqs = line.split('\t')[5]
        seqs = seqs.replace('\n','')
        assert len(seqs) == len(labels), 'MISMATCH'
        Seqs.append(seqs)
        Labels.append(labels)
        # if line is empty
        # end of file is reached
    #     if count >20:
    #         break
    return Seqs,Labels

def fragment_proteins(dataframe, segment_size):
    """
    pass in a test/train dataframe to segment it into smaller fragments for batch training
    """
    
    fragment_seqs = []
    fragment_labels = []

    for S in range(len(dataframe)):

        test_seq = dataframe.iloc[S]['seq']
        test_label = dataframe.iloc[S]['label']

        #find each PTM in label
        #for each label check if there is 10 nt to left
            #if not find how many

        for i in np.where(test_label == 1.0): #for each ptm location

            for ptm in i:
                if (ptm - segment_size >= 0) & (ptm + segment_size < len(test_seq)) : #simple case
                    fragment_seq = test_seq[ptm-segment_size+1:ptm+segment_size]
                    fragment_label = test_label[ptm-segment_size+1:ptm+segment_size]

                elif ptm - segment_size < 0: # front end
                    left_gap = segment_size + (ptm - segment_size)
                    right_gap = segment_size*2 - left_gap
                    fragment_seq = test_seq[ptm-left_gap:ptm+right_gap-1]
                    fragment_label = test_label[ptm-left_gap:ptm+right_gap-1]


                elif ptm + segment_size > len(test_seq): #back end
                    right_gap = len(test_seq)-ptm
                    left_gap =  segment_size*2 - right_gap
                    fragment_seq = test_seq[ptm-left_gap+1:ptm+right_gap]
                    fragment_label = test_label[ptm-left_gap+1:ptm+right_gap]



                assert len(fragment_seq) == segment_size*2 -1, f'wrong length \n{S} \n{fragment_seq} \n {fragment_label}'
                assert len(fragment_label) == segment_size*2 -1, f'wrong length \n{S} \n{fragment_seq} \n {fragment_label}'
                assert sum(fragment_label) > 0, f'no label \n{S} \n{fragment_seq} \n {fragment_label}'

                if fragment_seq not in fragment_seqs:
                    fragment_seqs.append(fragment_seq)
                    fragment_labels.append(fragment_label)

    return pd.DataFrame({'seq':fragment_seqs,'label':fragment_labels})


def tokenize_protein(sequence):
  #print(sequence)
  sequence = re.sub(r"[UZOB]", "X", sequence)
  sequence = ' '.join(sequence.replace('\n',''))
  return sequence

class NERprotein(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def index_char(string,character):
  """
  returns all index of a given character 
  e.g. all index of # in 'ATTTATTA#TATT#TA#TAT#T'
  """
  return [pos for pos, char in enumerate(string) if char == character]

def pop_string(string, locations):
  """
  removes several locations simulatenously
  pass in string and LIST of indexes
  """
  data, indexes = string, set(locations)
  return "".join([char for idx, char in enumerate(data) if idx not in indexes])

def split_string(string,size):
  """
  splits string into equal sized chunks (or shorter at end)
  """
  return [string[i:i+size] for i in range(0, len(string), size)]


  """
  predicting with models
ptmpredict = TokenClassificationPipeline(model=model,tokenizer=tokenizer)
predicts = ptmpredict(tokenize_protein(test['seq'][1]))
rank=np.argsort(scores)[:10]
actual=np.where(test['label'][1]==1)
np.intersect1d(rank,actual)

  """


def pad_labels(label,max_length=1024):
    """
    takes a label of array[1 0 0 0 0 1] 
    and pads it to desired length e.g. [1 0 0 0 0 1 0 0 ... 0]
    """
    new_label = np.zeros(max_length)
    new_label[:len(label)] = label
    return new_label