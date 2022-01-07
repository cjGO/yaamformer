from ast import literal_eval


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


