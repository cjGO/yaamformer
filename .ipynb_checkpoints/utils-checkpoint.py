def get_dataset():# Using readline()
    file1 = open('./data/YAAM.txt', 'r')
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
