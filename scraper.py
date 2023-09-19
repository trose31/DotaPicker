"""This file sources the test and training data from the API, and saves them in a txt file. """

import json
import requests
import random
import math

def trainLoader(n):
    loader('train', n)

def testLoader(n):
    loader('test', n)

def loader(source, n):
    for i in range(0,n):
        if source == 'train':
            sampleIndex = 7100000000+i*(math.ceil(200000000/n))         #7.1*1e10 is a match ID from March (the new frontiers game update), data older than this is not as useful
        else:
            sampleIndex = 7300000000+i*(math.ceil(12000000/n))

        key = True

        if key:
            keyID = '&api_key=53e34b0d-8d5c-4a04-8528-79ab386cd079'     #Using the API key increases the number of calls that can be made per minute but adds a cost (a very low one)
        else:
            keyID = ''
            
        urlSpec = 'less_than_match_id='+str(sampleIndex)+'&min_rank=30&max_rank=56'+keyID
        response_API = requests.get('https://api.opendota.com/api/publicMatches?'+urlSpec)
        
        data = response_API.text
        txt, labels = batchProcess(data)
        write(txt, labels, source)

        print('Batch: ', i, ' loaded to memory.')
    print('Done')
    

def write(txt, labels, dest):
    f = open('raw/'+str(dest)+'values.txt', 'a')
    g = open('raw/'+str(dest)+'labels.txt', 'a')

    for entry in txt:
        f.write('\n' + str(entry))
    f.close()

    for label in labels:
        g.write('\n' + str(label))
    g.close()


"""The only way to sample multiple matches in the same call is by requesting 'publicMatches' from the API,
which returns 100, however the returned string includes lots of unnecessary information which must be
filtered out. """
        
def batchProcess(batch):
    batch = batch.split("},{")

    j = 0
    data, labels = [[] for i in range(0,100)], [0 for i in range(0,100)]
    
    for item in batch:
        item = item.split('"')
        x = [0 for i in range(250)]
        
        for i in item[27].split(','):
            n = dictionaryConvert(int(i))
            x[n - 1] = 1
            
        for i in item[31].split(','):
            n = dictionaryConvert(int(i))
            x[n + 124] = 1
            
        y = int(item[6]==':true,')

        data[j], labels[j] = x, y
        j += 1

    output = [data, labels]
    return output

"""Frustratingly the owner of the API uses an indexing for the heroes with several gaps,
which would result in never activated input neurons in the network if left unchecked. So
here they are converted to consecutive indexes. """

def dictionaryConvert(n):
    dictionary = {
        119: 115,
        120: 116,
        121: 117,
        123: 118,
        126: 119,
        128: 120,
        129: 121,
        135: 122,
        136: 123,
        137: 124,
        138: 125}
    if (n < 115):
        return n
    else:
        return dictionary[n]
