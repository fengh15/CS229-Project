import math, ast, os.path
import random as rd
from collections import defaultdict
from multiprocessing import Pool

faces = {1: ' 1', 2: ' 2', 3: ' 3', 4: ' 4', 5: ' 5', 6: ' 6', 7: ' 7', 8: ' 8', 9: ' 9',
         10: '10', 11: ' J', 12: ' Q', 13: ' K', 14: ' A', 15: ' 2', 16: 'JK', 17: 'JR'}

antiface = {'3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, 'j':11, '12':12, 'q':12,
            '13':13, 'k':13, '14':14, 'a':14, '15':15, '2':15, '16':16, 'jk':16, '17':17, 'jr': 17}

def strHand(hand):
    a = '\n|'
    b = ' '
    for card in range(3,18):
        a += ''+faces[card]+'|'
        if hand[card] != 0:
            b += ' '+str(hand[card])+' '
        else:
            b += '   '
    return a+'\n'+b

#small library of convenient dictionary ops
def minusMove(d, move):
    if move is not None:
        start, pattern = move[0], move[1]
        for i, val in enumerate(pattern):
            d[start + i] -= val

def plusMove(d, move):
    if move is not None:
        start, pattern = move[0], move[1]
        for i, val in enumerate(pattern):
            d[start + i] += val

def minusDic(d1 = defaultdict(int), d2 = defaultdict(int)):
    total = defaultdict(int)
    for i in range(3,16):
        total[i] = 4 - d1[i] - d2[i]
    total[16] = 1 - d1[16] - d2[16]
    total[17] = 1 - d1[17] - d2[17]
    return total


#save and load weights
def saveWeights(weights, savefile='weights.txt', mode='a'):
    index = 1
    data = []
    if mode == 'a':
        if os.path.isfile(savefile):
            data = open(savefile, 'r').readlines()
            index = int(data[0]) + 1
            data[0] = str(index)+'\n'
        else:
            data.append(str(index) + '\n')
            data.append('\n')
    f = open(savefile, 'w')
    if mode != 'a':
        f.write('1\n\n')
    else:
        for line in data:
            f.write(line)
    f.write('Begin '+ str(index) + '\n')
    for k, v in weights.iteritems():
        f.write(str(k)+'|'+str(v)+'\n')
    f.write('End ' + str(index) + '\n\n')
    return

def loadWeights(savefile='weights.txt', index = 'max'):
    f = open(savefile, 'r')
    if index == 'max':
        index = int(f.readline())
    while True:
        line = f.readline()
        if line == 'Begin '+str(index)+'\n':
            break
    weights = defaultdict(float)
    while True:
        line = f.readline()
        if line == 'End ' + str(index)+'\n':
            break
        line = line.strip('\n')
        key, val = line.split('|')
        key = ast.literal_eval(key) #import ast!!!!
        val = ast.literal_eval(val)
        weights[key] = val
    return weights


# functions for feature extraction
def getPatterns(hand):
    patterns = [[], []]
    for card in hand.keys():
        if hand[card] >= 2 and hand[card + 1] >= 2:
            pattern = [hand[card], hand[card + 1]]
            i = card + 2
            while hand[i] >= 2:
                pattern.append(hand[i])
                i += 1
            que = []
            for i in range(2, pattern[0] + 1):
                que.append([i])
            listT = que[0]
            ind = 1
            while len(listT) < len(pattern):
                for i in range(2, pattern[len(listT)] + 1):
                    temp = listT[:]
                    temp.append(i)
                    patterns[0].append((card, temp))
                    que.append(temp)
                listT = que[ind]
                ind += 1
            patterns[1].append(patterns[0].pop())
    return patterns

def getPatternCounts(hand):
    patternCounts = [defaultdict(int), defaultdict(int)]
    for card in hand.keys():
        if hand[card] >= 2 and hand[card + 1] >= 2:
            pattern = [hand[card], hand[card + 1]]
            i = card + 2
            while hand[i] >= 2:
                pattern.append(hand[i])
                i += 1
            que = []
            for i in range(2, pattern[0] + 1):
                que.append([i])
            listT = que[0]
            ind = 1
            while len(listT) < len(pattern):
                for i in range(2, pattern[len(listT)] + 1):
                    temp = listT[:]
                    temp.append(i)
                    patternCounts[0][sum(temp)] += 1
                    que.append(temp)
                listT = que[ind]
                ind += 1
            patternCounts[0][sum(pattern)] -= 1
            patternCounts[1][sum(pattern)] += 1
    return patternCounts

# number of possible counters to a move based on the possible remaining hand
def C(x, y):
    if x<0 or y<0 or x<y:
        return 0
    val = 1
    for i in range(x-y+1, x+1):
        val*=i
    for i in range(1, y+1):
        val/=i
    return val

def counterCount(hand, move):
    if move is None:
        lowCard = 2
        pattern = []
    else:
        lowCard, pattern = move
    totalCounters = 0
    for i in range(lowCard + 1, 19 - len(pattern)):  # loop over start of patter
        legal = True
        for j in range(len(pattern)):  # loop over parts of pattern
            legal &= hand[i + j] >= pattern[j]
        if legal:
            totalCounters += 1
    for card in hand:
        if hand[card] == 4:  # a bomb
            totalCounters += 1
    return totalCounters

def counterProb(opCount, hand, move, numSamples = 1000):
    def hasLegalMove(nhand):
        for i in range(lowCard + 1, 19 - len(pattern)):  # loop over start of patter
            legal = True
            for j in range(len(pattern)):  # loop over parts of pattern
                legal &= nhand[i + j] >= pattern[j]
            if legal:
                return True
        return False

    if move is None:
        return 1
    lowCard, pattern = move
    if not hasLegalMove(hand):
        return 0

    cardList = []
    for k, v in hand.items():
        for i in range(v):
            cardList.append(k)
    sampleHand = defaultdict(int)
    numSuccess = 0
    for i in range(numSamples):
        sampleHandList = rd.sample(cardList, opCount)
        for val in sampleHandList:
            sampleHand[val] += 1
        numSuccess += 1*hasLegalMove(sampleHand)
        for val in sampleHandList:
            sampleHand[val] -= 1
    return 1.0*numSuccess/numSamples



#Archive of stuff not used:
def featNormalizer(features, nw=defaultdict(list)):
    nfeats = []
    for f, v in features:
        if f not in nw:
            nw[f] = [v, v]
        nw[f][0] = min(nw[f][0], v)
        nw[f][1] += max(nw[f][1], v)
        nfeats.append((f, 1 if nw[f][0]==nw[f][1] else (1.0*v - nw[f][0])/(nw[f][1] - nw[f][0])))
    return nfeats

def sumNormalizers(nw1, nw2):
    for key in nw2:
        if key not in nw1:
            nw1[key] = nw2[key]
        else:
            nw1[key][0] = min(nw1[key][0], nw1[key][0])
            nw1[key][1] = max(nw2[key][0], nw2[key][1])
    return nw1