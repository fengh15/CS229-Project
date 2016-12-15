import math
import random as rd
from collections import defaultdict
from multiprocessing import Pool
from TDUtil import *

# 2 feature extractors
def fE1(hand, table, index, patlen=0, fnum=1):
    feats = []

    #difference and fraction
    Alen = (table.AgentCardCount[index] - patlen)/18
    Olen = (table.AgentCardCount[index - 1])/18
    feats.append(((fnum, 'Alen'), Alen))
    feats.append(((fnum, 'Olen'), Olen))
    feats.append(((fnum, 'Dif1'), Alen - Olen))
    # feats.append(((fnum, 'Ratio'), (Alen - Olen)/(Alen + Olen)))

    type = ['0', 'single', 'double', 'triple', 'bomb']
    for card in hand:
        if hand[card] >= 1:
            feats.append(((fnum, type[hand[card]], card), 1))

    pC = getPatternCounts(hand) # two dictionaries for [sub patterns, full patterns]
    for key in pC[0]:
        if pC[0][key] > 0:
            feats.append(((fnum, 'SubP '+str(key)), pC[0][key]/3))
    for key in pC[1]:
        if pC[1][key] > 0:
            feats.append(((fnum, 'FullP '+str(key)), pC[1][key]))
    return feats

def fE2(hand, table, index, move):
    minusMove(hand, move)
    patlen = 0 if move is None else sum(move[1])
    feats = fE1(hand, table, index, patlen, 2)
    #add additional features
    plusMove(hand, move)
    return feats




# Trains and tests weights, train functions depend on train_generic, and play functions (for testing)
# depend on playVersus. x is always the number of iterations to run a function
class TDAlgorithm:
    def __init__(self, verbose=False, weights=defaultdict(float), featureExtractors = (fE1, fE2),
                 explorationProb = 0.2, margin=0.2):
        self.featureExtractors = featureExtractors
        self.weights = weights
        self.explorationProb = explorationProb
        self.iteration = 0
        self.margin = margin
        self.verbose = verbose

    def evalFeats(self, feats):
        value = 0
        for feat, val in feats:
            value += self.weights[feat] * val
        return value

    def stepSize(self):
        return 5/math.sqrt(self.iteration + 100)

    def train_generic(self, g, x):
        for i in range(x):
            if self.verbose:
                if self.iteration%5000 == 0:
                    print "Iteration: " + str(self.iteration)
            history = g.play()
            self.updateWeights(history)
            g.update(self.weights, self.featureExtractors, self.explorationProb)
            self.iteration += 1
        print "...done..."

    def train(self, x):
        g = game(2, 0, 0, 0, self.featureExtractors, self.weights, self.explorationProb)
        self.train_generic(g, x)

    def trainG(self, x):
        g = game(1, 0, 0, 1, self.featureExtractors, self.weights, self.explorationProb)
        self.train_generic(g, x)

    def trainR(self, x):
        g = game(1, 0, 1, 0, self.featureExtractors, self.weights, self.explorationProb)
        self.train_generic(g, x)

    def trainH(self, x):
        g = game(1, 0, 1, 0, self.featureExtractors, self.weights, self.explorationProb)
        self.train_generic(g, x)

    def reset(self):
        self.weights = defaultdict(float)
        self.iteration = 0

    def playVersus(self, x, g):
        wins = 0
        for i in range(x):
            history = g.play()
            wins += history[0][1] == 1
        return wins

    def playTDvR(self, x):
        g = game(1, 0, 1, 0, self.featureExtractors, self.weights, 0)
        return self.playVersus(x, g)

    def playTDvG(self, x):
        g = game(1, 0, 0, 1, self.featureExtractors, self.weights, 0)
        return self.playVersus(x, g)

    def playTDvH(self, x):
        g = game(1, 1, 0, 0, self.featureExtractors, self.weights, 0)
        return self.playVersus(x, g)

    def playRvG(self, x):
        g = game(0, 0, 1, 1, self.featureExtractors, self.weights, 0)
        return self.playVersus(x, g)

    def playHvR(self,x):
        g = game(0, 1, 1, 0, self.featureExtractors, self.weights, 0)
        return self.playVersus(x, g)

    def playHvG(self, x):
        g = game(0, 1, 0, 1, self.featureExtractors, self.weights, 0)
        return self.playVersus(x, g)

    def updateWeights(self, history):
        # TD learn with history
        for hist in history:
            if hist[0][0] == "TD":  # or hist[0][0] == "HUMAN":
                score = hist[1]
                fset = hist[0][1:]
                if len(fset) != 0: # Once in a long while (~1 in 100,000), a very lucky computer gets a hand that's a single long pattern
                    phi2 = fset[-1][1]
                    dv = self.evalFeats(phi2) - score
                    for f, v in phi2:
                        self.weights[f] -= self.stepSize()*dv*v
                    for i in range(1, 2*len(fset)):
                        phi2 = fset[-i/2][i%2]
                        phi1 = fset[-(i+1)/2][(i+1)%2]
                        dv = self.evalFeats(phi1) - self.evalFeats(phi2)
                        for f, v in phi1:
                            self.weights[f] -= self.stepSize()*dv*v*(self.margin**1)


# stores data on pattern, lowcard, card counts, and which cards have been played
class tableClass:
    def __init__(self, ACC):
        self.pattern = []
        self.lowcard = 0
        self.cards = defaultdict(int)
        self.AgentCardCount = ACC

    def incorporateMove(self, player, move):
        if move is None:
            self.pattern = []
            self.lowcard = 0
        else:
            self.pattern = move[1]
            self.lowcard = move[0]
            self.AgentCardCount[player] -= sum(move[1])
            plusMove(self.cards, move)

# def multiplay(g):
#     return g.play()
# inits a game using a number of different agents, and takes feature extractors, weights,
# and exploration prob for TD
class game:
    def draw(self):  # initializes 3 decks of cards
        deck = range(3, 16) * 4
        deck.append(16)
        deck.append(17)
        rd.shuffle(deck)
        hands = [defaultdict(int), defaultdict(int), defaultdict(int)]
        for i in range(18):
            hands[0][deck[i]] += 1
            hands[1][deck[i + 18]] += 1
            hands[2][deck[i + 36]] += 1
        return hands

    def __init__(self, numTD, numHumans, numRandom, numGreedy, featureExtractors=(fE1, fE2),
                 weights=defaultdict(float), explorationProb = 0.2):  # numC + numH <=2 for now, will expand to 3
        self.numPlayers = numTD + numHumans + numRandom + numGreedy
        self.numTD = numTD
        self.numHumans = numHumans
        self.featureExtractors = featureExtractors
        self.explorationProb = explorationProb
        self.hands = []
        self.agents = []
        self.gamesPlayed = 0
        for i in range(self.numPlayers):
            if i < numTD:
                self.agents.append(TDAgent(i, self.numPlayers, featureExtractors[0], featureExtractors[1],
                                           weights, explorationProb))
            elif i < numTD + numHumans:
                self.agents.append(humanAgent(i, self.numPlayers))
            elif i < numTD + numHumans + numRandom:
                self.agents.append(RandomAgent(i, self.numPlayers))
            else:
                self.agents.append(GreedyAgent(i, self.numPlayers))
        self.table = None

    def update(self, weights, featureExtractors, explorationProb):
        for i in range(self.numTD):
            self.agents[i] = TDAgent(i, self.numPlayers, featureExtractors[0], featureExtractors[1]
                                     , weights, explorationProb)

    def play(self):
        self.hands = self.draw()
        self.table = tableClass([18 for i in range(self.numPlayers)])
        player = rd.randint(0, self.numPlayers - 1)
        while True:
            move = self.agents[player].getMove(self.hands[player], self.table)
            self.table.incorporateMove(player, move)
            minusMove(self.hands[player], move)
            if self.numHumans == 1:
                if move is not None:
                    start = faces[move[0]].strip().upper()
                    print '\nPlayer '+str(player)+' plays '+str(start)+', '+str(move[1])
                else:
                    print '\nPlayer '+str(player)+' passes'
            if self.table.AgentCardCount[player] == 0: break
            player = (player + 1) % self.numPlayers
        if self.numHumans == 1:
            print '\nPlayer ', player, ' wins!!!'
            act = raw_input('\nPress [ENTER] to continue\n').strip()
        history = [(self.agents[i].history, 2.0*(i==player) - 1.0) for i in range(self.numPlayers)]
        self.gamesPlayed += 1
        if self.numHumans == 1:
            print 'Total Games Played:', self.gamesPlayed
        return history

# computerAgent and humanAgent extend this class
# ALL agents must have this structure, and especially 'getMove'
class Agent:
    def __init__(self, index, numPlayers):
        self.index = index
        self.numPlayers = numPlayers
        self.prev = (self.index - 1) % self.numPlayers
        self.prev2 = (self.index - 2) % self.numPlayers
        self.history = []

    def getMove(self, hand, table): raise NotImplementedError("Override me")

# TDAgent, RandomAgent, and GreedyAgent all extend this class
class computerAgent(Agent):
    def __init__(self, index, numPlayers):
        Agent.__init__(self, index, numPlayers)

    def getFirstMoves(self, hand):
        moves = []
        # singles, doubles, triples
        for card in hand.keys():
            for i in range(hand[card]):  # if 3 cards of a number, range is 0,1,2
                moves.append((card, [i + 1]))
        # bomb
        for card in hand.keys():
            if hand[card] == 4:
                moves.append((card, [4]))
        # consecutive moves
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
                        moves.append((card, temp))
                        que.append(temp)
                    listT = que[ind]
                    ind += 1
        return moves
    def getLegalMoves(self, hand, lowCard, pattern):
        moves = []
        if len(pattern) == 0:
            return self.getFirstMoves(hand)
        for i in range(lowCard + 1, 19 - len(pattern)):  # loop over start of patter
            legal = True
            for j in range(len(pattern)):  # loop over parts of pattern
                legal &= hand[i + j] >= pattern[j]
            if legal:
                moves.append((i, pattern))
        for card in hand:
            if hand[card] == 4:  # a bomb
                moves.append((card, [4]))
        if len(moves)==0:
            moves = [None]
        # moves.append(None)
        return moves

class TDAgent(computerAgent):
    def __init__(self, index, numPlayers, preFE, postFE, weights, explorationProb):
        computerAgent.__init__(self, index, numPlayers)
        self.preFE = preFE
        self.postFE = postFE
        self.weights = weights
        self.explorationProb = explorationProb
        self.history.append("TD")

    def evalFeats(self, feats):
        value = 0
        for feat, val in feats:
            value += self.weights[feat] * val
        return value


    def getMove(self, hand, table): #get move, add features to history
        moves = self.getLegalMoves(hand, table.lowcard, table.pattern)
        if rd.uniform(0,1) < self.explorationProb:
            return rd.choice(moves)
        else:
            f1 = self.preFE(hand, table, self.index)
            f2s = [self.postFE(hand, table, self.index, move) for move in moves]
            valAndMoves = [(self.evalFeats(f2s[i]), moves[i]) for i in range(len(moves))]
            maxVal = max(valAndMoves)[0]
            maxMoves = []
            for v, m in valAndMoves:
                if v == maxVal:
                    maxMoves.append(m)
            move = rd.choice(maxMoves)
            f2 = self.postFE(hand, table, self.index, move)
            self.history.append((f1, f2))
            return move

class RandomAgent(computerAgent): #Fully functional
    def __init__(self, index, numPlayers):
        computerAgent.__init__(self, index, numPlayers)
        self.history.append("RAND")

    def getMove(self, hand, table): #get move, add move to history
        moves = self.getLegalMoves(hand, table.lowcard, table.pattern)
        self.history.append(rd.choice(moves))
        return self.history[-1]

class GreedyAgent(computerAgent):
    def __init__(self, index, numPlayers):
        computerAgent.__init__(self, index, numPlayers)
        self.history.append("GREED")

    def getMove(self, hand, table): #get move, add move to history
        moves = self.getLegalMoves(hand, table.lowcard, table.pattern)
        # print "CM: " + str(moves)
        self.history.append(moves[0])
        return self.history[-1]

class humanAgent(Agent):
    def __init__(self, index, numPlayers):
        Agent.__init__(self, index, numPlayers)
        print '\nYou are player', index
        self.history.append("HUMAN")

    def getMove(self, hand, table):
        def moveStringToTuple(move):
            if move == "":
                return None
            if ':' in move:
                start, pattern = move.split(':')
            elif ' ' in move:
                start, pattern = move.split(' ')[0:2]
            else:
                return -1
            pattern = pattern.split(',')
            start = start.lower()
            if start in antiface:
                start = antiface[start]
            else:
                return -1
            if len(pattern) == 1 and pattern[0] == '':
                return start, [1]
            for i in range(len(pattern)):
                try:
                    pattern[i] = int(pattern[i])
                except ValueError:
                    return -1
            return start, pattern

        print "\n=== PLAYER" + str(self.index) + " TURN === \n Enter pattern or type 's' to see hand 'p' for pattern, \n \t 'c' for cards count, 'd' for cards played"
        while True:
            act = raw_input().strip()
            if act == 's':
                print strHand(hand)
            elif act == 'p':
                print str(table.lowcard) + ' , ' + str(table.pattern)
            elif act == 'c':
                for i in range(len(table.AgentCardCount)):
                    if self.index == i:
                        print 'You have', table.AgentCardCount[i], 'cards'
                    else:
                        print 'The Computer has', table.AgentCardCount[i], 'cards'
            elif act == 'd':
                print strHand(table.cards)
            elif act == 'v':
                t = minusDic(hand, table.cards)
                print 'Cards not played and not in your hand: '
                print strHand(t)
                print 'total : ', sum(t[key] for key in t)
            else:
                move = moveStringToTuple(act)
                if move != -1:
                    if self.isMoveLegal(hand, move, table.lowcard, table.pattern):
                        return move
                    else:
                        print 'illegal move'
                else:
                    print 'invalid syntax'

    # Specify a move as Lowest_card, [# of lowest card, # of next card, and so on]
    # e.g. to play 10,10,J,J,Q,Q, move would be "(10, [2,2,2])"
    # currently only checks if hand has the cards, needs to check if pattern
    # matches the pattern in play, and if it has a higher value
    def isMoveLegal(self, hand, move, lowcard, pattern):
        if move is None:
            return True
        legal = True
        if lowcard == 0:
            if len(move[1]) != 1:
                for val in move[1]:
                    legal &= val >= 2
        else:
            legal = (pattern == move[1] and lowcard < move[0]) or move[1] == [4]
        for i, val in enumerate(move[1]):
            legal &= hand[move[0] + i] >= val
        return legal