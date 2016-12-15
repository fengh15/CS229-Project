import math
import random as rd
from collections import defaultdict
from multiprocessing import Pool
from TDUtil import *
from TD import tableClass, Agent, computerAgent, RandomAgent,GreedyAgent, humanAgent

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

def fE2(hand, table, index, move, possibleHand, oppHasPassed):
    minusMove(hand, move)
    patlen = 0 if move is None else sum(move[1])
    feats = fE1(hand, table, index, patlen, 2)
    #add additional features
    cP = counterProb(table.AgentCardCount[index-1], possibleHand, move)
    # print cP
    feats.append(((2, 'CounterProb'), cP))
    plusMove(hand, move)
    return feats

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
        return history

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
                if self.iteration%100 == 0:
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
            if self.verbose:
                if i%100 == 0:
                    print "Game: " + str(i)
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

class TDAgent(computerAgent):
    def __init__(self, index, numPlayers, preFE, postFE, weights, explorationProb):
        computerAgent.__init__(self, index, numPlayers)
        self.preFE = preFE
        self.postFE = postFE
        self.weights = weights
        self.explorationProb = explorationProb
        self.history.append("TD")
        self.lastMove = None
        self.oppHasPassed = defaultdict(int) # dictionary of moves opp has passed on

    def evalFeats(self, feats):
        value = 0
        for feat, val in feats:
            value += self.weights[feat] * val
        return value


    def getMove(self, hand, table): #get move, add features to history
        moves = self.getLegalMoves(hand, table.lowcard, table.pattern)
        if rd.uniform(0, 1) < self.explorationProb:
            return rd.choice(moves)
        else:
            if len(table.pattern) == 0 and self.lastMove is not None:
                if tuple(self.lastMove[1]) not in self.oppHasPassed:
                    self.oppHasPassed[tuple(self.lastMove[1])] = self.lastMove[0]
                else:
                    self.oppHasPassed[tuple(self.lastMove[1])] = min(self.oppHasPassed[tuple(self.lastMove[1])], self.lastMove[0])
            f1 = self.preFE(hand, table, self.index)
            possibleHand = minusDic(table.cards, hand)
            f2s = [self.postFE(hand, table, self.index, move, possibleHand, self.oppHasPassed) for move in moves]
            valAndMoves = [(self.evalFeats(f2s[i]), moves[i]) for i in range(len(moves))]
            maxVal = max(valAndMoves)[0]
            maxMoves = []
            for v, m in valAndMoves:
                if v == maxVal:
                    maxMoves.append(m)
            move = rd.choice(maxMoves)
            self.lastMove = move
            f2 = self.postFE(hand, table, self.index, move, possibleHand, self.oppHasPassed)
            self.history.append((f1, f2))
            return move
