from TD4 import *
import sys
import cPickle

def playtest(td, numTest):
    print ''
    print 'Play versus Random: '+str(100.0*td.playTDvR(numTest)/numTest)+'%'
    print 'Play versus Greedy: '+str(100.0*td.playTDvG(numTest)/numTest)+'%'

def vTD(td, numTrain, numTest):
    print 'Reseting weights...'
    td.reset()
    print 'Training against self'
    td.train(numTrain)
    playtest(td, numTest)
    saveWeights(td.weights, 'TDvTD.txt')

def vGr(td, numTrain, numTest):
    print 'Reseting weights...'
    td.reset()
    print 'Training against greedy'
    td.trainG(numTrain)
    playtest(td, numTest)
    saveWeights(td.weights, 'TDvGr.txt')

def vRd(td, numTrain, numTest):
    print 'Reseting weights...'
    td.reset()
    print 'Training against random'
    td.trainR(numTrain)
    playtest(td, numTest)
    saveWeights(td.weights, 'TDvGr.txt')

def vGTD(td, numTrain, numTest):
    print 'Reseting weights...'
    td.reset()
    print 'Training against greedy'
    td.trainG(numTrain / 2)
    print 'Training against self'
    td.train(numTrain / 2)
    playtest(td, numTest)
    saveWeights(td.weights, 'TDvGTD.txt')

def vRTD(td, numTrain, numTest):
    print 'Reseting weights...'
    td.reset()
    print 'Training against random'
    td.trainR(numTrain / 2)
    print 'Training against self'
    td.train(numTrain / 2)
    playtest(td, numTest)
    saveWeights(td.weights, 'TDvRTD.txt')

def play(fname = 'TDNSAlgs.pickle'):
    f1 = open(fname)
    tds1 = cPickle.load(f1)

def main(argv):
    numTrain = int(argv[1])
    numTest = int(argv[2])

    td = TDAlgorithm(True)
    print 'Untrained'
    playtest(td, numTest)
    print '(Random vs Greedy): '+str(100.0*td.playRvG(numTest)/numTest)+'%'
    print '\n'
    vTD(td, numTrain, numTest)
    print '\n'
    vGr(td, numTrain, numTest)
    print '\n'
    vRd(td, numTrain, numTest)
    print '\n'
    vGTD(td, numTrain, numTest)
    print '\n'
    vRTD(td, numTrain, numTest)
    return



if __name__ == "__main__": main(sys.argv)