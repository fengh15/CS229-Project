from TD4 import *
from multiprocessing import Pool
import cPickle
import sys

def playtest(td, numTest):
    print ''
    print 'Play versus Random: '+str(100.0*td.playTDvR(numTest)/numTest)+'%'
    print 'Play versus Greedy: '+str(100.0*td.playTDvG(numTest)/numTest)+'%'

def tdV((td, numTest)):
    return 100.0*td.playTDvG(numTest)/numTest, td

def tr((td, x)):
    td.train(x)
    return td

def main(argv):
    tds = [TDAlgorithm(False) for i in range(50)]
    p = Pool(3)
    print 'training 50'
    tds = p.map_async(tr, [(td, 5000) for td in tds]).get(9999999)
    print 'testing 50'
    tdVs = p.map_async(tdV, [(td, 5000) for td in tds]).get(9999999)
    tdVs.sort(key=lambda el: -el[0])
    for i in range(40):
        tdVs.pop()
    tds = [td for v, td in tdVs]

    print 'training 10'
    tds = p.map_async(tr, [(td, 25000) for td in tds]).get(9999999)
    print 'testing 10'
    tdVs = p.map_async(tdV, [(td, 25000) for td in tds]).get(9999999)
    tdVs.sort(key=lambda el: -el[0])
    for i in range(7):
        tdVs.pop()
    tds = [td for v, td in tdVs]

    print 'training 3'
    tds = p.map_async(tr, [(td, 70000) for td in tds]).get(9999999)
    print 'testing 3'
    tdVs = p.map_async(tdV, [(td, 70000) for td in tds]).get(9999999)
    tdVs.sort(key=lambda el: -el[0])
    tds = [td for v, td in tdVs]
    for td in tds:
        playtest(td, 10000)
    print tds
    f = open('TDs.pickle', 'w')
    cPickle.dump(tds, f)

    return



if __name__ == "__main__": main(sys.argv)