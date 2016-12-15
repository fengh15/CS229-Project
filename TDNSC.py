from TD2 import *
from multiprocessing import Pool
from math import sqrt
import cPickle
import sys

def loadtds():
    fname = raw_input('Enter file name (or [enter] for default): ').strip()
    if len(fname) == 0:
        fname = 'TDs.pickle'
    f = open(fname)
    tds = cPickle.load(f)
    return tds

def main(argv):
    tds = loadtds()
    td = tds[0]
    while True:
        ans = raw_input('(P)lay, (T)rain, (S)ave, or (E)xit? ').strip().lower()
        if ans == 'p':
            ans = raw_input('Against (r)andom, (g)reedy, or (h)uman? ').strip().lower()
            num = int(raw_input('How many games? ').strip())
            if num != 0:
                wins = 0
                if ans == 'r':
                    wins = td.playTDvR(num)
                    rate = 1.0*wins/num
                if ans == 'g':
                    wins = td.playTDvG(num)
                    rate = 1.0*wins/num
                if ans == 'h':
                    wins = td.playTDvH(num)
                    rate = 1.0*wins/num
                print('TD won %.3f%%\t(%d games out of %d)' % (100*rate, wins, num))
                if num != 1:
                    err = 200.0*sqrt(rate*(1-rate)/(num*(num-1)))
                    low = 100*(rate - err)
                    high = 100*(rate + err)
                    print('True win rate in [%.3f%%, %.3f%%]' % (low, high))
        elif ans == 't':
            ans = raw_input('Against (s)elf, (r)andom, (g)reedy, or (h)uman? ').strip().lower()
            num = int(raw_input('How many games? ').strip())
            if ans == 'r':
                td.trainR(num)
            if ans == 'g':
                td.trainG(num)
            if ans == 'h':
                td.trainH(num)
            if ans == 's':
                td.train(num)
            print 'Agent has trained a total of ', td.iteration, ' times'
        elif ans == 's':
            fname = raw_input('Enter file name (or [enter] for default): ').strip()
            if len(fname) == 0:
                fname = 'TDs.pickle'
            f = open(fname, 'w')
            cPickle.dump(tds, f)
        elif ans == 'e':
            break
    return

if __name__ == "__main__": main(sys.argv)