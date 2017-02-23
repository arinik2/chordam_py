from numpy import *
import numpy as np
from array import array
from sys import byteorder
import pyaudio
import sys


FORMAT = pyaudio.paInt16
RATE = 44100
CHUNK_SIZE = RATE/int(sys.argv[1])
CHANNELS = 1

def readf(file,strs):
    data = []
    f = file.readlines()
    for i in xrange(strs):
        data.append(float(f[i][:-1]))
    return data

if __name__ == '__main__':
    i = 0
    f = open(sys.argv[2])
    print 'do it'
    while i<len(f.readlines())/RATE*int(sys.argv[1]):
        data_chunk = readf(f,CHUNK_SIZE)
        i+=1
        dft = np.absolute(np.fft.rfft(data_chunk)).tolist()
        for k in xrange(len(dft)):
            print str(i*CHUNK_SIZE/float(RATE))+'\t'+str(dft[k])+'\t'+str(k*int(sys.argv[1]))



