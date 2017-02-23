from numpy import *
import numpy as np
from array import array
from sys import byteorder
import pyaudio
import sys


FORMAT = pyaudio.paInt16
RATE = 44100
CHUNK_SIZE = RATE/int(sys.argv[3])
TIME_IN_CHUNKS = int(sys.argv[1])*int(sys.argv[3])
CHANNELS = 1

if __name__ == '__main__':
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    data_all,i,notes,count,sumFFT,toPlot = array('h'),0,[0 for c in xrange(CHUNK_SIZE/2)],0,np.array([0 for c in xrange(CHUNK_SIZE/2+1)]),[]

    print 'do it'
    while i<TIME_IN_CHUNKS:

        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)
        i+=1
        # for k in xrange(CHUNK_SIZE):
        #     data_chunk[k] = int(data_chunk[k]*win.Hamming(k,CHUNK_SIZE))
        # dft = np.absolute(np.fft.rfft(data_chunk)).tolist()
        # for k in xrange(len(dft)):
        #     print str(i*CHUNK_SIZE/float(RATE))+'\t'+str(dft[k])+'\t'+str(k*int(sys.argv[3]))
    for x in data_all:print x
    stream.stop_stream()
    stream.close()
    p.terminate()


