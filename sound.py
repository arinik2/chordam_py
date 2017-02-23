import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from array import array
from struct import pack
from sys import byteorder
import copy
import pyaudio
import wave
import argparse
import sys
from matplotlib import animation

DATA = array('h')
THRESHOLD = 1000  # audio levels not normalised.
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 44100
CHUNK_SIZE = RATE/int(sys.argv[3])
TIME_IN_CHUNKS = int(sys.argv[1])*int(sys.argv[3])
CHANNELS = 1
TRIM_APPEND = RATE / 4
TUNE = []
TIME = []
#
# def is_silent(data_chunk):
#     """Returns 'True' if below the 'silent' threshold"""
#     return max(data_chunk) < THRESHOLD
#
# def normalize(data_all):
#     """Amplify the volume out to max -1dB"""
#     # MAXIMUM = 16384
#     normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
#                         / max(abs(i) for i in data_all))
#
#     r = array('h')
#     for i in data_all:
#         r.append(int(i * normalize_factor))
#     return r
#
# def trim(data_all):
#     _from = 0
#     _to = len(data_all) - 1
#     for i, b in enumerate(data_all):
#         if abs(b) > THRESHOLD:
#             _from = max(0, i - TRIM_APPEND)
#             break
#
#     for i, b in enumerate(reversed(data_all)):
#         if abs(b) > THRESHOLD:
#             _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
#             break
#
#     return copy.deepcopy(data_all[_from:(_to + 1)])
# def plot(array):
#
#     plt.plot(array)
#     plt.show()
#
# def record():
#     """Record a word or words from the microphone and
#     return the data as an array of signed shorts."""
#
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)
#
#     silent_chunks = 0
#     audio_started = False
#     data_all = array('h')
#     i = 0
#     while True:
#         # little endian, signed short
#         i += 1
#         data_chunk = array('h', stream.read(CHUNK_SIZE))
#         if byteorder == 'big':
#             data_chunk.byteswap()
#         data_all.extend(data_chunk)
#         silent = is_silent(data_chunk)
#         if i>10000:break
#         if audio_started:
#             if silent:
#                 silent_chunks += 1
#                 if silent_chunks > SILENT_CHUNKS:
#                     break
#             else:
#                 silent_chunks = 0
#         elif not silent:
#             audio_started = True
#             print "Audio started"
#
#     sample_width = p.get_sample_size(FORMAT)
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
#     data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
#     data_all = normalize(data_all)
#
#     return sample_width, data_all
class window(object):
    def gausse(self,n,frameSize):
        a = (frameSize - 1)/2
        t = (n - a)/(0.5*a)
        t = t*t
        return np.exp(-t/2)
    def Hamming(self,n,frameSize):
        return 0.54 - 0.46*np.cos((2*np.pi*n)/(frameSize - 1))

def noise(array):
    sarray = sorted(array)
    x = []
    y = []
    der = []
    k = -1
    for i in xrange(len(array)):
        if i%100 == 0:
            y.append(i)
            x.append(sarray[i])
            k += 1
            if i == 0: continue
            der.append(100/(x[k]-x[k-1]))
    plt.plot(x[:-1],der)
    plt.show()
    return x[der.index(max(der))]*3

def record_to_file(path,sample_width,data):
    "Records from the microphone and outputs the resulting data to 'path'"

    data = pack('<' + ('h' * len(data)), *data)

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()
def clear(array):
    half = max(array)/2
    count = 0
    for i in xrange(len(array)-1):
        if (array[i]-half)*(array[i+1]-half)<0:count+=1
    return count
def exportNote(note,time):
    hertz = note*RATE/CHUNK_SIZE
    TUNE.append(hertz)
    TIME.append(time)
    n = 12*np.log2(hertz/440) + 9
    print hertz,time,n

# def init():
#     line.set_ydata(np.ma.array(x, mask=True))
#     return line,
# # def fourier(array):
# #     np.fft.rfft(array)*hamming(len(array)
# def animate(i):
#         data_chunk = array('h', stream.read(CHUNK_SIZE))
#         if byteorder == 'big':
#             data_chunk.byteswap()
#         line.set_ydata(np.fft.fft(data_chunk)*hamming(len(data_chunk))) #np.fft.rfft(data_chunk) * hamming(513)
#         return line,
if __name__ == '__main__':
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)
    fig = plt.figure()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p')
    namespace = parser.parse_args(sys.argv[4:])
    path = sys.argv[2]+'_'+sys.argv[1]+'s.dat'
    if namespace.p:
        path = namespace.p+sys.argv[2]+'_'+sys.argv[1]+'s.dat'
    # ax = plt.axes(xlim=(0, CHUNK_SIZE), ylim=(-10000, 10000))
    # x = np.arange(CHUNK_SIZE)
    # line, = ax.plot(x, np.sin(x))
    # ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1,
    #                               interval=25, blit=True)
    # plt.show()
    data_all,i,notes,count,sumFFT,toPlot = array('h'),0,[0 for c in xrange(CHUNK_SIZE/2)],0,np.array([0 for c in xrange(CHUNK_SIZE/2+1)]),[]
    f = open(path,'w')
    # sine = 30*np.sin(np.arange(4410)/4.0)
    # win = window()
    print 'do it'
    while i<TIME_IN_CHUNKS:

        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)
        i+=1
        # for k in xrange(CHUNK_SIZE):
        #     data_chunk[k] = int(data_chunk[k]*win.Hamming(k,CHUNK_SIZE))
        dft = np.absolute(np.fft.rfft(data_chunk)).tolist()
        for k in xrange(len(dft)):
            f.write(str(i*CHUNK_SIZE)+'\t'+str(dft[k])+'\t'+str(k*int(sys.argv[3]))+'\n')
        f.write('\n')
    f.close()

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    record_to_file('demo.wav',sample_width,data_all)


# fig = plt.figure()
# N = 60
# # sine = [0,0.707,1,0.707,0,-0.707,-1,0.38,0,-0.38,-0.707,-0.92,-1,-0.92,-0.707,-0.38,0]
# sine = np.sin(np.arange(N)/8.0)
# newsine = np.sin(np.arange(N)/2.0)
# plt.plot(newsine)
# plt.plot(sine)
# # plt.plot(sine*newsine)
# plt.plot(np.absolute(np.fft.rfft(sine*newsine)))
# # sm1 = 0
# # sm2 = 0
# #
# # for k in xrange(N):
# #     sm1 = 0
# #     sm2 = 0
# #     for i in xrange(N):
# #         sm1 += sine[i]*np.sin((2*np.pi*k*i)/N)
# #         sm2 += sine[i]*np.cos((2*np.pi*k*i)/N)
# #     print sm2,sm1
# plt.show()