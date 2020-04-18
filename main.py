import numpy as np
import random
import math
import matplotlib.pyplot as plt
from cmath import exp, pi


AMPLITUDE_MAX = 1
FULL_CIRCLE = 2 * math.pi

# variant 10
HARMONICS = 14
TICKS = 64
FREQUENCY = 1700


def random_signal(harmonics, ticks, freq):
    generated_signal = np.zeros(ticks)
    for i in range(harmonics):
        fi = FULL_CIRCLE * random.random()
        amplitude = AMPLITUDE_MAX * random.random()
        w = freq - i * freq / harmonics

        x = amplitude * np.sin(np.arange(0, ticks, 1) * w + fi)
        generated_signal += x
    return generated_signal


def fft(x):
    n = len(x)
    if n <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [exp(-2j*pi*k/n)*odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + \
           [even[k] - T[k] for k in range(n//2)]


if __name__ == '__main__':
    random.seed(10)
    x_line = [i for i in range(TICKS)]
    sig = random_signal(HARMONICS, TICKS, FREQUENCY)

    transformed = fft(sig)

    plt.subplot(311)
    p1 = plt.plot(x_line, sig, label='Random signal')
    plt.legend(handles=p1)

    plt.subplot(312)
    plt.title('FFT real')
    p2 = plt.stem(x_line, np.real(transformed), use_line_collection=True)

    plt.subplot(313)
    plt.title('FFT imag')
    p3 = plt.stem(x_line, np.imag(transformed), use_line_collection=True)

    plt.show()




