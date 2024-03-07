import numpy as np
import matplotlib.pyplot as plt

"""
Generate a set of random temperatures using a sinusoidal signal and noise
"""
def genSinTemperatures(num_samples, min_temp, max_temp, frequency, noise_std, plot=False):
    # generate a Time vector
    time = np.linspace(0, 100, num_samples) 

    # generate sinusoidal signal
    amplitude = (max_temp - min_temp) / 2 # half of temperature range
    offset = (max_temp + min_temp) / 2 # mid-range to shift signal along y 
    signal = amplitude * np.sin(2 * np.pi * frequency * time) + offset
    # print(signal)

    # generate random noise
    noise = np.random.normal(-noise_std, noise_std, num_samples)  # mean=0, std=noise_std
    # print(noise)

    # combine sinusoidal signal with noise
    temperature = signal + noise

    # plot data 
    if plot:
        plt.plot(time, temperature, label='Temperature')
        plt.title('Sinusoidal Random Temperatures')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.grid(True)
        plt.show()

    return (time, temperature)


###### generate random temperatures
data = genSinTemperatures(num_samples=500, min_temp=25, max_temp=35, \
                          frequency=0.3, noise_std=1.5, plot=True)
temperatures = data[1]

# print(data[0], len(data[0]))
# print(temperatures)