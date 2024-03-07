import numpy as np
import matplotlib.pyplot as plt

class utils:
    """
    Generate a set of random temperatures using a sinusoidal signal and noise
    """
    @staticmethod
    def genSinTemperatures(num_samples, min_temp, max_temp, frequency, noise_std, plot=False):
        # generate a Time vector
        time = np.linspace(0, 100, num_samples) 

        # generate sinusoidal signal
        #frequency = 0.2 # 0.05 # frequency in Hz
        amplitude = (max_temp - min_temp) / 2 # half of temperature range
        offset = (max_temp + min_temp) / 2 # mid-range to shift signal along y 
        signal = amplitude * np.sin(2 * np.pi * frequency * time) + offset
        # print(signal)

        # generate random noise
        noise = np.random.normal(-noise_std, noise_std, num_samples)  # mean=0, std=noise_std
        #noise = np.random.uniform(low=0, high=2, size=num_samples)
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

        return temperature
    
    """
    Prepare data for Long short-term memory network: 
    samples are made of previous Nsteps values, 
    labels are made of next step value
    return tuple (samples, labels)
    """
    @staticmethod
    def prepareLSTN_Data(data, Nsteps=1):
        prevNsteps_samples = []
        nextStep_labels = []
        for i in range(len(data) - Nsteps):
            prevNsteps_samples.append(data[i:i+Nsteps])
            nextStep_labels.append(data[i+Nsteps])

        # list to nparray
        X = np.array(prevNsteps_samples)
        y = np.array(nextStep_labels)

        # verify we prepared data correctly
        assert(len(X) == len(y) == (len(data) - Nsteps))

        return (X, y)
    
    """
    Split labelled training data into training data and testing data
    """
    @staticmethod
    def splitTrainingData(X, y, testingPercentage=.15):
        assert(len(X) == len(y)) # verify num samples = num lables

        dataSize = len(X)
        testSize = int(dataSize * testingPercentage)
        trainSize = dataSize - testSize
        X_train, y_train = X[0:trainSize], y[0:trainSize]
        X_test, y_test = X[trainSize:dataSize], y[trainSize:dataSize]

        return (X_train, y_train, X_test, y_test)
