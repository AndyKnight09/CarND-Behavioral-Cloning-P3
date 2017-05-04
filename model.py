import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

log_directory = 'data/'

# Hyperparameters
EPOCHS = 6
LEARNING_RATE = 0.001
BATCH_SIZE = 32

SIDE_CAMERA_STEERING_OFFSET = 0.3;
DROPOUT_RATE = 0.2

steering_correction = [0, +SIDE_CAMERA_STEERING_OFFSET, -SIDE_CAMERA_STEERING_OFFSET]

image_shape = (160,320,1)

def preprocess(image):
    return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=3)

def flip_horizontal(image):
    return np.expand_dims(cv2.flip(image,1), axis=3)
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                steer_angle = float(batch_sample[3])
                for iCamera in range(3):
                    # Parse image
                    source_path = batch_sample[iCamera].strip()
                    filename = source_path.split('\\')[-1]
                    image_path = log_directory + 'IMG/' + filename
                    image = cv2.imread(image_path)
                    
                    # Apply prepocessing
                    image = preprocess(image)
                    
                    images.append(image)

                    # Parse steer angle
                    angle = steer_angle + steering_correction[iCamera]
                    angles.append(angle)
                    
                    # Augment data with flipped images
                    images.append(flip_horizontal(image))
                    angles.append(-angle)
                    
                    # TODO: Augment data set using other techniques - brightness, translation?
                    
            # Return shuffled training set
            X = np.array(images)
            y = np.array(angles)
            
            yield shuffle(X, y)

def lenet_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=image_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=image_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(100))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(50))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    
    # Read samples from log data
    samples = []
    with open(log_directory + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        
        # skip header row
        next(reader, None)
        
        # append all other rows
        for line in reader:
            samples.append(line)
        
    # Create training and validation set
    train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
    
    print(image_shape)
    
    # Compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    # Create model    
    model = nvidia_model()

    # Model loss and optimization functions
    opt = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    # Train model
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, validation_data=validation_generator, validation_steps=len(validation_samples)/BATCH_SIZE, epochs=EPOCHS, verbose=1)

    # Save model
    model.save('model.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
