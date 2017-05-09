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

# Directory containing recorded simulation data
log_directory = 'data/'

# Hyperparameters used to tune the model
EPOCHS = 8                          # Number of training epochs
LEARNING_RATE = 0.0015              # Learning rate 
BATCH_SIZE = 32                     # Batch size (before taking into account left/centre/right cameras plus horizontally mirrored images)

SIDE_CAMERA_STEERING_OFFSET = 0.3;  # Steering correction to apply to left/right camera images
DROPOUT_RATE = 0.2                  # Fraction of input units to dropout for each dropout layer

# Form a list of steering corrections for center, left and right camera images
steering_correction = [0, +SIDE_CAMERA_STEERING_OFFSET, -SIDE_CAMERA_STEERING_OFFSET]

# Shape of the input images (once converted to grayscale)
image_shape = (160,320,1)

def preprocess(image):
    '''
    returns a grayscale version of the input image
    '''
    return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), axis=3)

def flip_horizontal(image):
    '''
    returns a horizontally mirrored version of the image
    '''
    return np.expand_dims(cv2.flip(image,1), axis=3)
    
def generator(samples, batch_size=32):
    '''
    returns batches of images as required to reduce memory overhead
    '''
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

                    # Parse steer angle
                    angle = steer_angle + steering_correction[iCamera]
                    
                    # Apply prepocessing to image
                    image = preprocess(image)
                    
                    # Add preprocessed image and steer angle to data set
                    images.append(image)
                    angles.append(angle)
                    
                    # Augment data set with horizontally flipped image and negated steer angle
                    images.append(flip_horizontal(image))
                    angles.append(-angle)
                    
            # Return shuffled training set
            X = np.array(images)
            y = np.array(angles)
            
            yield shuffle(X, y)

def lenet_model():
    '''
    return LeNet-based model architecture
    '''
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
    '''
    return Nvidea-based model architecture
    '''
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

    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
