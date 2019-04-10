import pandas as pd
import numpy as np
from keras import applications
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, GlobalAveragePooling2D
import keras
import argparse
import os

np.random.seed(0)

args_name_lst=['TimeStamp','POS_X','POS_Y','POS_Z','Q_W','Q_X','Q_Y','Q_Z','Throttle','Steering','Brake','Gear','Handbrake','RPM','Speed','center','right','left']

def load_data(args):
    data_df = pd.read_csv(os.path.join('./data/merge.csv'), names=args_name_lst)
    X = data_df['center'].values
    y = data_df['Steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(data_df, y, test_size=args.test_size, random_state=0)
    y_train = y_train.astype(np.float)
    y_valid = y_valid.astype(np.float)
    '''
    if not os.path.exists('./train'):
        os.makedirs('./train')
    if not os.path.exists('./valid'):
        os.makedirs('./valid')
    for i in X_train:
        shutil.copyfile('./images/%s'%(i), './train/%s'%(i))
    for i in X_valid:
        shutil.copyfile('./images/%s'%(i), './valid/%s'%(i))
    '''
    return data_df, X_train, X_valid, y_train, y_valid

def build_model(args):
    #base_model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (144, 256, 3))
    base_model= applications.InceptionV3(weights='imagenet',include_top=False, input_shape = (144, 256, 3))
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(100,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(50,activation='relu')(x) #dense layer 2
    x=Dense(10,activation='relu')(x) #dense layer 3
    preds=Dense(1)(x) #final layer with softmax activation
    model= Model(input = base_model.input, output = preds)
    
    model.summary()
    return model


def train_model(model, args, data_df, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    
    
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    print(X_train.shape, X_valid.shape)
    print(X_train['center'])
    train_generator = train_datagen.flow_from_dataframe(X_train, directory='./images',
        x_col='center',
        y_col='Steering',
        target_size=(144, 256),
        batch_size=args.batch_size, class_mode='other'
        )
    test_generator = test_datagen.flow_from_dataframe(X_valid, directory='./images',
        x_col='center',
        y_col='Steering',
        target_size=(144, 256),
        batch_size=args.batch_size, class_mode='other'
        )
    #validation_generator = test_datagen.flow(X_valid, y=y_valid, batch_size=args.batch_size)

    # fine-tune the model
    
    model.fit_generator(train_generator,
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=test_generator,
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)
'''
    model.fit(x=X_train, y=y_train, batch_size=args.batch_size, epochs=args.nb_epoch, 
        verbose=1, callbacks=[checkpoint],
        validation_data=(X_valid, y_valid),
        steps_per_epoch=args.samples_per_epoch)
'''

def s2b(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',dest='data_dir', type=str, default='images')
    parser.add_argument('-t', help='test size fraction',dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',dest='nb_epoch', type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',dest='samples_per_epoch', type=int, default=250)
    parser.add_argument('-b', help='batch size',dest='batch_size', type=int, default=100)
    parser.add_argument('-o', help='save best models only',dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate',dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

