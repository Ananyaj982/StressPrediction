import warnings
warnings.filterwarnings('ignore')
# IMPORT NECESSARY LIBRARIES
import csv
from datetime import datetime

import tensorflow.keras as keras
#doverbose=0 to stop printing model stuff
import librosa
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os # interface with underlying OS that python is running on
import sys
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.optimizers import SGD
from keras.regularizers import l2
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import keras
import keras.utils
#from keras import utils as np_utils
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

from pydub import AudioSegment
import math
flag=0
x1=[]
my_dict=dict()
stuff_list=[]
def fn(audiop1,i):
    k=[]
    class SplitWavAudioMubin():
        def __init__(self, folder, filename):
            self.folder = folder
            self.filename = filename
            self.filepath = folder + '\\' + filename

            self.audio = AudioSegment.from_wav(self.filepath)

        def get_duration(self):
            return self.audio.duration_seconds

        def single_split(self, from_min, to_min, split_filename):
            t1 = from_min * 3 * 1000
            t2 = to_min * 3 * 1000
            split_audio = self.audio[t1:t2]
            split_audio.export(self.folder + '\\' + split_filename, format="wav")

        def multiple_split(self, min_per_split):
            total_mins = math.floor(self.get_duration() / 3)
            for i in range(0, total_mins, min_per_split):
                split_fn = str(i) + '_' + self.filename
                self.single_split(i, i+min_per_split, split_fn)
                if(i!=i+min_per_split):
                    k.append(split_fn)
                #print("$$$$$$$$",split_fn)
                #print(str(i) + ' Done')
                if i == total_mins - min_per_split:
                    print('All splited successfully')
    folder = audiop1
    file = i
    split_wav = SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=1)
    #print(k)
    return k

def melstuff(audio_df):
    df = pd.DataFrame(columns=['mel_spectrogram'])
    #print("xy",audio_df.path)
    counter=0

    for index,path in enumerate(audio_df.path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)

        #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        #temporally average spectrogram
        log_spectrogram = np.mean(db_spec, axis = 0)

        # Mel-frequency cepstral coefficients (MFCCs)
    #     mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    #     mfcc=np.mean(mfcc,axis=0)

        # compute chroma energy (pertains to 12 different pitch classes)
    #     chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
    #     chroma = np.mean(chroma, axis = 0)

        # compute spectral contrast
    #     contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
    #     contrast = np.mean(contrast, axis= 0)

        # compute zero-crossing-rate (zcr:the zcr is the rate of sign changes along a signal i.e.m the rate at
    #     which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
    #     zcr = librosa.feature.zero_crossing_rate(y=X)
    #     zcr = np.mean(zcr, axis= 0)

        df.loc[counter] = [log_spectrogram]
        counter=counter+1
    return df
    #print(len(df))

def melstuff1(df1):
    df2 = pd.DataFrame(columns=['mel_spectrogram'])
    #print("xy",df1.path)
    counter=0

    for index,path in enumerate(df1.path):
        #X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=6,sr=44100,offset=0.5)
        #X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=5,sr=44100,offset=0.5)
        #X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=4,sr=44100,offset=0.5)
        X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)

        #get the mel-scaled spectrogram (ransform both the y-axis (frequency) to log scale, and the “color” axis (amplitude) to Decibels, which is kinda the log scale of amplitudes.)
        spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        #temporally average spectrogram
        log_spectrogram = np.mean(db_spec, axis = 0)

        # Mel-frequency cepstral coefficients (MFCCs)
    #     mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    #     mfcc=np.mean(mfcc,axis=0)

        # compute chroma energy (pertains to 12 different pitch classes)
    #     chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
    #     chroma = np.mean(chroma, axis = 0)

        # compute spectral contrast
    #     contrast = librosa.feature.spectral_contrast(y=X, sr=sample_rate)
    #     contrast = np.mean(contrast, axis= 0)

        # compute zero-crossing-rate (zcr:the zcr is the rate of sign changes along a signal i.e.m the rate at
    #     which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
    #     zcr = librosa.feature.zero_crossing_rate(y=X)
    #     zcr = np.mean(zcr, axis= 0)

        df2.loc[counter] = [log_spectrogram]
        counter=counter+1
    return df2



def train_new(df1):

    df2=melstuff1(df1)
    #print("bye,",len(df1))
    lb = x1[2]
    # CHECK TOP 5 ROWS
    #print("start",df_combined.head())
    df_combined_new = pd.concat([df1,pd.DataFrame(df2['mel_spectrogram'].values.tolist())],axis=1)
    df_combined_new = df_combined_new.fillna(0)
    # DROP PATH COLUMN FOR MODELING
    df_combined_new.drop(columns='path',inplace=True)
    # CHECK TOP 5 ROWS
    #df_combined_new.head()
    #print('--------------')
    #print(df_combined_new)
    mean=x1[0]
    std=x1[1]
    train1=df_combined_new
    X_test = train1.iloc[:,1:]
    X_test = (X_test - mean)/std
    X_test = np.array(X_test)
    X_test = X_test[:,:,np.newaxis]
    model=load_model('D://sem6//capstone//model//model1.h5')

    predictions = model.predict(X_test)
    predict_classes=np.argmax(predictions,axis=1)
    #print("YO PROBABILITY1 IS",predict_classes)
    #print("YO PROBABILITY2 IS",predictions)
    hj=[]
    hj.append(predictions)
    stuff_list.append(predictions)
    predictions=predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform((predictions)))
    predictions = pd.DataFrame({'Predicted Values': predictions})

    hj.append(predictions)
    #hj.append(predict_classes)
    return hj


def normalize(X_train,y_train,X_test,X_test_1,y_test_1):
    mean = np.mean(X_train, axis=0)

    std = np.std(X_train, axis=0)
    x1.append(mean)
    x1.append(std)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    X_test_1 = (X_test_1 - mean)/std
    # TURN DATA INTO ARRAYS FOR KERAS
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    X_test_1 = np.array(X_test_1)
    y_test_1=np.array(y_test_1)
    #y_test = np.array(y_test)
    # ONE HOT ENCODE THE TARGET
    # CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
    lb = LabelEncoder()
    y_train = to_categorical(lb.fit_transform(y_train))
    x1.append(lb)
    #y_test = to_categorical(lb.fit_transform(y_test))
    y_test_1 = to_categorical(lb.fit_transform(y_test_1))

    #print(y_test[0:10])
    #print(lb.classes_)

    # RESHAPE DATA TO INCLUDE 3D TENSOR use this for CNN and comment for the other models
    X_train = X_train[:,:,np.newaxis]
    X_test = X_test[:,:,np.newaxis]
    X_test_1 = X_test_1[:,:,np.newaxis]
    '''
    print("X_train.shape",X_train.shape)
    print(X_train)
    print('------------------------------------------------')
    print("y_train.shape",y_train.shape)
    print(y_train)
    print('************************************************')
    print("X_test.shape",X_test.shape)
    print(X_test)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    #print("y_test.shape",y_test.shape)
    #print(y_test)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
    model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(layers.MaxPooling1D(pool_size=(2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=(2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(2, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    #print(model.summary())
    checkpoint = ModelCheckpoint("best_initial_model.hdf5", monitor='val_accuracy', verbose=0,
    save_best_only=True, mode='max', period=1, save_weights_only=True)

    model_history=model.fit(X_train, y_train,batch_size=32, epochs=70, validation_data=(X_test_1, y_test_1),callbacks=[checkpoint],verbose=0)
    # assign location
    #path='D://sem6//capstone//calls'

    model.save('model1.h5')
    #print("setting flag to 1")
    '''
    flag=1
    #train1(X_test)
    #model_history=model.fit(X_train, y_train,batch_size=32, epochs=100,callbacks=[checkpoint])
    model1=load_model('model1.h5')

    predictions = model1.predict(X_test)
    predict_classes=np.argmax(predictions,axis=1)
    #print("YO PROBABILITY1 IS",predict_classes)
    #print("YO PROBABILITY2 IS",predictions)
    stuff_list.append(predictions)
    hj=[]
    hj.append(predictions)
    predictions=predictions.argmax(axis=1)
    predictions = predictions.astype(int).flatten()
    predictions = (lb.inverse_transform((predictions)))
    predictions = pd.DataFrame({'Predicted Values': predictions})

    hj.append(predictions)
    return hj
    '''
    # ACTUAL LABELS
    actual=y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))
    actual = pd.DataFrame({'Actual Values': actual})

    # COMBINE BOTH
    finaldf = actual.join(predictions)
    print(finaldf)

    '''
def train(df1):
    emotion = []
    gender = []
    actor = []
    file_path = []
    audio = "D://sem6//capstone//archive"
    actor_folders = os.listdir(audio) #list files in audio directory
    actor_folders.sort()

    os.chdir("D://sem6//capstone//archive")
    for i in actor_folders:
        try:
            #print("i is",i)
            filename = os.listdir(audio+"//"+ i) #iterate over Actor folders
            #print(filename)
            for f in filename: # go through files in Actor folder
                try:
                    part = f.split('.')[0].split('-')

                    x=part[2]

                    emotion.append(int(x))
                    actor.append(int(part[6]))
                    bg = int(part[6])
                    if bg%2 == 0:
                        bg = "female"
                    else:
                        bg = "male"
                    gender.append(bg)
                    file_path.append(audio + "//"+i + "//"+ f)
                except:
                    continue
        except:
            continue
    # PUT EXTRACTED LABELS WITH FILEPATH INTO DATAFRAME
    audio_df = pd.DataFrame(emotion)
    #surprise as stressed
    audio_df = audio_df.replace({1:'not stressed', 2:'not stressed', 3:'not stressed', 4:'stressed', 5:'stressed', 6:'stressed', 7:'stressed', 8:'stressed'})
    audio_df = pd.concat([pd.DataFrame(gender),audio_df,pd.DataFrame(actor)],axis=1)
    audio_df.columns = ['gender','emotion','actor']
    audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
    audio_df.to_csv('D://sem6//capstone//audio.csv',index=False)
    #print(audio_df)
    df=melstuff(audio_df)
    #print("hello",len(df))

    df2=melstuff1(df1)
    #print("bye,",len(df1))
    df_combined = pd.concat([audio_df,pd.DataFrame(df['mel_spectrogram'].values.tolist())],axis=1)
    df_combined = df_combined.fillna(0)
    # DROP PATH COLUMN FOR MODELING
    df_combined.drop(columns='path',inplace=True)
    # CHECK TOP 5 ROWS
    #print("start",df_combined.head())
    df_combined_new = pd.concat([df1,pd.DataFrame(df2['mel_spectrogram'].values.tolist())],axis=1)
    df_combined_new = df_combined_new.fillna(0)
    # DROP PATH COLUMN FOR MODELING
    df_combined_new.drop(columns='path',inplace=True)
    # CHECK TOP 5 ROWS
    df_combined_new.head()
    #print('--------------')
    #print(df_combined_new)
    train,test = train_test_split(df_combined, test_size=0.2, random_state=0,
                               stratify=df_combined[['emotion','gender','actor']])
    train1=df_combined_new
    X_train = train.iloc[:, 3:]
    y_train = train.iloc[:,:2].drop(columns=['gender'])
    #print(X_train.shape)
    #print(X_train.head())
    #print('--------------------------')
    #print(y_train.shape)
    #print(y_train.head())
    X_test_1=test.iloc[:,3:]
    y_test_1=test.iloc[:,:2].drop(columns=['gender'])
    X_test = train1.iloc[:,1:]
    #y_test = train1.iloc[:,:1]
    #print("yelp",X_test.shape)
    #print(X_test.head())
    #print('-----------------')
    #print(y_test.shape)
    #print(y_test)
    return normalize(X_train,y_train,X_test,X_test_1,y_test_1)

def main(username):
        flag=0
        #print("Enter folder name")
        x1=[]
        aj=[]
        #audiop1=input()
        audiop1="D://sem6//capstone//stuff1//check1"+"//"+username
        #audiop1 can be drive/folder name
        stuff1 = os.listdir(audiop1) #list files in audio directory
        #stuff1.sort(key=os.path.getmtime)
        #print(stuff1)
        #stuff1.sort(key=os.path.getmtime)
        #for file in sorted(stuff1,key=os.path.getmtime):
            #print(file)


        os.chdir(audiop1)

        k=sorted(filter(os.path.isfile, os.listdir('.')),key=os.path.getctime)
        #print("first, k is",k)
        for j in k:
            if ".wav" not in j:
                k.remove(j)
            else:
                my_dict[j]=""
        print("List of files is",k)
        if(len(k)==0):
            return -1
        for i in k:

            print("file is",i)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            fx="D://sem6//capstone//"+username+"audionew.csv"
            file = open(fx, 'w', newline ='') # Creating a new file called syn.csv
            header = ["path"]
            # Adding the appropriate headers
            # writing the data into the file
            with file:
                write = csv.writer(file, delimiter=',')
                write.writerow(header)
                file.close()


            file = open(fx, 'a', newline ='')
            with file:
                write = csv.writer(file)
                write.writerow([audiop1+"//"+str(i)])
                file.close()
            df1=pd.read_csv(fx)
            '''
            if(flag==0):
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("flag is 0,Current Time =", current_time)
                aj.append(train(df1))

                flag=1
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("After train Current Time =", current_time)
            else:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("flag is 1,Current Time =", current_time)

                aj.append(train_new(df1))
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("After train1 Current Time =", current_time)
            '''
            try:
                model1=load_model('D://sem6//capstone//model//model1.h5')
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("flag is 1,Current Time =", current_time)
                ghh=train_new(df1)
                my_dict[i]=ghh

                aj.append(my_dict[i])
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("After train1 Current Time =", current_time)
            except:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("flag is 0,Current Time =", current_time)
                ghh=train(df1)
                my_dict[i]=ghh
                aj.append(my_dict[i])

                flag=1
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("After train Current Time =", current_time)



            #else:
                #aj.append(train1(df1))
            #print("hiya removing i")
            #os.remove(audiop1+"//"+str(i))
            x1.append(audiop1+"//"+str(i))
        try:
            #print("removing csv files")
            os.remove(fx)
            k=[]

        except:
            print()
        for m in x1:
            try:
                #print("m is",m)

                os.remove(m)
            except:
                #print("could not remove")
                continue
        ##for j1 in aj:
            #print("j1 is",j1)
        #return aj

        #print(aj)
        for i in my_dict.keys():
            #print("hello1",max(my_dict[i][0][0]))
            #print("hello2",my_dict[i][1])
            x=my_dict[i][1]['Predicted Values'].to_string()
            if("not stressed" in x):
                print(i,":","not stressed")
            else:
                print(i,":","stressed")
        my_dict.clear()
        return aj
