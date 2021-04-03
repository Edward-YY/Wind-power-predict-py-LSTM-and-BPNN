from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from preprocess import data_to_supervised_3
import pandas as pd

#definite a LSTM predictor
class LSTMPredictor(object):
    """
    parameter of instructor
    n:predict n hours ago
    timeDf:the whole time list
    *timeList:a list containing start time of training,end time of traning,
    star time of testing , end time of testing
    """
    def __init__(self,n, timeDf, *timeList):
        self.shiffted_n = n
        self.shifted_value = data_to_supervised_3(value_df, self.shiffted_n)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.reframed = self.scaler.fit_transform(self.shifted_value)

        train_start = timeList[0]
        train_end = timeList[1]
        test_start = timeList[2]
        test_end = timeList[3]

        train_start_n = timeDf.index(train_start) - n
        train_end_n = timeDf.index(train_end) - n
        test_start_n = timeDf.index(test_start) - n
        test_end_n = timeDf.index(test_end) - n

        self.trainx = self.reframed[train_start_n: train_end_n, :-1]
        self.testx = self.reframed[test_start_n: test_end_n, :-1]
        # split into input and outputs
        self.trainy = self.reframed[train_start_n: train_end_n, -1]
        self.testy = self.reframed[test_start_n: test_end_n, -1]
        self.trainy = self.trainy[:, np.newaxis]
        self.trainx = self.trainx.reshape((self.trainx.shape[0], 1, self.trainx.shape[1]))
        self.testx = self.testx.reshape((self.testx.shape[0], 1, self.testx.shape[1]))
    #set a LSTM nueronetwork in LSTMPredictor
    def setmodel(self):
        self.model = Sequential()
        self.model.add(LSTM(60, input_shape=(self.trainx.shape[1], self.trainx.shape[2]),return_sequences=True))
        self.model.add(LSTM(60, activation='sigmoid',return_sequences=True))
        self.model.add(LSTM(60, activation='sigmoid'))
        self.model.add(Dense(1))
    #definite a method to train model
    def trainModel(self):
        self.model.compile(loss='mae', optimizer='adam')
        # fit network
        history = self.model.fit(self.trainx, self.trainy, epochs=200, batch_size=100,\
                                 validation_data=(self.trainx, self.trainy), verbose=2, shuffle=False)
        # plot history
        pyplot.plot(history.history['loss'], label='train loss')
        pyplot.plot(history.history['val_loss'], label='test loss')
        pyplot.legend()
        pyplot.show()
    #definite a method for prediction
    def predict(self):
        self.ypre = self.model.predict(self.testx)

    #definite a method to accurate shape
    def accur(self):
        self.testx = self.testx.reshape((self.testx.shape[0], self.testx.shape[2]))
        self.x_y_pre = np.concatenate((self.testx, self.ypre), axis=1)
        self.x_y_pre_inv = self.scaler.inverse_transform(self.x_y_pre)
        self.y_pre_inv = self.x_y_pre_inv[:, -1]

        self.testy = self.testy[:, np.newaxis]
        self.x_y_test = np.concatenate((self.testx, self.testy), axis=1)
        self.x_y_test_inv = self.scaler.inverse_transform(self.x_y_test)
        self.y_test_inv = self.x_y_test_inv[:, -1]

        self.rmse = np.math.sqrt(mean_squared_error(self.y_test_inv, self.y_pre_inv))
        self.nrmse = self.rmse / 24.75 * 100
        print('LSTM{},rmse:{},nrmse{}'.format(self.shiffted_n,self.rmse, self.nrmse))

    def getY_pre_inv(self):
        return self.y_pre_inv

    def getY_test_inv(self):
        return self.y_test_inv

    #definite a method to plot figure
    def plotFig(self):
        pyplot.plot(self.y_test_inv, label='actual')
        pyplot.plot(self.y_pre_inv, label='LSTM' + str(self.shiffted_n ))
        pyplot.legend(loc='upper right')
        pyplot.xlabel('Time Point(15min)')
        pyplot.ylabel('Wind Power(MW)')
        y_stick = np.arange(0, 25, 6)
        x_stick = np.arange(0, 97, 16)
        pyplot.xticks(x_stick[:])
        pyplot.yticks(y_stick[1:])
        pyplot.show()

#difinite a BPNNPrediction class
class BPNNPredictor(object):
    """
    parameter of instructor
    n:predict n hours ago
    timeDf:the whole time sequential data
    *timeList:a list containing start time of training,end time of traning,
    star time of testing , end time of testing
    """
    def __init__(self,n, timeDf, *timeList):
        self.shiffted_n = n
        self.shifted_value = data_to_supervised_3(value_df, self.shiffted_n)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.reframed = self.scaler.fit_transform(self.shifted_value)

        train_start = timeList[0]
        train_end = timeList[1]
        test_start = timeList[2]
        test_end = timeList[3]

        train_start_n = timeDf.index(train_start) - n
        train_end_n = timeDf.index(train_end) - n
        test_start_n = timeDf.index(test_start) - n
        test_end_n = timeDf.index(test_end) - n

        self.trainx = self.reframed[train_start_n: train_end_n, :-1]
        self.testx = self.reframed[test_start_n: test_end_n, :-1]
        # split into input and outputs
        self.trainy = self.reframed[train_start_n: train_end_n, -1]
        self.testy = self.reframed[test_start_n: test_end_n, -1]
        self.trainy = self.trainy[:, np.newaxis]
        self.trainx = self.trainx.reshape((self.trainx.shape[0], 1, self.trainx.shape[1]))
        self.testx = self.testx.reshape((self.testx.shape[0], 1, self.testx.shape[1]))

    def setmodel(self):
        self.model = Sequential()
        self.model.add(Dense(60, input_shape=(self.trainx.shape[1], self.trainx.shape[2]), activation='sigmoid'))
        #model.add(Dropout(0.2))
        self.model.add(Dense(60, activation='sigmoid'))
        #model.add(Dropout(0.2))
        self.model.add(Dense(60, activation='sigmoid'))
        #model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.train_y_3d = self.trainy.reshape(self.trainy.shape[0], 1,1)


    def trainModel(self):
        self.model.compile(loss='mae', optimizer='adam',metrics = ['accuracy'])
        # fit network
        self.history = self.model.fit(self.trainx, self.train_y_3d,\
                                      batch_size = 100,epochs = 200,validation_split = 0.2 )
        # plot history
        pyplot.plot(self.history.history['loss'], label='train loss')
        pyplot.plot(self.history.history['val_loss'], label='test loss')
        pyplot.legend()
        pyplot.show()

    def predict(self):
        self.ypre = self.model.predict(self.testx)

    def accur(self):
        self.testx = self.testx.reshape((self.testx.shape[0], self.testx.shape[2]))
        self.ypre = self.ypre.reshape(self.ypre.shape[0],self.ypre.shape[1])
        self.x_y_pre = np.concatenate((self.testx, self.ypre), axis=1)
        self.x_y_pre_inv = self.scaler.inverse_transform(self.x_y_pre)
        self.y_pre_inv = self.x_y_pre_inv[:, -1]

        self.testy = self.testy[:, np.newaxis]
        self.x_y_test = np.concatenate((self.testx, self.testy), axis=1)
        self.x_y_test_inv = self.scaler.inverse_transform(self.x_y_test)
        self.y_test_inv = self.x_y_test_inv[:, -1]

        self.rmse = np.math.sqrt(mean_squared_error(self.y_test_inv, self.y_pre_inv))
        self.nrmse = self.rmse / 24.75 * 100
        print('LSTM{},rmse:{},nrmse{}'.format(self.shiffted_n,self.rmse, self.nrmse))

    def getY_pre_inv(self):
        return self.y_pre_inv

    def plotFig(self):
        pyplot.plot(self.y_test_inv, label='actual')
        pyplot.plot(self.y_pre_inv, label='BPNN' + str(self.shiffted_n ))
        pyplot.legend(loc='upper right')
        pyplot.xlabel('Time Point(15min)')
        pyplot.ylabel('Wind Power(MW)')
        y_stick = np.arange(0, 25, 6)
        x_stick = np.arange(0, 97, 16)
        pyplot.xticks(x_stick[:])
        pyplot.yticks(y_stick[1:])
        pyplot.show()


# load dataset
path = 'dataset/wind power data 15min.csv'
df = pd.read_csv(path, header=0, index_col=None)

time_df = df.pop('時間').tolist()
df = df.drop(['風向','温度'],axis=1)
value_df = df.values.astype('float32')
time_list = ['2/19/2006 15:00:00',\
             '3/20/2006 15:00:00',\
             '3/20/2006 15:00:00',\
             '3/21/2006 15:00:00']

#create a lstm predict instance
predictor0 = LSTMPredictor(4, time_df, *time_list)
predictor0.setmodel()
predictor0.trainModel()
predictor0.predict()
predictor0.accur()
ylstm0 = predictor0.getY_pre_inv()
powerActual = predictor0.getY_test_inv()

predictorBPNN = BPNNPredictor(4, time_df, *time_list)
predictorBPNN.setmodel()
predictorBPNN.trainModel()
predictorBPNN.predict()
predictorBPNN.accur()
yBPNN = predictorBPNN.getY_pre_inv()
predictorBPNN.plotFig()

pyplot.plot(powerActual,label = 'actual',ls = '-',c = 'b')
pyplot.plot(ylstm0 ,label = 'LSTM',ls = '--',c = 'r')
pyplot.plot(yBPNN,label = 'BPNN',ls = '-.',c = 'g')

pyplot.legend(loc = 'upper right')
pyplot.xlabel('Time Point(15min)')
pyplot.ylabel('Wind Power(MW)')
y_stick = np.arange(0,25,6)
x_stick = np.arange(0,97,16)
pyplot.xticks(x_stick[:])
pyplot.yticks(y_stick[:])
pyplot.show()