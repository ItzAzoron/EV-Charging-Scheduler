from gurobipy import *
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import datetime as dt
import tensorflow as tf
from dataclasses import dataclass


from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import he_normal,Constant
from tensorflow.keras import utils,backend as K
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import PySimpleGUI as sg

# Let's save the parameters of our time series in the dataclass


def start_and_finish():
    a = 2016
    b = 2020

    start = dt.datetime(a, 6, 1)
    finish = dt.datetime(b, 2, 1)
    start_accurate = datetime.datetime(2016, 6, 1, 0, 0)
    finish_accurate = datetime.datetime(2020, 2, 1, 23, 45)
    end = "2020-02-02"
    # end="2017-02-02"
    return start, finish, end, start_accurate, finish_accurate




def initiate_database():
    start_date = dt.datetime(2016, 6, 1)  # de eerste datum
    end_date = dt.datetime(2020, 2, 1)  # de einddatum
    # df_solar = pd.read_csv('Data6.csv', header=0)  # data uit bestand halen
    df_solar = pd.read_csv('Data2016_2020.csv', header=0)
    df_solar['Time'] = df_solar['Date'].astype(str) + " " + df_solar["Time"]  # Datum en tijd in zelfde kolom
    del df_solar['Date']  # overbodige kolom verwijderen
    df_solar['Time'] = pd.to_datetime(df_solar['Time'])  # naar tijdsvariabele ipv string
    df_solar.set_index('Time', inplace=True)  # zorgen dat datum de index is

    # df_belpex = pd.read_csv('Belpex.csv', header=0)  # belpex lezen
    df_belpex = pd.read_csv('belpex2016_2020.csv', header=0)
    df_belpex = df_belpex.rename(columns={'Prijs': 'Price'})
    df_belpex['Time'] = pd.to_datetime(df_belpex['Time'], dayfirst=True)
    df_belpex.set_index('Time', inplace=True)

    # df_belpex = df_belpex.resample('15T').ffil()

    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    for d in dates:
        try:
            p = df_belpex.loc[d]
        except KeyError:
            df_belpex.loc[d] = df_belpex.loc[d - dt.timedelta(hours=1)]

    df_belpex = df_belpex.sort_index()

    df_belpex = df_belpex.fillna(method='pad')

    df_belpex[
        df_belpex.index.duplicated(keep=False)]  # Zoekt naar rijen met eenzelfde index maar een verschillende waarde.
    df_belpex = df_belpex[~df_belpex.index.duplicated(keep='first')]  # Behoud de eerste rij met die index.

    line = pd.to_datetime("2020-02-02 00:00:00", format="%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[0.00]], columns=['Price'], index=[line])
    df_belpex = pd.concat([df_belpex, pd.DataFrame(new_row)], ignore_index=False)

    df_belpex = df_belpex.resample('15T').ffill()
    df_belpex = df_belpex.drop(line)

    dates = pd.date_range(start=start_date, end=end_date, freq='15T')
    for d in dates:
        try:
            p = df_solar.loc[d]
        except KeyError:
            df_solar.loc[d] = df_solar.loc[d - dt.timedelta(minutes=15)]

    df_solar = df_solar.sort_index()

    df_solar = df_solar.fillna(method='pad')

    df_solar[
        df_solar.index.duplicated(keep=False)]  # Zoekt naar rijen met eenzelfde index maar een verschillende waarde.
    df_solar = df_solar[~df_solar.index.duplicated(keep='first')]  # Behoud de eerste rij met die index.

    d = {'belpex': df_belpex.values.flatten(), 'solar': df_solar.values.flatten()}
    data = pd.DataFrame(index=df_solar.index, data=d)

    data['belpex'] = data['belpex'].div(1000)
    return data


data = initiate_database()


def plot_solar_history(data):
    start = datetime.datetime(2018, 8, 1, 0, 0)
    end = datetime.datetime(2019, 8, 31, 23, 45)
    plt.plot(data.solar[start:end])
    plt.xlabel("Time [-]")
    plt.ylabel("Solar Output [kWh/quarter hour]")
    plt.show()


def plot_belpex_history(data):
    plt.figure()
    start = datetime.datetime(2018, 8, 1, 0, 0)
    end = datetime.datetime(2019, 8, 31, 23, 0)
    plt.plot(data.belpex[start:end])
    plt.xlabel("Time [-]")
    plt.ylabel("Net Price[€/MWh]")
    plt.show()


# extreme cutting
def mean_data(database, n_std=5):
    mean = database.mean()
    std = database.std()
    n_std = 5
    database[(database >= mean + n_std * std)] = mean + n_std * std
    database[(database <= mean - n_std * std)] = mean + n_std * std
    return database


# kies hier model input grootte (jaar verdeelt in maanden/weken/dagen)
def dataset_creator(type_model=0, data=initiate_database(), size=1, test_setsize=10, nodesperlayer=[96 * 7, 96],
                    testdays=[dt.datetime(2018, 9, 20)]):
    # creating X, the training set input
    interval_dim = [96 * size, 96]
    start = datetime.datetime(2016, 6, 1, 0, 0)
    end = datetime.datetime(2020, 2, 1, 23, 45)
    X = []
    testset = []
    if type_model == "solar" or type_model == 0:
        X_unprocessed = data.solar[start:end].resample('15T').mean().values.reshape(-1, 96)

        begintestset = [testday - dt.timedelta(days=size) for testday in testdays]
    elif type_model == "belpex" or type_model == 1:
        X_unprocessed = data.belpex[start:end].resample('15T').mean().values.reshape(-1, 96)
        begintestset = [testday - dt.timedelta(days=size) for testday in testdays]

    for j in range(0, len(X_unprocessed) - size):
        temp = []
        if j not in [(X - start).days for X in begintestset]:
            for i in range(j, j + size):
                temp += list(X_unprocessed[i])
            X.append(temp)
        else:
            for i in range(j, j + size):
                temp += list(X_unprocessed[i])
            testset.append(temp)
        '''
        if j < len(X_unprocessed) - size - test_setsize:
            for i in range(j, j + size):
                temp += list(X_unprocessed[i])
            X.append(temp)
        else:
            for i in range(j, j + size):
                temp += list(X_unprocessed[i])
            testset.append(temp)
        '''
    testset = np.array(testset)
    X = np.array(X)

    # creating y, the training set expected output, which the model trains on
    start = datetime.datetime(2016, 6, 1 + size, 0, 0)
    end = datetime.datetime(2020, 2, 1, 23, 45)
    if type_model == "solar" or type_model == 0:
        y = data.solar[start:end].resample('15T').mean().values.reshape(-1, interval_dim[1])
        for x in begintestset:
            np.delete(y, (x - start).days)
    elif type_model == "belpex" or type_model == 1:
        y = data.belpex[start:end].resample('15T').mean().values.reshape(-1, interval_dim[1])
        for x in begintestset:
            np.delete(y, (x - start).days)
    n_features = nodesperlayer
    # n_features = [interval_dim[0], (size * interval_dim[1]) , (size * interval_dim[1]), interval_dim[1]] #aantal nodes per layer
    return X, y, testset, n_features


def sequential_modelcreator(amount, n_features, activations):
    model = Sequential(name="Optimizer")
    model.add(Input(name="input", shape=n_features[0], ))
    for i in range(1, amount):
        model.add(Dense(name="layer" + str(i), units=n_features[i], activation=activations[i - 1]))
        model.add(Dropout(name="drop" + str(i), rate=0.2))
    model.add(Dense(name="output", units=n_features[amount], activation=activations[amount]))

    # define metrics
    def R2(y, y_hat):
        ss_res = K.sum(K.square(y - y_hat))
        ss_tot = K.sum(K.square(y - K.mean(y)))
        return (1 - ss_res / (ss_tot + K.epsilon()))

    # compile the neural network
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[R2])
    return model



# train/validation
def train_model(model, X, y, type_model="solar", epo=100, batch=16, reset=0, display=0):
    if type_model == "solar" or type_model == 0:
        cp = ModelCheckpoint('SeqSolPrevious/', save_best_only=True)

        # model.load_weights('modelweigths_sol.h5')
    elif type_model == "belpex" or type_model == 1:
        cp = ModelCheckpoint('SeqBelPrevious/', save_best_only=True)
        # model.load_weights('modelweigths_bel.h5')
    training = model.fit(x=X, y=y, batch_size=batch, epochs=epo, shuffle=True, verbose=0, validation_split=0.3,callbacks=[cp])
    mse = training.history['val_loss'][-1]
    score = training.history['val_R2'][-1]
    if type_model == "solar" or type_model == 0:
        model = load_model('SeqSolPrevious/', compile=False)
    elif type_model == "belpex" or type_model == 1:
        model = load_model('SeqBelPrevious/', compile=False)
    if display == 1:


        # plot
        metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))
        ## training
        ax[0].set(title="Training")
        ax11 = ax[0].twinx()
        ax[0].plot(training.history['loss'], color='black')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss', color='black')

        for metric in metrics:
            ax11.plot(training.history[metric], label=metric)
            ax11.set_ylabel("Score", color='steelblue')
        ax11.legend()

        ## validation
        ax[1].set(title="Validation")
        ax22 = ax[1].twinx()
        ax[1].plot(training.history['val_loss'], color='black')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax22.plot(training.history['val_' + metric], label=metric)
            ax22.set_ylabel("Score", color="steelblue")
        plt.show()
    return model, mse, score




def predictor(model, inp):
    predicted = model.predict(inp)
    return predicted

def initiate_lstmdatabase(n_std=2):
    start_date, end_date, end, start_accurate, finish_accurate = start_and_finish()
    df_solar = pd.read_csv('Data2016_2020.csv', header=0)  # data uit bestand halen
    df_solar['Time'] = df_solar['Date'].astype(str) + " " + df_solar["Time"]  # Datum en tijd in zelfde kolom
    del df_solar['Date']  # overbodige kolom verwijderen
    df_solar['Time'] = pd.to_datetime(df_solar['Time'])  # naar tijdsvariabele ipv string
    df_solar.set_index('Time', inplace=True)  # zorgen dat datum de index is

    df_belpex = pd.read_csv('belpex2016_2020.csv', header=0)  # belpex lezen
    df_belpex = df_belpex.rename(columns={'Prijs': 'Price'})
    df_belpex['Time'] = pd.to_datetime(df_belpex['Time'], dayfirst=True)
    df_belpex.set_index('Time', inplace=True)

    # df_belpex = df_belpex.resample('15T').ffil()

    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    for d in dates:
        try:
            p = df_belpex.loc[d]
        except KeyError:
            df_belpex.loc[d] = df_belpex.loc[d - dt.timedelta(hours=1)]

    df_belpex = df_belpex.sort_index()

    df_belpex = df_belpex.fillna(method='pad')

    df_belpex[df_belpex.index.duplicated(keep=False)]  # Zoekt naar rijen met eenzelfde index maar een verschillende waarde.
    df_belpex = df_belpex[~df_belpex.index.duplicated(keep='first')]  # Behoud de eerste rij met die index.

    line = pd.to_datetime(end + " 00:00:00", format="%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[0.00]], columns=['Price'], index=[line])
    df_belpex = pd.concat([df_belpex, pd.DataFrame(new_row)], ignore_index=False)

    df_belpex = df_belpex.resample('15T').ffill()
    df_belpex = df_belpex.drop(line)

    dates = pd.date_range(start=start_date, end=end_date, freq='15T')
    for d in dates:
        try:
            p = df_solar.loc[d]
        except KeyError:
            df_solar.loc[d] = df_solar.loc[d - dt.timedelta(minutes=15)]

    df_solar = df_solar.sort_index()

    df_solar = df_solar.fillna(method='pad')

    df_solar[df_solar.index.duplicated(keep=False)]  # Zoekt naar rijen met eenzelfde index maar een verschillende waarde.
    df_solar = df_solar[~df_solar.index.duplicated(keep='first')]  # Behoud de eerste rij met die index.

    d = {'belpex': df_belpex.values.flatten(), 'solar': df_solar.values.flatten()}
    data = pd.DataFrame(index=df_solar.index, data=d)
    data['belpex'] = data['belpex'].div(1000)

    mean = data['solar'].mean()
    std = data['solar'].std()
    data['solar'][(data['solar'] >= mean + 4 * std)] = mean + 4 * std
    data['solar'][(data['solar'] <= mean - 4 * std)] = mean + 4 * std

    mean = data['belpex'].mean()
    std = data['belpex'].std()
    data['belpex'][(data['belpex'] >= mean + n_std * std)] = mean + n_std * std
    data['belpex'][(data['belpex'] <= mean - n_std * std)] = mean + n_std * std

    solar_series = data['solar']
    belpex_series = data['belpex']

    return solar_series,belpex_series,data

@dataclass
class G:
    SOLAR_SERIES,BELPEX_SERIES,DATA = initiate_lstmdatabase()
    WINDOW_SIZE = 96 # how many data points will we take into account to make our prediction

def plot_series(series, format="-", start=0, end=None):
    """Helper function to plot our time series"""
    plt.plot(series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


def df_to_X_y(df, window_size=G.WINDOW_SIZE):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(0,len(df_as_np)-window_size,96):
        row = [r for r in df_as_np[i+14:i+80]]
        #80=>window_size
        X.append(row)
        label=[]
        #80=>96
        for part in df_as_np[i+window_size+14:i+window_size+80]:
            label.append(part[0])
        y.append(label)
    return np.array(X), np.array(y)

def preprocess(X,solar_training_mean, solar_training_std):
    #scaler  = MinMaxScaler(feature_range = (0,1))
    #X[:,:,0] = scaler.fit_transform(X[:,:,0])
    X[:, :, 0] = (X[:, :, 0] - solar_training_mean) / solar_training_std
    return X


def lstm_modelcreator(amount, n_features, mode =0):
    initializer = tf.keras.initializers.HeNormal()
    model = Sequential()
    model.add(InputLayer((66, 5)))
    # G.WINDOW_SIZE
    model.add(LSTM(128))
    if mode == 0:
        for i in range(1, amount):
            model.add(Dense(n_features[i], bias_initializer=Constant(0.1), kernel_initializer=initializer))
            model.add(LeakyReLU(alpha=0.01))
            model.add(Dropout(0.2))

        model.add(Dense(66, 'linear'))
    elif mode == 1:
        for i in range(1, amount):
            model.add(Dense(n_features[i], bias_initializer=Constant(0), kernel_initializer=initializer,activation = "elu"))
        model.add(Dense(66, 'linear'))
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    # 96
    model.summary()
    return model


def looper(min_nodes, max_nodes, layers, step=1, inputlayer=66):
    feetures = []
    start = str(min_nodes) * layers
    end = str(max_nodes) * layers
    j = [min_nodes for x in range(layers)]

    while True:

        if j[0] >= max_nodes:
            return feetures
        feetures.append([inputlayer] + j + [66])
        for tel in range(len(j)):
            j[tel] += step


def train_lstmmodel(model, X_train, y_train, epoch=50):
    cp = ModelCheckpoint('ModelPrevious/', save_best_only=True)
    training = model.fit(X_train, y_train, epochs=epoch, callbacks=[cp], verbose=0,shuffle=True,validation_split=0.3)
    mse = training.history['val_loss'][-1]
    print(min(training.history['val_loss']))
    #training = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epoch, verbose=0)
    model = load_model('ModelPrevious/')
    # modelbel.save_weights('Belpexmodel1.h5')

    plt.figure()
    plt.plot(training.history['val_loss'], color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    return model, mse



def automatic_lstmmodel(input_size, min_layers, max_layers, min_nodes, max_nodes, step,repeat=0, mode = 0):
    woordenboek = {}
    min_layers += 1
    max_layers += 2
    if repeat == 0:
        for i in range(min_layers, max_layers):
            features = looper(min_nodes, max_nodes, i - 1, step, input_size)
            for j in range(len(features)):
                testmod = lstm_modelcreator(i, features[j])
                woordenboek["model" + str(i - 1) + "-" + str(j)] = testmod
    else:
        features = looper(min_nodes, max_nodes, min_layers, step, input_size)
        for i in range(repeat):

            testmod = lstm_modelcreator(min_layers, features[0],mode)
            woordenboek["model" + str(i - 1)] = testmod

    return woordenboek


def auto_trainer(woordenboek, X_train, y_train, X_val, y_val, epochs=50):
    antwoord = [[], [], []]
    text = []
    with open('checkpoint.txt', 'w') as txtfile:
        for model in woordenboek:
            woordenboek[model], mse = train_lstmmodel(woordenboek[model], X_train, y_train, X_val, y_val, epochs)
            print(model + ": mse=" + str(mse))
            antwoord[0].append(model)
            antwoord[1].append(mse)

            txtfile.write('\n' + model + ": mse=" + str(mse))

            # text.append(str(model)+": mse="+str(mse)+" score="+str(score))
    return woordenboek, antwoord


def Gurobi_optimizer(to_optimize_sol, to_optimize_bel, cars_plan, upper_bound_loading=12.5, lower_bound_loading=1.25,
                     upper_bound_discharge=100000):
    # creates optimized schedule for multiple cars
    # to_optimize: predictions of solar energy and Belpex price
    # cars: list of cars, each car is list like [battery, start, end]
    # upper_bound_loading: max kWh charge in 15 minutes for each loading station
    # upper_bound_discharge: max kWh able to give to net
    # lower_bound_loading: min nonzero kWh charge in 15 minutes for each loading station
    cars = []
    for car in cars_plan:
        ratio = (car[2] - car[1] + 1) * upper_bound_loading / car[0]  # max/needed
        if ratio < 1:
            cars.append([ratio * car[0], car[1], car[2], ratio])
        else:
            cars.append([car[0], car[1], car[2], ratio])

    # predicted data
    sun = to_optimize_sol[33:73]  # solar energy
    price_buy = to_optimize_bel[33:73]  # price of electricity
    price_sell = [x / 2 for x in price_buy]

    end = len(sun)  # index of end of the working day
    # intitialize model
    m = Model("schedule")

    # variables
    E_charge = []
    zero = []
    total = 0
    for c in range(len(cars)):
        total += cars[c][0]  # total kWh to charge all cars
        E_charge.append(m.addVars(len(sun), lb=0, ub=upper_bound_loading, vtype=GRB.CONTINUOUS,
                                  name="charge" + str(c)))  # list with schedules for each car
        zero.append(m.addVars(len(sun), vtype=GRB.BINARY,
                              name="zero" + str(c)))  # binary decision variable to determine zero or greater than lb
        for t in range(len(sun)):
            if (t < cars[c][1]) or (t > cars[c][2]):  # if car is not present
                m.addConstr(E_charge[c][t] == 0)
        m.addConstrs((zero[c][t] == 1) >> (E_charge[c][t] == 0) for t in range(len(sun)))
        m.addConstrs((zero[c][t] == 0) >> (E_charge[c][t] >= lower_bound_loading) for t in range(len(sun)))
    for c in range(len(cars)):
        m.addConstr(sum(E_charge[c][t] for t in range(len(sun))) == cars[c][
            0])  # car has to be fully charged at the end of the day
    m.Params.LogToConsole = 0

    electricity_buy = m.addVars(len(sun), lb=0, ub=len(cars) * upper_bound_loading, vtype=GRB.CONTINUOUS,
                                name="electricity_buy")
    electricity_sell = m.addVars(len(sun), lb=0, ub=upper_bound_discharge, vtype=GRB.CONTINUOUS,
                                 name="electricity_sell")
    charge = m.addVars(len(sun), vtype=GRB.BINARY)
    discharge = m.addVars(len(sun), vtype=GRB.BINARY)
    result = m.addVar(lb=-GRB.INFINITY, name='result')  # variable to store minimized price
    # constraints
    for t in range(len(sun)):  # charge 'decides' wether we buy E from the net or we sell E
        m.addConstr((charge[t] == 1) >> (sum(E_charge[i][t] for i in range(len(cars))) >= sun[t]))
        m.addConstr((charge[t] == 0) >> (sum(E_charge[i][t] for i in range(len(cars))) <= sun[t]))
        m.addConstr((charge[t] == 1) >> (electricity_buy[t] == sum(E_charge[i][t] for i in range(len(cars))) - sun[t]))
        m.addConstr((charge[t] == 0) >> (electricity_sell[t] == sun[t] - sum(E_charge[i][t] for i in range(len(cars)))))
        m.addConstr((charge[t] == 1) >> (electricity_sell[t] == 0))
        m.addConstr((charge[t] == 0) >> (electricity_buy[t] == 0))

    m.addConstrs(electricity_buy[t] <= (len(cars) * upper_bound_loading - sun[t]) * charge[t] for t in range(len(sun)))
    m.addConstrs(electricity_sell[t] <= upper_bound_discharge * discharge[t] for t in range(len(sun)))

    m.addConstrs(0 <= electricity_buy[t] for t in range(len(sun)))
    m.addConstrs(0 <= electricity_sell[t] for t in range(len(sun)))
    m.addConstrs(sun[t] - electricity_sell[t] >= 0 for t in range(len(sun)))
    m.addConstrs(charge[t] + discharge[t] <= 1 for t in range(len(sun)))
    m.addConstr(sum(sun[t] + electricity_buy[t] - electricity_sell[t] for t in range(len(sun))) == total)

    m.addConstr(
        result == sum(electricity_buy[t] * price_buy[t] - electricity_sell[t] * price_sell[t] for t in range(len(sun))))
    # objective function
    m.setObjective(
        sum(electricity_buy[t] * price_buy[t] - electricity_sell[t] * price_sell[t] for t in range(len(sun))),
        GRB.MINIMIZE)

    m.optimize()

    # t = -15 is start time, value at t=0 was produced over 15 minutes before 0, so if we need to make schedule from 8:00 till...,
    # then starttime of the used data has to be t = 8:15
    solution = []
    for c in range(len(cars)):
        x = []
        y = []
        for t in range(len(E_charge[c])):
            y.append(E_charge[c][t].X * 4)
            y.append(E_charge[c][t].X * 4)
            x.append(8 + (t) * 0.25)
            x.append(8 + (t + 1) * 0.25)
        l = len(solution)
        if l == 0:
            solution.append(y)
        else:
            solution.append([y[i] + solution[l - 1][i] for i in range(len(y))])
    return solution  # oplossing is lijst met schema's, waarvan laatste schema het totaal is van alle auto's


def plot_optimized_schedule(schedule, cars, upper_bound_loading, sun_predicted, filename,title, legend=False):
    # makes a plot of the optimized schedule
    # schedule: optimized solution found with Gurobi (special list with summed schedules of cars cf. Gurobi function)
    # cars: list with cars, each car is a list: [battery,start,end]
    # upper_bound_loading: max kWh charge in 15 minutes for each loading station
    # sun_predicted: predicted solar data, if not given will not be plotted

    # find ratio of total possible charge over wanted charge for each car
    cars_effective = []
    for car in cars:
        ratio = (car[2] - car[1] + 1) * upper_bound_loading / car[0]  # max/needed
        if ratio < 1:
            cars_effective.append([ratio * car[0], car[1], car[2], ratio])
        else:
            cars_effective.append([car[0], car[1], car[2], ratio])
    # kWh/15min -> kW, so multiply by 4
    sun_power = []
    x = []
    for t in range(len(schedule[-1]) // 2):
        if len(sun_predicted) > 0:
            sun_power.append(sun_predicted[t] * 4)
            sun_power.append(sun_predicted[t] * 4)
        x.append(8 + (t) * 0.25)
        x.append(8 + (t + 1) * 0.25)
    L = len(schedule)
    plt.figure(22 + len(filename))
    for l in range(L):
        plt.plot(x, schedule[L - l - 1], color=((256 / len(cars)) * l / 256, (256 / len(cars)) * l / 256, 256 / 256),
                 label='charging power car ' + str(L - l) + ": " + (
                     str(round(cars_effective[L - l - 1][3] * 100, 2)) + "%" if cars_effective[L - l - 1][
                                                                                    3] < 1 else "100%"), zorder=-L)
        plt.xticks([8 + i for i in range(11)])
        plt.fill_between(x, 0, schedule[L - l - 1],
                         color=((256 / len(cars)) * l / 256, (256 / len(cars)) * l / 256, 256 / 256))  # colors[l]
    plt.plot(x, [sun_power[t] for t in range(len(sun_power))], color='red', label='solar power', zorder=9)
    plt.plot(8,0)
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.ylabel('kW')
    if legend:
        plt.legend(loc="upper left")
    plt.title(title, fontweight="bold")
    plt.savefig(filename)
    legend = plt.legend()
    fig = legend.figure
    fig.canvas.draw()
    fig.set_size_inches(18.5, 10.5)
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend.png', dpi="figure", bbox_inches=bbox)
    plt.close()


def plot_predicted_price(price_buy_pred, price_buy_real):
    # function that plots predicted Belpex data
    # price_buy: prediction of Belpex data
    plt.figure()
    price_power_pred = []
    price_power_real = []
    x = []
    for t in range(len(price_buy_pred)):
        price_power_pred.append(price_buy_pred[t])
        price_power_pred.append(price_buy_pred[t])
        price_power_real.append(price_buy_real[t])
        price_power_real.append(price_buy_real[t])
        x.append(8 + (t) * 0.25)
        x.append(8 + (t + 1) * 0.25)
    plt.xticks([8 + i for i in range(11)])
    plt.grid()
    plt.plot(x, price_power_pred, color='green', label='predicted price')
    plt.plot(x, price_power_real, color='blue', label='real price')
    plt.plot(8,0)
    plt.xlabel('Time [hours]')
    plt.ylabel('price (euros per kWh)')
    plt.title('Predicted and real belpex data', fontweight="bold")
    plt.legend(loc="lower left")
    plt.savefig('belpexcomparison.png')
    plt.show()


def plot_predicted_sun(sun_pred, sun_real):
    plt.figure()
    solar_power_pred = []
    solar_power_real = []
    x = []
    for t in range(len(sun_pred)):
        solar_power_pred.append(sun_pred[t] * 4)
        solar_power_pred.append(sun_pred[t] * 4)
        solar_power_real.append(sun_real[t] * 4)
        solar_power_real.append(sun_real[t] * 4)
        x.append(8 + (t) * 0.25)
        x.append(8 + (t + 1) * 0.25)
    plt.xticks([8 + i for i in range(11)])
    plt.grid()
    plt.plot(8,0)
    plt.plot(x, solar_power_pred, color='green', label='predicted solar energy')
    plt.plot(x, solar_power_real, color='blue', label='real solar energy')
    plt.xlabel('Time [hours]')
    plt.ylabel('kW')
    plt.title('Predicted and real solar data', fontweight="bold")
    plt.legend(loc="upper right")
    plt.savefig('suncomparison.png')
    plt.show()

def plot_relative_real(oplossing, sun, price_buy, cars, upper_bound_loading):
    # function that creates plot to view price, solar and schedule
    cars_effectief = []
    for car in cars:
        ratio = (car[2] - car[1] + 1) * upper_bound_loading / car[0]  # max/needed
        if ratio < 1:
            cars_effectief.append([ratio * car[0], car[1], car[2], ratio])
        else:
            cars_effectief.append([car[0], car[1], car[2], ratio])

    # realistsische plot maken van tijd -> vermogen in kW
    sun_power = []
    price_power = []
    x = []
    for t in range(int(len(oplossing[0]) / 2)):
        sun_power.append(sun[t] * 4)
        sun_power.append(sun[t] * 4)
        price_power.append(price_buy[t])
        price_power.append(price_buy[t])
        x.append(8 + (t) * 0.25)
        x.append(8 + (t + 1) * 0.25)

    L = len(oplossing)
    colors = ['blue', 'red', 'green', 'purple']
    plt.figure(L)
    for l in range(L):
        plt.plot(x, [i / max(sun_power) for i in oplossing[L - l - 1]],
                 color=((256 / len(cars)) * l / 256, (256 / len(cars)) * l / 256, 256 / 256),
                 label='car ' + str(L - l) + ": " + (
                     str(cars_effectief[L - l - 1][3] * 100) + "%" if cars_effectief[L - l - 1][3] < 1 else "100%"),
                 zorder=-L)
        plt.xticks([8 + i for i in range(11)])
        plt.fill_between(x, 0, [i / max(sun_power) for i in oplossing[L - l - 1]],
                         color=((256 / len(cars)) * l / 256, (256 / len(cars)) * l / 256, 256 / 256))  # colors[l]
    plt.plot(x, [sun_power[t] / max(sun_power) for t in range(len(sun_power))], color='red', label='solar power',
             zorder=9)
    plt.plot(x, [(i - min(price_power)) / (max(price_power) - min(price_power)) for i in price_power], color='green',
             label='belpex')
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.title('Normalised plot of schedule and real data', fontweight="bold")
    plt.savefig('normalisedschedulereal.png')
    legend = plt.legend()
    fig = legend.figure
    fig.canvas.draw()
    fig.set_size_inches(18.5, 10.5)
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend.png', dpi="figure", bbox_inches=bbox)
    plt.close()

def plot_relative_predicted(oplossing, sun, price_buy, cars, upper_bound_loading):
    # function that creates plot to view price, solar and schedule
    cars_effectief = []
    for car in cars:
        ratio = (car[2] - car[1] + 1) * upper_bound_loading / car[0]  # max/needed
        if ratio < 1:
            cars_effectief.append([ratio * car[0], car[1], car[2], ratio])
        else:
            cars_effectief.append([car[0], car[1], car[2], ratio])

    # realistsische plot maken van tijd -> vermogen in kW
    sun_power = []
    price_power = []
    x = []
    for t in range(int(len(oplossing[0]) / 2)):
        sun_power.append(sun[t] * 4)
        sun_power.append(sun[t] * 4)
        price_power.append(price_buy[t])
        price_power.append(price_buy[t])
        x.append(8 + (t) * 0.25)
        x.append(8 + (t + 1) * 0.25)

    L = len(oplossing)
    colors = ['blue', 'red', 'green', 'purple']
    plt.figure(L)
    for l in range(L):
        plt.plot(x, [i / max(sun_power) for i in oplossing[L - l - 1]],
                 color=((256 / len(cars)) * l / 256, (256 / len(cars)) * l / 256, 256 / 256),
                 label='car ' + str(L - l) + ": " + (
                     str(cars_effectief[L - l - 1][3] * 100) + "%" if cars_effectief[L - l - 1][3] < 1 else "100%"),
                 zorder=-L)
        plt.xticks([8 + i for i in range(11)])
        plt.fill_between(x, 0, [i / max(sun_power) for i in oplossing[L - l - 1]],
                         color=((256 / len(cars)) * l / 256, (256 / len(cars)) * l / 256, 256 / 256))  # colors[l]
    plt.plot(x, [sun_power[t] / max(sun_power) for t in range(len(sun_power))], color='red', label='solar power',
             zorder=9)
    plt.plot(x, [(i - min(price_power)) / (max(price_power) - min(price_power)) for i in price_power], color='green',
             label='belpex')
    plt.grid()
    plt.xlabel('Time [hours]')
    plt.title('Normalised plot of schedule and predicted data', fontweight="bold")
    plt.savefig('normalisedschedulepred.png')
    legend = plt.legend()
    fig = legend.figure
    fig.canvas.draw()
    fig.set_size_inches(18.5, 10.5)
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend.png', dpi="figure", bbox_inches=bbox)
    plt.close()


def create_naive_schedule(cars, upper_bound_loading=12.5, duration_day=40, make_plot=False):
    # function that creates a 'naive' schedule, meaning each cars charges max power from start time untill it's full
    # cars: list with cars, each car is a list: [battery,start,end]
    # upper_bound_loading: max kWh charge in 15 minutes for each loading station
    # duration_day: amount of quarter hours during day
    # make_plot: Boolean to choose if you want a plot of the schedule

    cars_effectief = []
    for car in cars:
        ratio = (car[2] - car[1] + 1) * upper_bound_loading / car[0]  # max/needed
        if ratio < 1:
            cars_effectief.append([ratio * car[0], car[1], car[2], ratio])
        else:
            cars_effectief.append([car[0], car[1], car[2], ratio])

    naive_loading = []
    for car in cars_effectief:
        naive_car = []
        duration_whole = int(car[0] // upper_bound_loading)
        duration_comma = (car[0] % upper_bound_loading) / upper_bound_loading
        start = car[1]
        for t in range(duration_day):
            if t in [i for i in range(start, start + duration_whole + 1)]:
                if t == start + duration_whole:
                    naive_car.append(duration_comma * upper_bound_loading)
                else:
                    naive_car.append(upper_bound_loading)
            else:
                naive_car.append(0)
        naive_loading.append(naive_car)
    total_schedule = []
    for t in range(duration_day):
        s = 0
        for schedule in naive_loading:
            s += schedule[t]
        total_schedule.append(s)
    total = []
    x = []
    # kWh per 15 minutes -> kW
    for t in range(duration_day):
        total.append(total_schedule[t] * 4)
        total.append(total_schedule[t] * 4)
        x.append(8 + (t) * 0.25)
        x.append(8 + (t + 1) * 0.25)
    if make_plot:
        plt.figure(200)
        plt.plot(x, total, label='charging rate')
        # plt.plot(x,sun_power,label='solar energy')
        plt.xticks([8 + i for i in range(11)])
        plt.grid()
        plt.fill_between(x, 0, total, color='blue')
        plt.xlabel('Time [hours]')
        plt.ylabel('kW')
        plt.legend(loc="upper left")
        plt.title("Naive schedule")
        plt.show()
    return total


def compare_prices(schedule_predicted_old, schedule_naive_old, price_buy_real, price_sell_real, sun_real):
    # function that compares prices paid with optimized schedule versus naive schedule
    # schedule_predicted: solution from Gurobi function
    # schedule naive: solution from create_naive_schedule
    # price_buy_real: real Belpex data from that day
    # sun_real: real solar data from that day
    schedule_predicted = []
    schedule_naive = []
    for i in range(0, len(schedule_predicted_old[-1]), 2):
        schedule_predicted.append(schedule_predicted_old[-1][i] / 4)
        schedule_naive.append(schedule_naive_old[i] / 4)
    price_naive = 0
    for t in range(len(schedule_naive)):
        if schedule_naive[t] > sun_real[t]:
            price_naive += (schedule_naive[t] - sun_real[t]) * price_buy_real[t]
        else:
            price_naive += (schedule_naive[t] - sun_real[t]) * price_sell_real[t]

    price_smart = 0
    for t in range(len(schedule_predicted)):
        if schedule_predicted[t] > sun_real[t]:
            price_smart += (schedule_predicted[t] - sun_real[t]) * price_buy_real[t]
        else:
            price_smart += (schedule_predicted[t] - sun_real[t]) * price_sell_real[t]
    delta = (price_naive-price_smart)
    return ("optimized: " + str(round(price_smart, 2)), "naive: " + str(round(price_naive, 2)),"winst is: "+str(round(delta,2))), delta


def generate_schedule(predictions_sol, predictions_bel, carslist, voorspeldatum, max_charge_rate, min_charge_rate,
                      testdatums, data):
    optimized_schedule = Gurobi_optimizer(predictions_sol[testdatums.index(voorspeldatum)],
                                          predictions_bel[testdatums.index(voorspeldatum)], carslist, max_charge_rate,
                                          min_charge_rate)
    bel_real = list(
        data["belpex"][voorspeldatum + dt.timedelta(hours=8, minutes=15):voorspeldatum + dt.timedelta(hours=18)])
    sun_real = list(
        data["solar"][voorspeldatum + dt.timedelta(hours=8, minutes=15):voorspeldatum + dt.timedelta(hours=18)])
    sun_pred = predictions_sol[testdatums.index(voorspeldatum)][33:73]
    bel_pred = predictions_bel[testdatums.index(voorspeldatum)][33:73]
    plot_optimized_schedule(optimized_schedule, carslist, max_charge_rate, sun_real, 'schedulereal.png','Optimized schedule and real data')
    plot_optimized_schedule(optimized_schedule, carslist, max_charge_rate, sun_pred, 'schedulepredicted.png','Optimized schedule and predicted data')
    plot_relative_real(optimized_schedule, sun_real, bel_real, carslist, max_charge_rate)
    plot_relative_predicted(optimized_schedule,sun_pred,bel_pred,carslist,max_charge_rate)
    plot_predicted_price(bel_pred, bel_real)
    plot_predicted_sun(sun_pred, sun_real)
    return optimized_schedule

def daymaker(lijst):
    temporary = []
    for i in lijst:
        temporary.append([0 for x in range(14)] + list(i) + [0 for x in range(16)])
    return temporary



def dataset_separator(df, testdays, data=G.DATA):
    day = 60 * 60 * 24
    year = 365.2425 * day

    df['Day sin'] = np.sin(df['Seconds'] * (2 * np.pi / day))
    df['Day cos'] = np.cos(df['Seconds'] * (2 * np.pi / day))
    df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
    df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
    df = df.drop('Seconds', axis=1)

    max_val = 1340
    max_day = max_val
    X, y = df_to_X_y(df)
    start, W, V, Yo, Z = start_and_finish()
    begintestset = [testday - dt.timedelta(days=1) for testday in testdays]
    dagen = [(A - start).days for A in begintestset] + [max_day]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    previous = 0
    for day in dagen:
        t1 = X[previous:day]
        t2 = y[previous:day]
        if day != max_day:
            t3 = [X[day]]
            t4 = [y[day]]
            X_test.append(t3)
            y_test.append(t4)
        X_train.append(t1)
        y_train.append(t2)

        previous = day + 1

    #X_val, y_val = X[max_day:max_val], y[max_day:max_val]
    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    #X_val, y_val, in het mideen
    return X_train, y_train, X_test, y_test



def selfrelu(lijst):
    antw = []
    for i in lijst:
        temp_var = []
        for j in i:
            temp_var.append(max(0,j))
        antw.append(temp_var)
    return antw



def lstm_definer(train,testdatums, belpex_layers,n_features_bel,bel_epochs,solar_layers,n_features_sol,sol_epochs):
    belmode = 0
    solmode = 1

    # structuur van het neuraal netwerk
    solar_df = pd.DataFrame({'Power': G.SOLAR_SERIES})
    solar_df['Seconds'] = solar_df.index.map(pd.Timestamp.timestamp)

    # In[4]:
    # X2_val, y2_val,
    X2_train, y2_train, X2_test, y2_test = dataset_separator(solar_df, testdatums)
    print(X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape)

    # In[5]:

    solar_training_mean = np.mean(X2_train[:, :, 0])
    solar_training_std = np.std(X2_train[:, :, 0])

    # In[6]:

    X2_train = preprocess(X2_train, solar_training_mean, solar_training_std)
    #X2_val = preprocess(X2_val, solar_training_mean, solar_training_std)
    X2_test = preprocess(X2_test, solar_training_mean, solar_training_std)

    # In[7]:

    belpex_df = pd.DataFrame({'Power': G.BELPEX_SERIES})
    belpex_df['Seconds'] = belpex_df.index.map(pd.Timestamp.timestamp)

    # In[8]:
    # X1_val, y1_val,
    X1_train, y1_train, X1_test, y1_test = dataset_separator(belpex_df, testdatums)

    # In[9]:

    belpex_training_mean = np.mean(X2_train[:, :, 0])
    belpex_training_std = np.std(X2_train[:, :, 0])

    # In[10]:

    X1_train = preprocess(X1_train, belpex_training_mean, belpex_training_std)
    #X1_val = preprocess(X1_val, belpex_training_mean, belpex_training_std)
    X1_test = preprocess(X1_test, belpex_training_mean, belpex_training_std)

    belpex_model = lstm_modelcreator(belpex_layers - 1, n_features_bel, belmode)
    solar_model = lstm_modelcreator(solar_layers - 1, n_features_sol, solmode)
    if train =="y":

        belpex_model, mse = train_lstmmodel(belpex_model, X1_train, y1_train, epoch=bel_epochs)
        belpex_model.save_weights('BelpexPrevious.h5')
        solar_model, mse = train_lstmmodel(solar_model, X2_train, y2_train, epoch=sol_epochs)
        solar_model.save_weights('SolarPrevious.h5')
    else:
        #belpex_model.load_weights('BelpexPrevious.h5')
        #solar_model.load_weights('SolarPrevious.h5')

        solar_model= load_model('FinalSolarSaved/')
        belpex_model = load_model('FinalBelpexSaved/')

    belpex_predictions = daymaker(belpex_model.predict(X1_test))

    # In[14]:

    solar_predictions = selfrelu(daymaker(solar_model.predict(X2_test)))
    y1_test=daymaker(y1_test)
    y2_test=daymaker(y2_test)


    return solar_predictions, belpex_predictions, y1_test,y2_test

def seq_definer(train,testdatums,epochs_bel,epochs_sol,activ_bel,activationfunctions_sol,nodesperlayer_bel,nodesperlayer_sol,begin=0,daysahead =1,testset_size=1,days=1):
    # data uit het bestand importeren en verwerken
    # grootte van inputs(trainingselementen) of dus voorspellen via dagen, weken, maanden op voorhand(30 is max)
    daysahead = 1
    # grootte van testset (aantal ongetrainde dagen)
    test_setsize = 1

    days = 1
    # vanaf welke dag van de testset wilt men weergeven
    begin = 0
    # structuu
    data = initiate_database()
    # uitschieters uit de data halen
    data.solar = mean_data(data.solar)
    data.belpex = mean_data(data.belpex)

    # crëert trainingset(X,Y) en testset(X,Y), n_features is de nodelijst ==> input: (type_model,data ,size, test_setsize, nodesperlayer)
    # type_model is "solar"(0) of "belpex"(1)
    type_model_sol = 0
    X_sol, y_sol, testset_sol, nodesperlayer_sol = dataset_creator(type_model_sol, data, daysahead, test_setsize,
                                                                   nodesperlayer_sol, testdatums)
    type_model_bel = 1
    X_bel, y_bel, testset_bel, nodesperlayer_bel = dataset_creator(type_model_bel, data, daysahead, test_setsize,
                                                                   nodesperlayer_bel, testdatums)
    # model met eigenschappen ==> input: (hidden layers + output layers,n_features, activations)
    model_sol = modelcreator("sequential", len(activationfunctions_sol) - 1, nodesperlayer_sol, activationfunctions_sol)
    model_bel = modelcreator("sequential", len(activ_bel) - 1, nodesperlayer_bel, activ_bel)
    model_sol.summary()
    model_bel.summary()
    # train model met variabelen ==> input: (model, X, y, type_model, epochs, batch_size, reset, displaymodel)
    # data or reset = 1 will display data or reset the model
    batch_size_sol = 16
    reset_sol = 0  # 1 als je model wilt resetten
    display_sol = 1  # 1 als je model wilt displayen
    batch_size_bel = 16
    reset_bel = 0  # 1 als je model wilt resetten
    display_bel = 1  # 1 als je model wilt displayen
    if train == "y":
        model_sol, mse, score = train_model(model_sol, X_sol, y_sol, type_model_sol, epochs_sol, batch_size_sol, reset_sol,display_sol)
        model_bel, mse, score = train_model(model_bel, X_bel, y_bel, type_model_bel, epochs_bel, batch_size_bel,reset_bel, display_bel)
    else:

        model_sol = load_model('SolarSeq/', compile=False)
        model_bel = load_model('BelpexSeq/', compile=False)


    # voorspelt een bepaalde dataset, bijvoorbeeld training data ==> input: (model, inp)
    predictions_sol = selfrelu(predictor(model_sol, testset_sol))
    predictions_bel = predictor(model_bel, testset_bel)
    return predictions_sol,predictions_bel

def modelcreator(modeltype,amount,n_features, activations):
    if modeltype == "sequential":
        return sequential_modelcreator(amount, n_features, activations)
    if modeltype == "RNN":
        pass

def MachineLearningGUI(predictions_sol,predictions_bel, testdatums, data=G.DATA):
    sg.set_options(text_justification='right')
    sg.theme('DarkTeal7')
    carslist = []
    carslist_GUI = []
    possible_days = [dt.date(2016, 6, 25), dt.date(2016, 8, 15), dt.date(2016, 12, 8), dt.date(2017, 4, 1),
                     dt.date(2017, 5, 20), dt.date(2017, 11, 11), dt.date(2018, 1, 3), dt.date(2018, 7, 13),
                     dt.date(2018, 10, 26), dt.date(2019, 2, 3), dt.date(2019, 5, 24), dt.date(2019, 9, 20)]

    command_line_parms = [[sg.Text('Starting hour', size=(10, 1), pad=((7, 3))),
                           sg.Spin(values=[i for i in range(8, 19)], initial_value=8, size=(10, 1)),
                           sg.Text('Ending hour', size=(9, 1)),
                           sg.Spin(values=[i for i in range(8, 19)], initial_value=18, size=(10, 1))
                              , sg.Text('Capacity', size=(7, 1)),
                           sg.Spin(values=[i for i in range(0, 1000)], initial_value=60, size=(10, 1))]]

    date_params = [[sg.Combo(possible_days)]]

    # date_params =  [[sg.Text('Month', size=(8, 1), pad=((7, 3))), sg.Spin(values=[i for i in range(1, 13)], initial_value=1, size=(10, 1)),
    # sg.Text('Day', size=(8, 1)), sg.Spin(values=[i for i in range(1, 32)], initial_value=18, size=(10, 1))
    # ,sg.Text('Year', size=(8, 1)), sg.Spin(values=[i for i in range(2016, 2021)], initial_value=2016, size=(10, 1))]]

    optimize_params = [[sg.Combo(possible_days),
                        sg.Text('Max Charge Rate', size=(14, 1), pad=((7, 3))),
                        sg.Spin(values=[i for i in np.arange(1, 30, 0.25)], initial_value=12.5, size=(10, 1)),
                        sg.Text('Min Charge Rate', size=(14, 1)),
                        sg.Spin(values=[i for i in np.arange(1, 30, 0.25)], initial_value=1.25, size=(10, 1))
                        ]]

    first_column = [
        [sg.Frame('Schedule parameters',
                  [  # [sg.Frame('Date',date_params, title_color='black',font = 'Any 12')],
                      [sg.Frame('Parameters (kWh/15 minutes)', optimize_params, title_color='black', font='Any 12')]])],
        [sg.Frame('Car details', command_line_parms, title_color='black', font='Any 12')],
        [sg.Button('Generate Schedule'), sg.Button('Reset Schedule'), sg.Button("Add car"), sg.Button("Delete car"),
         sg.Button('Show legend'), sg.Exit()],
        [sg.Table(values=carslist_GUI, headings=["Car", "Capacity", "Starting hour", "Ending hour"], key='-TABLE-',
                  size=(10000, 40), auto_size_columns=False, max_col_width=55),
         sg.Image(filename='legend.png', visible=False, key='-LEGEND-', size=(100, 200))],
    ]

    second_column = [[sg.Button('Real sun and schedule', visible=False, key="Real sun and schedule"),
                      sg.Button('Predicted sun and schedule', visible=False, key='Predicted sun and schedule'),
                      sg.Button('Normalised schedule real', visible=False, key="Normalised schedule real")],
                     [sg.Button('Normalised schedule predicted', visible=False, key="Normalised schedule predicted"),
                      sg.Button('Sun predictions', visible=False, key="Sun predictions"),
                      sg.Button('Belpex predictions', visible=False, key="Belpex predictions")],
                     [sg.Image(filename='schedulereal.png', visible=False, key='-REALSCHEDULE-'),
                      sg.Image(filename='schedulepredicted.png', visible=False, key='-PREDICTEDSCHEDULE-'),
                      sg.Image(filename='normalisedschedulereal.png', visible=False, key='-NORMALISEDSCHEDULEREAL-'),
                      sg.Image(filename='normalisedschedulepred.png', visible=False, key='-NORMALISEDSCHEDULEPRED-'),
                      sg.Image(filename='belpexcomparison.png', visible=False, key='-BELPEXCOMPARISON-'),
                      sg.Image(filename='suncomparison.png', visible=False, key='-SUNCOMPARISON-')],
                     [sg.Button('Compare with naive schedule', visible=False, key='-COMPARE-')],
                     [sg.Text(visible=False, key='-COMPTEXT-')]
                     ]

    layout = [[sg.Column(first_column),
               sg.VSeperator(pad=(0, 0)),
               sg.Column(second_column)
               ]]

    sg.set_options(text_justification='left')

    window = sg.Window('GOOEY', layout, font=("Helvetica", 12), size=(1000, 600), resizable=True).Finalize()
    window.Maximize()
    while True:
        event, values = window.read()
        if event is not None:
            if event in (sg.WIN_CLOSED, 'Exit', 'Cancel'):
                break
            if event == 'Generate Schedule':
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=False, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=False, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')
                window['-COMPARE-'].Update(visible=False)
                window['-COMPTEXT-'].Update(visible=False)
                window['-LEGEND-'].Update(visible=False)
                if len(carslist) > 0:
                    if type(values[0]) == dt.date:
                        # do prediction and optimization given cars=carslist and max_charge_rate = values[3],...
                        # prediction with date, returns predicted solar and belpex data for said date
                        datum = values[0]
                        datum = dt.datetime(datum.year, datum.month, datum.day, 0, 0, 0)
                        # optimize schedule for given predictions
                        max_charge_rate = values[1]
                        min_charge_rate = values[2]
                        # schedule = Gurobi_optimizer([],[],carslist, max_charge_rate, 50000, min_charge_rate)
                        # plot_optimized_schedule(schedule,carslist,max_charge_rate)
                        optimized_schedule = generate_schedule(predictions_sol, predictions_bel, carslist, datum,
                                                               max_charge_rate, min_charge_rate, testdatums, data)
                        window['-REALSCHEDULE-'].Update(visible=True, filename='schedulereal.png')
                        window['-COMPARE-'].Update(visible=True)
                        window["Real sun and schedule"].Update(visible=True)
                        window["Predicted sun and schedule"].Update(visible=True)
                        window["Normalised schedule real"].Update(visible=True)
                        window["Normalised schedule predicted"].Update(visible=True)
                        window["Sun predictions"].Update(visible=True)
                        window["Belpex predictions"].Update(visible=True)

            if event == 'Reset Schedule':
                window['-REALSCHEDULE-'].Update(visible=False)
                window['-COMPARE-'].Update(visible=False)
                window['-COMPTEXT-'].Update(visible=False)
                window['-LEGEND-'].Update(visible=False)
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULE-'].Update(visible=False, filename='normalisedschedule.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')
                window['-COMPARE-'].Update(visible=False)
                window["Real sun and schedule"].Update(visible=False)
                window["Predicted sun and schedule"].Update(visible=False)
                window["Normalised schedule real"].Update(visible=False)
                window["Normalised schedule predicted"].Update(visible=False)
                window["Sun predictions"].Update(visible=False)
                window["Belpex predictions"].Update(visible=False)
            if event == "Add car":
                carslist_GUI.append([len(carslist) + 1, values[5], values[3], values[4]])
                carslist.append([values[5], (values[3] - 8) * 4, (values[4] - 8) * 4 - 1])
                window['-TABLE-'].Update(values=carslist_GUI)
            if event == "Delete car":
                if len(carslist) > 0:
                    carslist.pop(-1)
                    carslist_GUI.pop(-1)
                window['-TABLE-'].Update(values=carslist_GUI)
            if event == '-COMPARE-':
                # do compare function if all is given
                if window['Real sun and schedule'].visible == True:
                    sun_real = list(
                        data["solar"][datum + dt.timedelta(hours=8, minutes=15):datum + dt.timedelta(hours=18)])
                    price_buy_real = list(
                        data["belpex"][datum + dt.timedelta(hours=8, minutes=15):datum + dt.timedelta(hours=18)])
                    price_sell_real = [x / 2 for x in price_buy_real]
                    window['-COMPTEXT-'].Update(visible=True)
                    optimized_schedule = generate_schedule(predictions_sol, predictions_bel, carslist, datum,
                                                           max_charge_rate, min_charge_rate, testdatums, data)
                    message, CasBex = compare_prices(optimized_schedule, create_naive_schedule(carslist, values[1]), price_buy_real,
                                       price_sell_real, sun_real)
                    window['-COMPTEXT-'].Update(message)
                        
            if event == 'Show legend':
                if window['-LEGEND-'].visible == True:
                    window['-LEGEND-'].Update(visible=False, filename='legend.png')
                else:
                    window['-LEGEND-'].Update(visible=True, filename='legend.png')

            if event == 'Real sun and schedule' or event == "-sunreal-":
                window['-REALSCHEDULE-'].Update(visible=True, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=False, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=False, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')

            if event == 'Predicted sun and schedule' or event == "-sunpred-":
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=True, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=False, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=False, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')

            if event == 'Normalised schedule real':
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=True, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=False, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')

            if event == 'Normalised schedule predicted':
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=False, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=True, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')
                
            if event == 'Sun predictions':
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=False, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=False, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=False, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=True, filename='suncomparison.png')

            if event == 'Belpex predictions':
                window['-REALSCHEDULE-'].Update(visible=False, filename='schedulereal.png')
                window['-PREDICTEDSCHEDULE-'].Update(visible=False, filename='schedulepredicted.png')
                window['-NORMALISEDSCHEDULEREAL-'].Update(visible=False, filename='normalisedschedulereal.png')
                window['-NORMALISEDSCHEDULEPRED-'].Update(visible=False, filename='normalisedschedulepred.png')
                window['-BELPEXCOMPARISON-'].Update(visible=True, filename='belpexcomparison.png')
                window['-SUNCOMPARISON-'].Update(visible=False, filename='suncomparison.png')

    window.close()

    # CustomMeter()