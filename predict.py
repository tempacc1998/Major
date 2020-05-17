import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def main(file, interval):
    # to delete existing plot
    if os.path.exists('static/plot.png'):
        print(os.remove('static/plot.png'))

    df = pd.read_csv(file)
    forecast_out = int(interval)
    x, y, new_df = getData(df, forecast_out)
    lPredictedValue,sPredictedValue = predict(x, y, new_df, forecast_out)
    actual = df['Adj Close'].tail(forecast_out)

    # Creating a sample Dataframe to compare values
    # pred_df = pd.DataFrame(data=actual, columns=['Actual'])
    # pred_df['Predicted Value'] = predictedValue
    lPV = np.array(lPredictedValue)
    sPV = np.array(sPredictedValue)
    aV = np.array(actual)
    savePlot(aV, lPV,sPV, np.array(df['Date'][-forecast_out:]))
    # pred_df.set_index('Date',inplace=True)
    # pred_df.drop(['Date'],inplace=True)
    return aV, lPV,sPV


# Saving the plot in the directory
def savePlot(aV, lPV,sPV, date):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(date, aV, label='Actual Price')
    ax.set_xlabel('Date')
    ax.plot(date, lPV, label='Predicted Price')
    ax.set_ylabel('Price')
    ax.plot(date, sPV, label='Predicted Price')
    fig.savefig('static/plot.png')


def getData(df, forecast_out):
    df['Adj Close'].fillna(df['Adj Close'].mean(), inplace=True)

    # Create a new df for manipulation/adding/removing coloumns
    new_df = df[['Adj Close']]

    # add column with target/dependent varible shifted by 'forecast' units
    new_df['Prediction'] = new_df['Adj Close'].shift(-forecast_out)

    # Create independent dataset X
    # convert dataset to numpy array
    X = np.array(new_df.drop(['Prediction'], 1))

    # Removing the last 'n' rows
    X = X[:-forecast_out]

    # Create Dependent dataset Y
    # Convert dataset to numpy array

    Y = np.array(new_df['Prediction'])
    Y = Y[:-forecast_out]
    return X, Y, new_df


def predict(x, y, new_df, forecast_out):
    # split the data into 80% train and 20% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Training the model
    # Linear Regression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("Linear regression score: ",lr.score(x_test,y_test))

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    x_forecast = np.array(new_df.drop(['Prediction'], 1))[-forecast_out:]
    l_predict = lr.predict(x_forecast)

    s_predict = svr_rbf.predict(x_forecast)

    return l_predict, s_predict
