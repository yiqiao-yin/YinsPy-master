class YinsMM:
    
    """
    Yin's Money Managmeent Package 
    Copyright © YINS CAPITAL, 2009 – Present
    """
    
    # Define function
    def MarkowitzPortfolio(tickers, start_date, end_date):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Input
        start_date = pd.to_datetime('2013-01-01')
        end_date = pd.to_datetime('2019-12-6')
        tickers = ['aapl', 'fb']

        # Run
        temp = YinsMM.MarkowitzPortfolio(tickers, start_date, end_date)
        print('Optimal Portfolio has the following information', testresult['Optimal Portfolio'])
        """
        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function

        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Verbose
        for i in range(len(tickers)):
            print(
                'Normalize Return:', stockData[tickers[i]]['Normalize Return'], 
                '\n',
                'Expected Return:', np.mean(stockData[tickers[i]]['Normalize Return']),
                '\n',
                'Risk', np.std(stockData[tickers[i]]['Normalize Return']),
                '\n',
                'Sharpe Ratio:', np.mean(stockData[tickers[1]]['Normalize Return']) / np.std(stockData[tickers[0]]['Normalize Return']))
        retMatrix = pd.concat([stockData[tickers[0]]['Normalize Return'], stockData[tickers[1]]['Normalize Return']], axis=1, join='inner')

        # Compute the following for Markowitz Portfolio
        w1 = np.linspace(start=0, stop=1, num=50)
        w2 = 1 - w1
        r1 = np.mean(stockData[tickers[0]]['Normalize Return'])
        r2 = np.mean(stockData[tickers[1]]['Normalize Return'])
        sd1 = np.std(stockData[tickers[0]]['Normalize Return'])
        sd2 = np.std(stockData[tickers[1]]['Normalize Return'])
        rho = np.array(retMatrix.corr())[0][1]

        # Compute paths for returns and risks
        returnPath = np.zeros([1, len(w1)])
        riskPath = np.zeros([1, len(w2)])
        for i in range(len(w1)):
            returnPath[0][i] = w1[i] * r1 + w2[i] * r2
            riskPath[0][i] = w1[i]**2 * sd1**2 + w2[i]**2 * sd2**2 + 2*w1[i]*w2[i]*sd1*sd2*rho

        # Optimal Portfolio
        maximumSR = returnPath / riskPath
        maxVal = maximumSR.max()
        for i in range(len(maximumSR[0])):
            if maximumSR[0][i] == maxVal:
                idx = i

        # Visualization
        import matplotlib.pyplot as plt
        marginsize = 1e-5
        data_for_plot = pd.concat({'Return': pd.DataFrame(returnPath), 'Risk': pd.DataFrame(riskPath)}, axis=0).T
        data_for_plot
        data_for_plot.plot(x='Risk', y='Return', kind='scatter', figsize=(15,5))
        plt.plot(riskPath[0][idx], returnPath[0][idx], marker='o', markersize=10, color='green') # insert an additional dot: this is the position optimal portfolio
        plt.xlim([np.min(riskPath) - marginsize, np.max(riskPath) + marginsize])
        plt.ylim([np.min(returnPath) - marginsize, np.max(returnPath) + marginsize])
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')

        # Output
        return {'Return Matrix': retMatrix, 
                'Correlation Matrix': retMatrix.corr(), 
                'Covariance Matrix': retMatrix.cov(), 
                'Parameters': [w1, w2, r1, r2, sd1, sd2, rho], 
                'Return Path': returnPath, 
                'Risk Path': riskPath,
                'Optimal Portfolio': {
                    'Optimal Weight': [w1[idx], w2[idx]], 
                    'Optimal Return': w1[idx] * r1 + w2[idx] * r2, 
                    'Optimal Volatility': w1[idx]**2 * sd1**2 + w2[idx]**2 * sd2**2 + 2*w1[idx]*w2[idx]*sd1*sd2*rho,
                    'Optimal SR': (w1[idx] * r1 + w2[idx] * r2) / (w1[idx]**2 * sd1**2 + w2[idx]**2 * sd2**2 + 2*w1[idx]*w2[idx]*sd1*sd2*rho)
                }}
    # End of function

    
    # Define function
    def YinsTimer(start_date, end_date, ticker, k=20):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Run
        what_stock_do_you_like = 'SPY'
        temp = YinsMM.YinsTimer(start_date, end_date, what_stock_do_you_like, k = 1)
        print('Optimal Portfolio has the following information', temp['Optimal Portfolio'])
        """
        dta = yf.download(ticker, start_date, end_date)
        dta_stock = pd.DataFrame(dta)

        # Define Checking Functions:
        def chk(row):
            if row['aveDIST'] < 0:
                val = row['aveDIST']*(-1)
            else:
                val = 0
            return val

        # Create Features
        df_stock = dta_stock
        close = df_stock['Adj Close']
        df_stock['Normalize Return'] = close / close.shift() - 1
        df_stock['SMA20'] = close.rolling(window=20).mean()
        df_stock['SMA50'] = close.rolling(window=50).mean()
        df_stock['SMA100'] = close.rolling(window=100).mean()
        df_stock['SMA200'] = close.rolling(window=200).mean()
        df_stock['DIST20'] = close / df_stock['SMA20'] - 1
        df_stock['DIST50'] = close / df_stock['SMA50'] - 1
        df_stock['DIST100'] = close / df_stock['SMA100'] - 1
        df_stock['DIST200'] = close / df_stock['SMA200'] - 1
        df_stock['aveDIST'] = (df_stock['DIST20'] + df_stock['DIST50'] + df_stock['DIST100'] + df_stock['DIST200'])*k
        df_stock['Signal'] = df_stock.apply(chk, axis = 1)

        # Plot
        import matplotlib.pyplot as plt
        # No. 1: the first time-series graph plots adjusted closing price and multiple moving averages
        data_for_plot = df_stock[['Adj Close', 'SMA20', 'SMA50', 'SMA200']]
        data_for_plot.plot(figsize = (15,6))
        plt.show()
        # No. 2: the second time-series graph plots signals generated from investigating distance matrix
        data_for_plot = df_stock[['Signal']]
        data_for_plot.plot(figsize = (15,3))
        plt.show()

        # Return
        return {'data': dta_stock, 
                'estimatedReturn': np.mean(dta_stock['Normalize Return']), 
                'estimatedRisk': np.std(dta_stock['Normalize Return'])}
    
    # Define function
    def CAPM(tickers, start_date, end_date):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Run
        start_date = pd.to_datetime('2013-01-01')
        end_date = pd.to_datetime('2019-12-6')
        tickers = ['AAPL', 'SPY']
        testresult = mmCAPM(tickers, start_date, end_date)
        print(testresult['Beta'], testresult['Alpha'])
        """
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function

        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        import matplotlib.pyplot as plt
        target = stockData[tickers[0]]
        benchmark = stockData[tickers[1]]
        target['Cumulative'] = target['Close'] / target['Close'].iloc[0]
        benchmark['Cumulative'] = benchmark['Close'] / benchmark['Close'].iloc[0]
        target['Cumulative'].plot(label=tickers[0], figsize = (15,5))
        benchmark['Cumulative'].plot(label='Benchmark')
        plt.legend()
        plt.title('Cumulative Return')
        plt.show()

        target['Daily Return'] = target['Close'].pct_change(20)
        benchmark['Daily Return'] = benchmark['Close'].pct_change(20)
        plt.scatter(target['Daily Return'], benchmark['Daily Return'], alpha = 0.3)
        plt.xlabel('Target Returns')
        plt.ylabel('Benchmark Returns')
        plt.title('Daily Returns for Target and Benchmark')
        plt.show()

        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        x = np.array(benchmark['Daily Return']).reshape(-1, 1)
        x = np.array(x[~np.isnan(x)]).reshape(-1, 1)
        y = np.array(target['Daily Return']).reshape(-1, 1)
        y = np.array(y[~np.isnan(y)]).reshape(-1, 1)
        linearModel = lm.fit(x, y)
        y_pred = linearModel.predict(x)

        plt.scatter(x, y, alpha=0.3)
        plt.plot(x, y_pred, 'g')
        plt.xlabel('Benchmark')
        plt.ylabel('Target')
        plt.title('Scatter Dots: Actual Target Returns vs. \nLinear (green): Estimated Target Returns')
        plt.show()

        from sklearn.metrics import r2_score
        score = r2_score(y, y_pred)
        print('R-square is:', score)
        RMSE = np.sqrt(np.mean((y - y_pred)**2))
        print('Root Mean Square Error (RMSE):', RMSE)

        return {'Beta': linearModel.coef_, 
                'Alpha': linearModel.intercept_, 
                'Returns': y, 
                'Estimated Returns': y_pred, 
                'R square': score, 
                'Root Mean Square Error': RMSE}
    # End of function
    
    # Define Function
    def RNN3_Regressor(
        start_date = '2013-01-01',
        end_date   = '2019-12-6',
        tickers    = 'AAPL', cutoff = 0.8,
        l1_units = 50, l2_units = 50, l3_units = 50,
        optimizer = 'adam', loss = 'mean_squared_error',
        epochs = 50, batch_size = 64,
        plotGraph = True,
        verbatim = True
    ):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Run
        tmp = YinsDL.RNN4_Regressor(
            start_date = '2013-01-01',
            end_date   = '2019-12-6',
            tickers    = 'FB', cutoff = 0.8,
            l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
            optimizer = 'adam', loss = 'mean_squared_error',
            epochs = 30, batch_size = 64,
            plotGraph = True,
            verbatim = True )
        """
        
        # Initiate Environment
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function
        
        start_date = pd.to_datetime(start_date)
        end_date   = pd.to_datetime(end_date)
        tickers    = [tickers]
        
        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Take a look
        # print(stockData[tickers[0]].head(2)) # this is desired stock
        # print(stockData[tickers[1]].head(2)) # this is benchmark (in this case, it is S&P 500 SPDR Index Fund: SPY)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler

        stockData[tickers[0]].iloc[:, 4].head(3)

        data = stockData[tickers[0]].iloc[:, 4:5].values
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_dta = sc.fit_transform(data)
        scaled_dta = pd.DataFrame(scaled_dta)

        training_set = scaled_dta.iloc[0:round(scaled_dta.shape[0] * cutoff), :]
        testing_set = scaled_dta.iloc[round(cutoff * scaled_dta.shape[0] + 1):scaled_dta.shape[0], :]

        # print(training_set.shape, testing_set.shape)

        X_train = []
        y_train = []

        for i in range(100, training_set.shape[0]):
            X_train.append(np.array(training_set)[i-100:i, 0])
            y_train.append(np.array(training_set)[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        print(X_train.shape, y_train.shape)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)

        X_test = []
        y_test = []

        for i in range(100, testing_set.shape[0]):
            X_test.append(np.array(testing_set)[i-100:i, 0])
            y_test.append(np.array(testing_set)[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        print(X_test.shape, y_test.shape)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(X_test.shape)

        ### Build RNN

        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialize RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units = 1))

        regressor.summary()

        ### Train RNN

        # Compiling the RNN
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

        ### Predictions

        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        real_stock_price = np.reshape(y_test, (y_test.shape[0], 1))
        real_stock_price = sc.inverse_transform(real_stock_price)

        ### Performance Visualization

        # Visualising the results
        import matplotlib.pyplot as plt
        if plotGraph:
            plt.plot(real_stock_price, color = 'red', label = f'Real {tickers[0]} Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {tickers[0]} Stock Price')
            plt.title(f'{tickers[0]} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(f'{tickers[0]} Stock Price')
            plt.legend()
            plt.show()

        import math
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        if verbatim:
            print(f'---------------------------------------------------------------------------------')
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'------------------')
            print(f'Interpretation:')
            print(f'------------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')
            print(f'---------------------------------------------------------------------------------')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function
    
    # Define Function
    def RNN4_Regressor(
        start_date = '2013-01-01',
        end_date   = '2019-12-6',
        tickers    = 'AAPL', cutoff = 0.8,
        l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
        optimizer = 'adam', loss = 'mean_squared_error',
        epochs = 50, batch_size = 64,
        plotGraph = True,
        verbatim = True
    ):
        """
        MANUAL: Try run the following line by line in a Python Notebook
        
        # Load
        %run "../scripts/YinsDL.py"
        
        # Run
        tmp = YinsDL.RNN4_Regressor(
            start_date = '2013-01-01',
            end_date   = '2019-12-6',
            tickers    = 'FB', cutoff = 0.8,
            l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
            optimizer = 'adam', loss = 'mean_squared_error',
            epochs = 30, batch_size = 64,
            plotGraph = True,
            verbatim = True )
        """
        
        # Initiate Environment
        from scipy import stats
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt

        # Define function
        def getDatafromYF(ticker, start_date, end_date):
            stockData = yf.download(ticker, start_date, end_date)
            return stockData
        # End function
        
        start_date = pd.to_datetime(start_date)
        end_date   = pd.to_datetime(end_date)
        tickers    = [tickers]
        
        # Start with Dictionary (this is where data is saved)
        stockData = {}
        for i in tickers:
            stockData[i] = pd.DataFrame(getDatafromYF(str(i), start_date, end_date))
            close = stockData[i]['Adj Close']
            stockData[i]['Normalize Return'] = close / close.shift() - 1

        # Take a look
        # print(stockData[tickers[0]].head(2)) # this is desired stock
        # print(stockData[tickers[1]].head(2)) # this is benchmark (in this case, it is S&P 500 SPDR Index Fund: SPY)

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler

        stockData[tickers[0]].iloc[:, 4].head(3)

        data = stockData[tickers[0]].iloc[:, 4:5].values
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled_dta = sc.fit_transform(data)
        scaled_dta = pd.DataFrame(scaled_dta)

        training_set = scaled_dta.iloc[0:round(scaled_dta.shape[0] * cutoff), :]
        testing_set = scaled_dta.iloc[round(cutoff * scaled_dta.shape[0] + 1):scaled_dta.shape[0], :]

        # print(training_set.shape, testing_set.shape)

        X_train = []
        y_train = []

        for i in range(100, training_set.shape[0]):
            X_train.append(np.array(training_set)[i-100:i, 0])
            y_train.append(np.array(training_set)[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        print(X_train.shape, y_train.shape)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)

        X_test = []
        y_test = []

        for i in range(100, testing_set.shape[0]):
            X_test.append(np.array(testing_set)[i-100:i, 0])
            y_test.append(np.array(testing_set)[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)

        print(X_test.shape, y_test.shape)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(X_test.shape)

        ### Build RNN

        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        # Initialize RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l1_units, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l2_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l3_units, return_sequences = True))
        regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = l4_units))
        regressor.add(Dropout(0.2))

        # Adding the output layer
        regressor.add(Dense(units = 1))

        regressor.summary()

        ### Train RNN

        # Compiling the RNN
        regressor.compile(optimizer = optimizer, loss = loss)

        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

        ### Predictions

        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        real_stock_price = np.reshape(y_test, (y_test.shape[0], 1))
        real_stock_price = sc.inverse_transform(real_stock_price)

        ### Performance Visualization

        # Visualising the results
        import matplotlib.pyplot as plt
        if plotGraph:
            plt.plot(real_stock_price, color = 'red', label = f'Real {tickers[0]} Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {tickers[0]} Stock Price')
            plt.title(f'{tickers[0]} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(f'{tickers[0]} Stock Price')
            plt.legend()
            plt.show()

        import math
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        if verbatim:
            print(f'---------------------------------------------------------------------------------')
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'------------------')
            print(f'Interpretation:')
            print(f'------------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')
            print(f'---------------------------------------------------------------------------------')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function