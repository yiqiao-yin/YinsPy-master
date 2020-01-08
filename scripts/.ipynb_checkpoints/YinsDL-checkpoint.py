class YinsDL:

    """
    Yin's Deep Learning Package 
    Copyright © YINS CAPITAL, 2009 – Present
    """

   # Define function
    def NN3_Classifier(
        X_train, y_train, X_test, y_test, 
        l1_act='relu', l2_act='relu', l3_act='softmax',
        layer1size=128, layer2size=64, layer3size=2,
        num_of_epochs=10):
        
        """
        MANUAL:
        
        # One can use the following example.
        house_sales = pd.read_csv('../data/kc_house_data.csv')
        house_sales.head(3)
        house_sales = house_sales.drop(['id', 'zipcode', 'lat', 'long', 'date'], axis=1)
        house_sales.info()

        X_all = house_sales.drop('price', axis=1)
        y = np.log(house_sales.price)
        y_binary = (y > y.mean()).astype(int)
        y_binary
        X_all.head(3), y_binary.head(3)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_binary, test_size=0.3, random_state=0)
        print(X_train.shape, X_test.shape)
        print(y_train)

        testresult = DL_NN_Classification(X_train, y_train, X_test, y_test, 
                                 l1_act='relu', l2_act='relu', l3_act='softmax',
                                 layer1size=128, layer2size=64, layer3size=2,
                                 num_of_epochs=50)
        """

        # TensorFlow and tf.keras
        import tensorflow as tf
        from tensorflow import keras

        # Helper libraries
        import numpy as np
        import matplotlib.pyplot as plt

        print(tf.__version__)

        # Normalize
        # Helper Function
        def helpNormalize(X):
            return (X - X.mean()) / np.std(X)

        X_train = X_train.apply(helpNormalize, axis=1)
        X_test = X_test.apply(helpNormalize, axis=1)

        # Model
        model = tf.keras.Sequential([
            keras.layers.Dense(units=layer1size, input_shape=[X_train.shape[1]]),
            keras.layers.Dense(units=layer2size, activation=l2_act),
            keras.layers.Dense(units=layer3size, activation=l3_act)
        ])

        # Compile
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Model Fitting
        model.fit(X_train, y_train, epochs=num_of_epochs)

        # Prediction
        predictions = model.predict(X_test)

        # Performance
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import pandas as pd
        confusion = confusion_matrix(y_test, np.argmax(predictions, axis=1))
        confusion = pd.DataFrame(confusion)
        test_acc = sum(np.diag(confusion)) / sum(sum(np.array(confusion)))
        """ Code Ends Here"""

        # Output
        return {
            'Data': [X_train, y_train, X_test, y_test],
            'Shape': [X_train.shape, len(y_train), X_test.shape, len(y_test)],
            'Model Fitting': model,
            'Performance': {
                'test_acc': test_acc, 
                'confusion': confusion
            },
        }
    # End of function
    
    
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
        MANUAL
        
        # Load Package
        %run "../scripts/YinsMM.py"
        
        # Run
        tmp = YinsDL.RNN4_Regressor(
                start_date = '2013-01-01',
                end_date   = '2019-12-6',
                tickers    = 'AMD', cutoff = 0.8,
                l1_units = 50, l2_units = 50, l3_units = 50, l4_units = 50,
                optimizer = 'adam', loss = 'mean_squared_error',
                epochs = 50, batch_size = 64,
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
            print(f'Root Mean Square Error is {round(rmse,2)} for test set.')
            print(f'Interpretation: ---------------')
            print(f'On the test set, the performance of this LSTM architecture guesses ')
            print(f'{tickers[0]} stock price on average within the error of ${round(rmse,2)} dollars.')

        # Output
        return {
            'Information': [training_set.shape, testing_set.shape],
            'Data': [X_train, y_train, X_test, y_test],
            'Test Response': [predicted_stock_price, real_stock_price],
            'Test Error': rmse
        }
    # End function