class YinsDL:
    
   # Define function
    def NN3_Classifier(
        X_train, y_train, X_test, y_test, 
        l1_act='relu', l2_act='relu', l3_act='softmax',
        layer1size=128, layer2size=64, layer3size=2,
        num_of_epochs=10):

        """ Code Starts Here"""
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