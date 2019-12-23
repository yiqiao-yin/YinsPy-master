class InteractionBasedLearning:
    # Define function
    def iscore(X, y):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random

        # Create Partition
        partition = X.iloc[:, 0].astype(str)
        if X.shape[1] >= 2:
            for i in range(X.shape[1]-1):
                partition = partition.astype(str) + '_' + X.iloc[:, i].astype(str)
        else:
            partition = partition

        # Local Information
        list_of_partitions = pd.DataFrame(partition.value_counts())
        Pi = pd.DataFrame(list_of_partitions.index)
        local_n = pd.DataFrame(list_of_partitions.iloc[:, :])

        # Compute Influence Score:
        import collections
        list_local_mean = []
        Y_bar = y.mean()
        local_mean_vector = []
        grouped = pd.DataFrame({'y': y, 'X': partition})
        local_mean_vector = pd.DataFrame(grouped.groupby('X').mean())
        iscore = np.mean(np.array(local_n).reshape(1, local_n.shape[0]) * np.array((local_mean_vector['y'] - Y_bar)**2))/np.std(y)

        # Output
        return {
            'X': X,
            'y': y,
            'Local Mean Vector': local_mean_vector,
            'Global Mean': Y_bar,
            'Partition': Pi,
            'Number of Samples in Partition': local_n,
            'Influence Score': iscore}
    # End of function
    
    # Define function
    def FSFE_BDA(newX, y, num_initial_draw = 4):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        
        # Random Sampling
        newX = newX.iloc[:, random.sample(range(newX.shape[1]), num_initial_draw)]

        # BDA
        newX_copy = newX
        iscorePath = []
        selectedX = {}
        for j in range(newX_copy.shape[1]-1):
            unit_scores = []
            for i in range(newX.shape[1]):
                unit_scores.append(InteractionBasedLearning.iscore(
                    X=newX.iloc[:, :].drop([str(newX.columns[i])], axis=1), y=y)['Influence Score'])
                #print(i, unit_scores, np.max(unit_scores), unit_scores.index(max(unit_scores)))
            iscorePath.append(np.max(unit_scores))
            to_drop = unit_scores.index(max(unit_scores))
            newX = newX.iloc[:, :].drop([str(newX.columns[to_drop])], axis=1)
            selectedX[str(j)] = newX

        # Final Output
        finalX = pd.DataFrame(selectedX[str(iscorePath.index(max(iscorePath)))])

        # Output
        return {
            'Path': iscorePath,
            'MaxIscore': np.max(iscorePath),
            'newX': finalX,
            'Summary': {
                'Variable Module': np.array(finalX.columns), 
                'Influence Score': np.max(iscorePath)
            }
        }
    # End of function