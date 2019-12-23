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
    def BDA(X, y, num_initial_draw = 4):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        
        # Random Sampling
        newX = X.iloc[:, random.sample(range(X.shape[1]), num_initial_draw)]

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
                'Influence Score': np.max(iscorePath) },
            'Brief': [[[np.array(finalX.columns)], [np.max(iscorePath)]]]
            }
    # End of function
    
    # Define function
    def InteractionLearning(newX, y, testSize=0.3, 
                                 num_initial_draw=7, total_rounds=10, top_how_many=3, 
                                 verbatim=True):
        # Environment Initiation
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        import time
        
        # Split Train and Validate
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size=testSize, random_state = 0)
        
        # Start Learning
        start = time.time()
        listVariableModule = []
        listInfluenceScore = []
        from tqdm import tqdm
        for i in tqdm(range(total_rounds)):
            oneDraw = InteractionBasedLearning.BDA(X=X_train, y=y_train, num_initial_draw=num_initial_draw)
            listVariableModule.append([np.array(oneDraw['newX'].columns)])
            listInfluenceScore.append(oneDraw['MaxIscore'])
        end = time.time()
        
        # Time Check
        if verbatim == True: print('Time Consumption', end - start)
        
        # Update Features
        listVariableModule_copy = listVariableModule
        listInfluenceScore_copy = listInfluenceScore
        selectedNom = listVariableModule[listInfluenceScore.index(np.max(listInfluenceScore))]
        informativeX = pd.DataFrame(newX[selectedNom[0]])
        listVariableModule_copy = np.delete(listVariableModule_copy, listInfluenceScore_copy.index(np.max(listInfluenceScore)))
        listInfluenceScore_copy = np.delete(listInfluenceScore_copy, listInfluenceScore_copy.index(np.max(listInfluenceScore)))

        for j in range(2, top_how_many):
            selectedNom = listVariableModule_copy[listInfluenceScore_copy.tolist().index(np.max(listInfluenceScore_copy))]
            informativeX = pd.concat([informativeX, pd.DataFrame(newX[selectedNom])], axis=1)
            listVariableModule_copy = np.delete(
                listVariableModule_copy, 
                listInfluenceScore_copy.tolist().index(np.max(listInfluenceScore_copy)))
            listInfluenceScore_copy = np.delete(
                listInfluenceScore_copy, 
                listInfluenceScore_copy.tolist().index(np.max(listInfluenceScore_copy)))
        
        briefResult = pd.DataFrame({'Modules': listVariableModule, 'Score': listInfluenceScore})
        briefResult = briefResult.sort_values(by=['Score'], ascending=False)

        # Output
        return {
            'List of Variable Modules': listVariableModule,
            'List of Influence Measures': listInfluenceScore,
            'Brief': briefResult,
            'New Features': informativeX
        }
    # End of function