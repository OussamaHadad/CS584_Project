import pandas as pd
import numpy as np

#1 urls of the datasets
movieLens_train_rating_url = "https://raw.githubusercontent.com/Leavingseason/NeuralCF/master/Data/ml-1m.train.rating"
movieLens_test_rating_url = "https://raw.githubusercontent.com/Leavingseason/NeuralCF/master/Data/ml-1m.test.rating"    
pinterest_train_rating_url = "https://raw.githubusercontent.com/Leavingseason/NeuralCF/master/Data/pinterest-20.train.rating"
pinterest_test_rating_url = "https://raw.githubusercontent.com/Leavingseason/NeuralCF/master/Data/pinterest-20.test.rating"

#2 Import the data
movieLens_train_rating_data = pd.read_csv(movieLens_train_rating_url, sep = '\t', header=None)
movieLens_train_rating_data.columns = ["userID", "itemID", "rating", "timestamp"]

movieLens_test_rating_data = pd.read_csv(movieLens_test_rating_url, sep = '\t', header=None)
movieLens_test_rating_data.columns = ["userID", "itemID", "rating", "timestamp"]

pinterest_train_rating_data = pd.read_csv(pinterest_train_rating_url, sep = '\t', header=None)
pinterest_train_rating_data.columns = ["userID", "itemID", "rating", "timestamp"]

pinterest_test_rating_data = pd.read_csv(pinterest_test_rating_url, sep = '\t', header=None)
pinterest_test_rating_data.columns = ["userID", "itemID", "rating", "timestamp"]

#3 Number of users & items
movieLens_num_users= movieLens_train_rating_data.userID.max() + 1
movieLens_num_items= movieLens_train_rating_data.itemID.max() + 1

pinterest_num_users= pinterest_train_rating_data.userID.max() + 1
pinterest_num_items= pinterest_train_rating_data.itemID.max() + 1

#4 Drop the timestamp column as it is useless for this study
movieLens_train_rating_data= movieLens_train_rating_data.drop(columns="timestamp", axis=1)
movieLens_test_rating_data= movieLens_test_rating_data.drop(columns="timestamp", axis=1)
pinterest_train_rating_data= pinterest_train_rating_data.drop(columns="timestamp", axis=1)
pinterest_test_rating_data= pinterest_test_rating_data.drop(columns="timestamp", axis=1)

#5 Transform data from explicit feedback data to implicit feedback data for MovieLens datasets; i.e: from rating in [0, 5] into rating in {0, 1}, and add negative instances ....
def df_to_dict(df: pd.DataFrame):
    df= df.set_index(['userID', 'itemID'])
    df= df['rating'].to_dict()
    return df

def getTrainSet(dictTrainSet: pd.DataFrame, dictTestSet: pd.DataFrame, numItems: int, numNegatives= 4):
    userID, itemID, rating = [], [], []
    for (u, i) in dictTrainSet.keys():
        # positive instances; i.e: u & i interacted
        userID.append(u)
        itemID.append(i)
        rating.append(1)
        # for every positive interaction we add numNegatives interactions
        for _ in range(numNegatives):
            j = np.random.randint(numItems)
            while (u, j) in dictTrainSet.keys() or (u, j) in dictTestSet.keys(): 
                j = np.random.randint(numItems)
            userID.append(u)
            itemID.append(j)
            rating.append(0)
    return [np.array(userID), np.array(itemID)], np.array(rating)

def getEvaluationSet(dictTrainSet: pd.DataFrame, dictTestSet: pd.DataFrame, numItems: int):
    userID, itemID, rating = [], [], []
    for (u, i) in dictTestSet.keys():   # every user has 1 positive interaction
        userID.append(u)
        itemID.append(i)
        rating.append(1)
        for _ in range(100):    # every user has 100 negative interactions
            j = np.random.randint(numItems)
            while (u, j) in dictTrainSet.keys() or j == i: # (u, j) is a negative interaction
                j = np.random.randint(numItems)
            userID.append(u)
            itemID.append(j)
            rating.append(0)
    return [np.array(userID), np.array(itemID)], np.array(rating)