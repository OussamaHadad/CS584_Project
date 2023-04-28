import numpy as np

def HR(rank: int, K: int):
    return 1.*(rank <= K)

def NDCG(rank: int, K: int):
    if (rank <= K):
        return 1./(1. + np.log(rank))
    else:
        return 0.
    
def evalModel(modelPredictions: np.array, numUsers: int, K= 10):
    # modelPrediction is an array with the predictions of all the users
    prediction= modelPredictions.reshape(-1)
    assert len(prediction) == 101*numUsers
    hr= []
    ndcg= []
    for i in range(numUsers):
        userPredictions= prediction[i*101: (i+1)*101]
        positiveInteraction= userPredictions[0]
        userPredictions= np.sort(userPredictions, kind='quicksort')[::-1]
        rank= np.where(userPredictions == positiveInteraction)[0][0]+ 1 # add 1 since index starts from 0 not 1
        hr.append(HR(rank, K= K))
        ndcg.append(NDCG(rank, K= K))
    hr= np.array(hr)
    ndcg= np.array(ndcg)
    return np.mean(hr), np.mean(ndcg)