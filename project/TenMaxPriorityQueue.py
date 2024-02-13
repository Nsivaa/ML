import heapq

import numpy as np


# elem sara una tupla del tipo (val_mean:float, (val_variance:float,tr_err:float,combination:dict))
# la priority queue mantiene sempre i 10 elem con mean maggiore, inoltre list[0] continene sempre la mean minore tra gli
# per poter mantenere questa proprietà, dato che heapq prevede solo operazioni per minPriorityQueue, il valore mean,
# che utilizziamo come priority, sarà di segno negativo (all'interno della queue), in questo modo il minore sarà
# quello in valore assoluto maggiore (che è quello che eventualmente ci interessa togliere dalla queue)
def push(list_, elem):
    val_mean, (variance,tr_mean,comb) = elem
    if val_mean== 1000000.0: 
        return
    new_mean= val_mean* (-1)
    if len(list_)< 10:
        heapq.heappush(list_,(new_mean,(variance,tr_mean,comb)))
    elif list_[0][0] == new_mean and list_[0][1][0] > variance:
        #l'elemento da aggiungere ha la stessa mean ma minore varianza
        heapq.heappop(list_)
        heapq.heappush(list_,(new_mean,(variance,tr_mean,comb)))
    elif new_mean > list_[0][0]:
        heapq.heappushpop(list_,(new_mean,(variance,tr_mean,comb)))


def printQueue(list_,file=None):
    temp=[]
    for i in range(len(list_)):
        val_mean, (variance,tr_mean,comb) = heapq.heappop(list_)
        val_mean=abs(val_mean)
        temp.append((val_mean, (variance,tr_mean,comb)))
    k=1
    for i in range(len(temp)):
        val_mean, (variance,tr_mean,comb) = temp[i]
        if file != None:
            print(f"{k}. {comb}\nValidation mean = {val_mean}, Variance = {variance}\nTraining mean (ES) = {tr_mean}\n",file=file)
        else:
            print(f"{k}. {comb}\nValidation mean = {val_mean}, Variance = {variance}\nTraining mean (ES) = {tr_mean}\n")
        k+=1
