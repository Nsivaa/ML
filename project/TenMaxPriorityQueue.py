import heapq

import numpy as np


# elem sara una tupla del tipo (mean:float, (variance:float,combination:dict))
# la priority queue mantiene sempre i 10 elem con mean maggiore, inoltre list[0] continene sempre la mean minore tra gli
# per poter mantenere questa proprietà, dato che heapq prevede solo operazioni per minPriorityQueue, il valore mean,
# che utilizziamo come priority, sarà di segno negativo (all'interno della queue), in questo modo il minore sarà
# quello in valore assoluto maggiore (che è quello che eventualmente ci interessa togliere dalla queue)
def push(list, elem):
    mean, (variance,comb) = elem
    new_mean= mean* (-1)
    if len(list)< 10:
        heapq.heappush(list,(new_mean,(variance,comb)))
    elif list[0][0] == new_mean and list[0][1][0] > variance:
        #l'elemento da aggiungere ha la stessa mean ma minore varianza
        heapq.heappop(list)
        heapq.heappush(list,(new_mean,(variance,comb)))
    elif new_mean > list[0][0]:
        heapq.heappushpop(list,(new_mean,(variance,comb)))


def printQueue(list,file=None):
    temp=[]
    for i in range(len(list)):
        mean, (variance,comb) = heapq.heappop(list)
        mean=abs(mean)
        temp.append((mean, (variance,comb)))
    k=1
    for i in range(len(temp)):
        mean, (variance,comb) = temp[i]
        if file != None:
            print(f"{k}. {comb}\nLoss mean = {mean}, Variance = {variance}\n",file=file)
        else:
            print(f"{k}. {comb}\nLoss mean = {mean}, Variance = {variance}\n")
        k+=1
