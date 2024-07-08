import heapq
import numpy as np

'''
elem is a tupla like (val_mean:float, (val_variance:float,tr_err:float,combination:dict))
list[0] always contains the lowest mean among the list, the priority queue always keeps the 10 elements with the highest mean.
To be able to maintain this property, since heapq only provides operations for minPriorityQueue, 
the 'val_mean' value that we use as priority, will be within the queue with negative sign, so the smaller value will be
the one with the highest absolute value (which is the one we are interested in removing from the queue)
'''

def push(list_, elem):
    val_mean, (variance,tr_mean,comb) = elem
    if val_mean == 1000000.0:
        return
    new_mean= val_mean* (-1)
    if len(list_)< 10:
        try:
            heapq.heappush(list_,(new_mean,(variance,tr_mean,comb)))
        except TypeError:
            pass
    elif list_[0][0] == new_mean and list_[0][1][0] > variance:
        #l'elemento da aggiungere ha la stessa mean ma minore varianza
        try:
            heapq.heappop(list_)
            heapq.heappush(list_,(new_mean,(variance,tr_mean,comb)))
        except TypeError:
            pass
    elif new_mean > list_[0][0]:
        try:
            heapq.heappushpop(list_,(new_mean,(variance,tr_mean,comb)))
        except TypeError:
            pass

def printQueue(list_,file=None):
    temp=[]

    for i in range(len(list_)):
        val_mean, (variance,tr_mean,comb) = heapq.heappop(list_)
        val_mean=abs(val_mean)
        temp.append((val_mean, (variance,tr_mean,comb)))
    k=1
    for i in range(len(temp)):
        val_mean, (variance,tr_mean,comb) = temp[len(temp)-1-i]
        if file != None:
            print(f"{k}. {comb}\nValidation mean = {val_mean}, Variance = {variance}\nTraining mean (ES) = {tr_mean}\n",file=file)
        else:
            print(f"{k}. {comb}\nValidation mean = {val_mean}, Variance = {variance}\nTraining mean (ES) = {tr_mean}\n")
        k+=1
