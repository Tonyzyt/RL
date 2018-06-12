import numpy as np

WORLD_SIZE=4
actionProb=0.25
REWARD=-1

world=np.zeros((WORLD_SIZE,WORLD_SIZE))
actions=[0,1,2,3]

def Next(i,j,action):
    if action==0:
        if i==0:
            pass
        else:
            i-=1
    if action==1:
        if i==3:
            pass
        else:
            i+=1
    if action==2:
        if j==0:
            pass
        else:
            j-=1
    if action==3:
        if j==3:
            pass
        else:
            j+=1   

    return i,j


states=[]
for i in range(0,WORLD_SIZE):
    for j in range(0,WORLD_SIZE):
        if (i==0 and j==0) or (i==3 and j==3):
            continue
        else:
            states.append([i,j])


# while True:
for _ in range(0,1000):
    newWorld=np.zeros((WORLD_SIZE,WORLD_SIZE))
    for i,j in states:
        for action in actions:
            newPos=Next(i,j,action)
            newWorld[i,j]+=actionProb*(REWARD+world[newPos[0],newPos[1]])
    # if np.sum(np.abs(world-newWorld)) < 1e-4:
    #     print('Random Policy')
    #     print(newWorld)
    #     break
    world = newWorld

print(world)