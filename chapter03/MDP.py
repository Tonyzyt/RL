import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

WORLD_SIZE=5
A_Pos=[0,1]
Next_A=[4,1]
B_Pos=[0,3]
Next_B=[2,3]
discount=0.9

world=np.zeros((WORLD_SIZE,WORLD_SIZE))

actions=[0,1,2,3]

actionProb=[]

for i in range(0,WORLD_SIZE):
    actionProb.append([])
    for j in range(0,WORLD_SIZE):
        actionProb[i].append([0.25,0.25,0.25,0.25])


def Next(i,j,action):
    state=[i,j]
    reward=0
    if action==0:
        if i==0:
            state=[i,j]
            reward=-1.0
        else:
            state=[i-1,j]
            reward=0.0
    if action==1:
        if i==4:
            state=[i,j]
            reward=-1.0
        else:
            state=[i+1,j]
            reward=0.0
    if action==2:
        if j==0:
            state=[i,j]
            reward=-1.0
        else:
            state=[i,j-1]
            reward=0.0
    if action==3:
        if j==4:
            state=[i,j]
            reward=-1.0
        else:
            state=[i,j+1]
            reward=0.0
    if [i,j]==A_Pos:
        state=Next_A
        reward=10.0
        
    if [i,j]==B_Pos:
        state=Next_B
        reward=5.0

    return state,reward

def draw_image(image):
    fig,ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(image):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = 'white'

        tb.add_cell(i, j, width, height, text=val, 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(range(len(image))):
        tb.add_cell(i, -1, width, height, text=label+1, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(range(len(image))):
        tb.add_cell(-1, j, width, height/2, text=label+1, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    plt.show()

while True:
    newWorld=np.zeros((WORLD_SIZE,WORLD_SIZE))
    for i in range(0,WORLD_SIZE):
        for j in range(0,WORLD_SIZE):
            for action in actions:
                newPos,newReward=Next(i,j,action)
                newWorld[i,j]+=actionProb[i][j][action]*(discount*world[newPos[0],newPos[1]]+newReward)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Random Policy')
        draw_image(np.round(newWorld, decimals=2))
        break
    world=newWorld


while True:
    newWorld=np.zeros((WORLD_SIZE,WORLD_SIZE))
    for i in range(0,WORLD_SIZE):
        for j in range(0,WORLD_SIZE):
            values=[]
            for action in actions:
                newPos,newReward=Next(i,j,action)
                values.append(discount*world[newPos[0],newPos[1]]+newReward)
            newWorld[i,j]=np.max(values)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Optimal Policy')
        draw_image(np.round(newWorld, decimals=2))
        break
    world=newWorld
