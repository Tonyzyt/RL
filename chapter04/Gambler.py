import numpy as np
import matplotlib.pyplot as plt


prob=0.4
GOAL=100
Reward=1.0

states=np.arange(GOAL)
values=np.zeros(GOAL)
policy=np.zeros(GOAL)

k=0
while True:
    delta=0.0
    for state in states[1:GOAL]:
        actions=np.arange(1,min(state,GOAL-state)+1)
        newvalue=0.0
        for action in actions:
            
            if action+state==GOAL:
                newvalue=max(newvalue,prob*Reward+(1-prob)*values[state-action])
            else:
                newvalue=max(newvalue,prob*values[state+action]+(1-prob)*values[state-action])
        delta+=np.abs(values[state]-newvalue)
        values[state]=newvalue
    # k+=1
    # if k==3:
    #     break
    if delta < 1e-9:
        break

print(values)
            
for state in states[1:GOAL]:
    actions=np.arange(1,min(state,GOAL-state)+1)
    actionreturn=[]
    for action in actions:
        if action+state==GOAL:
           actionreturn.append(prob*Reward+(1-prob)*values[state-action])
        else:
            actionreturn.append(prob*values[state+action]+(1-prob)*values[state-action])
    policy[state]=actions[np.argmax(actionreturn)]

# print(policy)
        
plt.figure(1)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.plot(values)
plt.figure(2)
plt.scatter(states, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.show()