# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:40:15 2019

@author: Maple
"""

# %%
# Load Data
from scipy.io import loadmat
import numpy as np

dat = loadmat('BZ_reformat.mat')
BZ_original = np.array(dat['BZ_tensor'])

BZx = BZ_original[150:200, 190:240, :];

[x,y,time] = np.shape(BZx)
dt = 1;
dx = 1;     
m = x*y
BZr = np.zeros([x*y, time])
for t in np.arange(time):
    BZr[:,t] = np.reshape(BZx[:,:,t], (x*y,))
    
# %% 
BZdot = np.zeros(np.shape(BZx))
for i in np.arange(x):
    for j in np.arange(y):
        for k in np.arange(time)[1:-1]:
            BZdot[i,j,k] = (BZx[i,j,k+1] - BZx[i,j,k-1]) / (2*dt)
        
            
            
#%% 
BZrdot = np.zeros(np.shape(BZr))
for t in np.arange(time)[1:-1]:
    for n in np.arange(x*y):
        BZrdot[n, t] = (BZr[n, t+1] - BZr[n, t-1]) / (2*dt)
        
BZrdot = BZrdot[:, 1:-1]

# %% Create Derivatives
m = x*y;
D = np.zeros([m,m])
D2 = np.zeros([m,m])

for j in np.arange(m-1):
  D[j,j+1]=1;
  D[j+1,j]=-1;

  D2[j,j+1]=1;
  D2[j+1,j]=1;
  D2[j,j]=-2;


D[m-1,1]=1
D[1,m-1] = -1

D2[m-1,m-1]=-2;
D2[m-1,1]=1;
D2[1,m-1]=1;
D2 = D2/( dx ** 2);

# %% 
u = np.reshape(np.transpose(BZr[:,1:-1]),[(time-2)*m,1] );

#%%
ux = np.zeros(np.shape(np.transpose(BZr)))
uxx = np.zeros(np.shape(np.transpose(BZr)))
u2x = np.zeros(np.shape(np.transpose(BZr)))

for jj in np.arange(2, time-1):
   ux[jj-1,:] = D@BZr[:,jj]  # u_x
   uxx[jj-1,:] = D2@BZr[:,jj]  # u_xx
   u2x[jj-1,] = D@(BZr[:,jj] ** 2)   # (u^2)_x

# %%
ux1 = ux[1:-1, :]   
uxx1 = uxx[1:-1, :]   
u2x1 = u2x[1:-1, :]   

# %% 
Ux = np.reshape(ux1,[(time-2)*m,1])
Uxx = np.reshape(uxx1,[(time-2)*m,1])
U2x = np.reshape(u2x1,[(time-2)*m,1])

# %%
A= np.transpose(np.squeeze(np.array([u/u, u, u ** 2, u ** 3, Ux, Uxx, U2x, Ux*u, Ux*Ux, Ux*Uxx])));



# %% 
Udot = np.reshape(BZrdot, [(time-2)*m, 1])


# %%
from sklearn import linear_model as lm
clf = lm.Lasso(alpha=0.8)
clf.fit(A, Udot)
coef = clf.coef_
b = clf.intercept_


# %% 
import matplotlib.pyplot as plt

text = 'Library Terms: \n 1 - Constant \n 2 - u \n 3 - u^2 \n 4 - u ^ 3 \n 5 - u_x \n 6 - u_xx \n 7 - (u^2)_x u*u_x \n 8 - ux^2 \n 9 = u_x*u_xx'
props = props = dict(boxstyle='round', facecolor='white', alpha=0.8)

labels =np.arange(len(coef))
plt.bar(labels, coef)
plt.ylim(-8e-9,1e-9)
plt.hlines(0, -1, len(coef)+1)
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.xticks(labels + 1)
plt.xlim(0,len(coef) + 1)
plt.title('Coefficient')
plt.text(11.4, -5.2e-9,text, fontsize=12, horizontalalignment='left', bbox = props)
plt.savefig('lt.png', bbox_inches="tight")
plt.show()












