# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:42:33 2019

@author: Maple
"""
# %% 
from scipy.io import loadmat
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D


"""
Receives 2 matrices of x,y,z coordinates and graphs the 
    trajectory of both to compare the predicted against actual.
"""
def plot_trajectory(actual, predicted, file_name):
    fig = plt.figure(figsize=(10,10))
    rho = actual[0,3]
    ax = plt.axes(projection='3d')
    ax.plot3D(actual[:,0], actual[:,1], actual[:,2], c='black')
    ax.plot3D(predicted[:,0], predicted[:,1], predicted[:,2], '-', c = 'blue')
    ax.scatter(actual[0,0], actual[0,1], actual[0,2], c='red', s = 120)
    ax.legend(['Actual Trajectory','Predicted Trajectory', 'Starting Point'])
    ax.set_title(r'Trajectory Comparisons, $\rho = {}$'.format(rho), fontsize=15, pad=40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    n = np.shape(actual)[0]
    ax1 = fig.add_axes([1, 0.7, 0.4, 0.2])
    ax1.plot(np.arange(n), actual[:,0])
    ax1.plot(np.arange(n), predicted[:,0])
    ax1.legend(['Actual', 'Predicted'])
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('x')
    ax1.set_title('x Trajectories')
    
    ax2 = fig.add_axes([1, 0.43, 0.4, 0.2])
    ax2.plot(np.arange(n), actual[:,1])
    ax2.plot(np.arange(n), predicted[:,1])
    ax2.legend(['Actual', 'Predicted'])
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('y')
    ax2.set_title('y Trajectories')
    
    ax3 = fig.add_axes([1, 0.15, 0.4, 0.2])
    ax3.plot(np.arange(n), actual[:,2])
    ax3.plot(np.arange(n), predicted[:,2])
    ax3.legend(['Actual', 'Predicted'])
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('z')
    ax3.set_title('z Trajectories')
    plt.savefig(file_name,bbox_inches="tight")
    plt.show()
    
#%% 
# Load Lorenz Data
dat = loadmat('lorenz.mat')

inp = dat['input']
out = dat['output']

# %% Initialize Neural Net
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=4))
model.add(Dense(128, activation='relu'))
model.add(Dense(3))
 
model.compile(loss='mse',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['mse'])

# Train Data
model.fit(inp, out, 
          epochs=50,
          batch_size=32)


#%% Load Test Data for rho = 17,35
test_dat = loadmat('lorenz_test.mat')
input17_test = test_dat['input2'][0:799]
output17_test = test_dat['output2'][0:799]

input35_test = test_dat['input2'][800:]
output35_test = test_dat['output2'][800:]


# Use NN to predict trajectory 
output17_pred = np.zeros(np.shape(output17_test))
output35_pred = np.zeros(np.shape(output35_test))
output17_pred[0,:] = input17_test[0,0:3]
output35_pred[0,:] = input35_test[0,0:3]

for i in np.arange(1,np.shape(input17_test)[0]):
    output17_pred[i,:] = model.predict(np.asarray([np.append(output17_pred[i-1,:], 17)]))
    output35_pred[i,:] = model.predict(np.asarray([np.append(output35_pred[i-1,:], 17)]))

# Plot Predicted vs Actual
plot_trajectory(input17_test, output17_pred, 'rho17.png')
plot_trajectory(input35_test, output35_pred, 'rho35.png')

#%% Predict Transistion for r = 28
dat28 = inp[120000:240000, 0]

dat28_reshape = np.zeros([800, 150])
for k in np.arange(1,151):
    dat28_reshape[:,k-1] = dat28[(k-1)*800:(k)*800]

# %%
out28 = np.zeros(np.shape(dat28_reshape))
for j in np.arange(150):
    for k in np.arange(799):
        aa =  dat28_reshape[k, j]
        bb = dat28_reshape[k+1, j]
        if aa*bb < 0:
            out28[k+1, j] = 0
        else:
            out28[k+1, j] = out28[k, j] + 1
            


# %% Create output data
inp28_train = np.transpose(dat28_reshape[:,0:148])
inp28_test = dat28_reshape[:, -1]

out28_train = np.transpose(out28[:, 0:148])
out28_test = out28[:, 149]

# %%
model28 = Sequential()
model28.add(Dense(1024, activation='relu', input_dim=800))
model28.add(Dense(1024, activation='relu'))
model28.add(Dense(800))

model28.compile(loss='mse',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['mse'])

# %%
model28.fit(inp28_train, out28_train,
              epochs = 200,
              batch_size=32)

# %%
out28_pred = model28.predict([[inp28_test]])

# %%
#plt.plot(inp28_test)
plt.figure(figsize=(10,5))
plt.plot(out28_test)
plt.plot(np.arange(800),np.transpose(out28_pred))
plt.title('Time Until Lobe Tranistion')
plt.ylim(-50,400)
plt.ylabel('Time (Wait in Time Steps)')
plt.xlabel('Time')
plt.savefig('b.png')
plt.show()



#%% 
# load KS data
# For the KS data - the uu matrix is matrix where one column is one time shot
#   and the row is the value at that position x
Ks_dat = loadmat('kuramoto_sivishinky.mat')
outks = Ks_dat['output']
inpks = Ks_dat['input']
tt = Ks_dat['tt']
x = Ks_dat['x']


# %% Initialize NN

modelks = Sequential()
act = 'relu'
modelks.add(Dense(256, activation=act, input_dim=np.shape(x)[0]))
modelks.add(Dense(256, activation=act))
modelks.add(Dense(np.shape(x)[0]))
 
# compile model
modelks.compile(loss='mse',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['mse'])

#%%
# Train Data
modelks.fit(np.transpose(inpks), np.transpose(outks), 
          epochs=30,
          batch_size=32)

# %% 
# Load test data
ks_test = loadmat('ks_test.mat')
inks_test = ks_test['input_test']
outks_test = ks_test['output_test']

# Predict
outks_pred = np.zeros(np.shape(outks_test))
x0 = np.asarray([inks_test[:,0]])

for k in np.arange(np.shape(inks_test)[1]):
    x0 = modelks.predict(x0)
    outks_pred[:,k] = x0

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(np.transpose(outks_test), aspect='auto', origin = 'lower')
ax[0].set_title('Actual')
ax[0].set_ylim((0, 250))
ax[0].set_ylabel('Time t')
ax[0].set_xlabel('Spatial x')
ax[1].imshow(np.transpose(outks_pred), aspect='auto', origin = 'lower')
ax[1].set_title('Predicted')
ax[1].set_ylim((0, 250))
ax[1].set_ylabel('Time t')
ax[1].set_xlabel('Spatial x')
plt.savefig('ks.png')
plt.show()


#%% Load reaction diffusion data
datdf = loadmat('reaction_diffusion_big.mat')
U = datdf['u']
V = datdf['v']
t = np.shape(datdf['t'])[0]
x = np.shape(datdf['x'])[1]

# %% Reshape Data
reshape = np.zeros([t, 2*x**2])
for tt in np.arange(t):
    u_frame = U[:,:,tt]
    u_frame_reshape = np.reshape(u_frame,[1, x ** 2])
    v_frame = V[:,:,tt]
    v_frame_reshape = np.reshape(v_frame,[1, x ** 2])
    reshape[tt, 0:(x**2)] = u_frame_reshape
    reshape[tt, (x**2):] = u_frame_reshape
    

# %%
reshape_train = reshape[0:201, :]
reshape_test = reshape[201:241, :]
u, s, vh = la.svd(np.transpose(reshape_train), full_matrices = False)

# %% Graph Singular values
plt.figure(figsize=(10,5))
plt.plot(s/np.sum(s), 'r.', markersize = 10)
plt.title('Amount of Variation in Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Variation')
plt.ylim([0,1])
plt.savefig('sv.png')

# %% 
rank = np.arange(0,2)
u_r = u[:,rank]
v_r = vh[rank, :]
s_r = np.diag(s[rank])


vs_mat = s_r@v_r

inpdf = np.transpose(vs_mat[:,0:200])
outdf = np.transpose(vs_mat[:,1:201])


# %% 
modelrd = Sequential()
act = 'relu'
modelrd.add(Dense(128, activation=act, input_dim=2))
modelrd.add(Dense(128, activation=act))
modelrd.add(Dense(2))
 
# compile model
modelrd.compile(loss='mse',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['mse'])

# %%
modelrd.fit(inpdf, outdf, 
          epochs=500,
          batch_size=32)

# %% 
frameuv = np.transpose(np.transpose(u_r)@reshape_test[0,:])
frameuvdt_pred = modelrd.predict([[frameuv]])
frameuvdt_pred2 = u_r@np.transpose(frameuvdt_pred)

# %%
frameuvdt_true = reshape_test[1,:]
frame = np.reshape(reshape_test[0,0:(x**2)], (x,x))
framedt = np.reshape(reshape_test[1,0:(x**2)], (x,x))
framedt_pred = np.reshape(frameuvdt_pred2[0:(x ** 2), :], (x,x))

fig, ax = plt.subplots(1,3, figsize=(10,3))
plt.tight_layout()
ax[0].imshow(frame)
ax[0].set_title('U at t = 10')
ax[0].axis('off')
ax[1].imshow(framedt)
ax[1].set_title('U at t = 10 + dt Actual')
ax[1].axis('off')
ax[2].imshow(framedt_pred)
ax[2].set_title('U at t = 10 + dt Predicted')
ax[2].axis('off')
plt.savefig('df.png')
plt.show()



