# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:57:07 2019

@author: Maple
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:58:56 2019

@author: Maple
"""

# %% Load Data
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

dat = loadmat('population.mat')
year = np.squeeze(dat['year'])
year = np.array([float(x) for x in year])
hare = np.squeeze(dat['hare'])
hare = np.array([float(x) for x in hare])
lynx = np.squeeze(dat['lynx'])
lynx = np.array([float(x) for x in lynx])
leng = len(hare)

#%% Plotting Data

plt.figure(figsize=(8,4))
plt.plot(year, hare)
plt.plot(year, lynx)
plt.legend(['Hare', 'Lynx'])
plt.xlabel('Time (Year)')
plt.ylabel('Population')
plt.title('Hare and Lynx Population vs Time')
plt.savefig('pop_fig.png')
plt.show()

# %% Find derivative:

haredot = np.zeros(leng)
lynxdot = np.zeros(leng)
dt = year[1] - year[0]
lynxdot[0] = (-3*lynx[0] + 4*lynx[1] - lynx[2]) / (2*dt)
haredot[0] = (-3*hare[0] + 4*hare[1] - hare[2]) / (2*dt)
for k in (np.arange(leng)[1:-1]):
    haredot[k] = (hare[k+1] - hare[k-1]) / (2*dt)
    lynxdot[k] = (lynx[k+1] - lynx[k-1]) / (2*dt)

lynxdot[-1] = (-3*lynx[-1] + 4*lynx[-2] - lynx[-3]) / (2*dt)
haredot[-1] = (-3*hare[-1] + 4*hare[-2] - hare[-3]) / (2*dt)

# %% Create Library
A = np.transpose(np.array([hare/hare, hare, lynx, hare ** 2, lynx ** 2, hare * lynx, 
     (hare ** 2)* lynx, (lynx ** 2)*hare, lynx ** 3, hare ** 3, np.sin(lynx), 
     np.cos(lynx), np.sin(hare), np.cos(hare), np.log(lynx), np.log(hare)]))

# %%  Solve using Lasso
from sklearn import linear_model as lm
clf_h = lm.Lasso(alpha=1.5)
clf_l = lm.Lasso(alpha=1.5)
clf_h.fit(A, haredot)
clf_l.fit(A, lynxdot)

# %%
xhare = np.squeeze(np.asarray(clf_h.sparse_coef_.todense()))
xlynx = np.squeeze(np.asarray(clf_l.sparse_coef_.todense()))
    
# %% Plot Importance as determined by Lasso
labels = np.arange(16) + 1
text = 'Labels: \n 1 - Constant \n 2 - H \n 3 - L \n 4 - H^2 \n 5 - L^2 \n 6 - H*L \n 7 - LH^2 \n 8 - HL^2 \n 9 - L^3 \n 10 - H^3 \n 11 - sin(L) \n 12 - cos(L) \n 13 - sin(H) \n 14 - cos(H) \n 15 - log(L) \n 16 - log(H)'
props = props = dict(boxstyle='round', facecolor='white', alpha=0.8)

plt.figure(figsize=(8,4))
plt.bar(labels, xhare)
plt.xticks(labels)
plt.title('Hare Coefficients')
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.hlines(0, -1, 20)
plt.xlim(0, 17)
plt.text(18, -2.3,text, fontsize=12, horizontalalignment='left', bbox = props)
#plt.tight_layout()
#plt.savefig('hare_coeff.png')
plt.show()

# %%
plt.figure(figsize=(8,4))
plt.bar(labels, xlynx)
plt.xticks(labels)
plt.title('Lynx Coefficients')
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.hlines(0, -1, 20)
plt.xlim(0, 17)
plt.text(18, -.04,text, fontsize=12, horizontalalignment='left', bbox = props)
# plt.tight_layout()
# plt.savefig('lynx_coeff.png')
plt.show()

# %% Recreate Function
a = [np.abs(xlynx) > 1] # Index 2, 13, 14 True
b = [np.abs(xhare) > 1] # Index 13, 15

lynx_test = xlynx[2]*lynx + xlynx[13]*np.cos(hare) + xlynx[14]*np.log(lynx)
hare_test = xhare[13]*np.cos(hare) + xhare[15]*np.log(hare)

# Plot against original
plt.figure(figsize=(8,4))
plt.plot(year, hare_test + hare[0])
plt.plot(year, lynx_test + lynx[0])
plt.plot(year, hare, '--')
plt.plot(year, lynx, '--')
plt.legend(['Hare Predicted', 'Lynx Predicted', 'Hare Actual', 'Lynx Actual' ])
plt.xlabel('Time (Year)')
plt.ylabel('Population')
plt.title('Hare and Lynx Population vs Time')
plt.savefig('pop_fig_recreate.png')
plt.show()

# %%  Repeat with a different library
A2 = np.concatenate((hare/hare, hare, lynx, hare ** 2, lynx ** 2, hare * lynx, 
     (hare ** 2)* lynx, (lynx ** 2)*hare, np.sin(hare), np.cos(hare), 
     np.sin(lynx), np.cos(lynx)), axis = 1)

clf_h2 = lm.Lasso(alpha=1.5)
clf_l2 = lm.Lasso(alpha=1.5)
clf_h2.fit(A2, haredot)
clf_l2.fit(A2, lynxdot)

xhare2 = np.squeeze(np.asarray(clf_h2.sparse_coef_.todense()))
xlynx2 = np.squeeze(np.asarray(clf_l2.sparse_coef_.todense()))

# %%
labels2 = np.arange(12) + 1
text = 'Labels: \n 1 - Constant \n 2 - H \n 3 - L \n 4 - H^2 \n 5 - L^2 \n 6 - H*L \n 7 - H^2L \n 8 - L^2H \n 9 - sin(H) \n 10 - cos(H) \n 11 - sin(L) \n 12 - cos(L)'
props = props = dict(boxstyle='round', facecolor='white', alpha=0.8)

plt.figure(figsize=(8,4))
plt.bar(labels2, xhare2)
plt.xticks(labels2)
plt.title('Hare Coefficients')
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.hlines(0, -1, 20)
plt.xlim(0, 12)
plt.text(12.5, -2.8,text, fontsize=12, horizontalalignment='left', bbox = props)
plt.tight_layout()
plt.savefig('hare_coeff2.png')
plt.show()
# %%
plt.figure(figsize=(8,4))
plt.bar(labels2, xlynx2)
plt.xticks(labels2)
plt.title('Lynx Coefficients')
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.hlines(0, -1, 20)
plt.xlim(0, 12)
plt.text(12.5, -2,text, fontsize=12, horizontalalignment='left', bbox = props)
plt.tight_layout()
plt.savefig('lynx_coeff2.png')
plt.show()

# %% Recreate Function 2
a2 = [np.abs(xlynx2) > 1] # Index 8, 9 True
b2 = [np.abs(xhare2) > 1] # Index 9

lynx_test2 = xlynx2[8]*np.sin(hare) + xlynx2[9]*np.cos(hare)
hare_test2 = xhare2[8]*np.sin(hare)

plt.figure(figsize=(8,4))
plt.plot(year, hare_test + hare[0])
plt.plot(year, lynx_test + lynx[0])
plt.plot(year, hare)
plt.plot(year, lynx)
plt.legend(['Hare Predicted', 'Lynx Predicted', 'Hare Actual', 'Lynx Actual' ])
plt.xlabel('Time (Year)')
plt.ylabel('Population')
plt.title('Hare and Lynx Population vs Time')
plt.savefig('pop_fig_recreate2.png')
plt.show()


# %% Repeat with a different library
A3 = np.concatenate((np.log(hare), np.log(lynx), hare*np.log(hare), 
                     hare*np.log(lynx), lynx*np.log(hare), lynx*np.log(lynx)), axis = 1)

clf_h3 = lm.Lasso(alpha=2)
clf_l3 = lm.Lasso(alpha=2)
clf_h3.fit(A3, haredot)
clf_l3.fit(A3, lynxdot)

xhare3 = np.squeeze(np.asarray(clf_h3.sparse_coef_.todense()))
xlynx3 = np.squeeze(np.asarray(clf_l3.sparse_coef_.todense()))

# %%
labels3 = np.arange(len(xhare3)) + 1
text = 'Labels: \n 1 - log(H) \n 2 - log(L) \n 3 - H*log(H) \n 4 - H*log(L) \n 5 - L*log(H) \n 6 - L*log(L)'
props = props = dict(boxstyle='round', facecolor='white', alpha=0.8)

plt.figure(figsize=(8,4))
plt.bar(labels3, xhare3)
plt.xticks(labels2)
plt.title('Hare Coefficients')
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.hlines(0, -1, 20)
plt.xlim(0, 12)
plt.text(12.5, -0.01,text, fontsize=12, horizontalalignment='left', bbox = props)
plt.savefig('hare_coeff3.png')
plt.show()
# %%
plt.figure(figsize=(8,4))
plt.bar(labels2, xlynx3)
plt.xticks(labels2)
plt.title('Lynx Coefficients')
plt.xlabel('Library Terms')
plt.ylabel('Coefficient')
plt.hlines(0, -1, 20)
plt.xlim(0, 12)
plt.text(12.5, -2,text, fontsize=12, horizontalalignment='left', bbox = props)
plt.tight_layout()
plt.savefig('lynx_coeff3.png')
plt.show()

#%%
# Interpolation
hare_new = []
lynx_new = []
year_new = []
for n in np.arange(leng)[0:-1]:
    year_new.append(year[n])
    interph = (year[n] + year[n+1]) / 2
    year_new.append(interph)
    
    lynx_new.append(lynx[n])
    interph = (lynx[n] + lynx[n+1]) / 2
    lynx_new.append(interph)
    
    hare_new.append(hare[n])
    interph = (hare[n] + hare[n+1]) / 2
    hare_new.append(interph)
    
hare = np.array(hare_new)
lynx = np.array(lynx_new)
year = np.array(year_new)
leng = len(hare)





