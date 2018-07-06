import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prody import *
import sys
import time
sys.path.insert(0, '../src/')

def shift_structure(oxgs,dx=0,dy=0,dz=0):
    '''
    shifts the structure by by dx, dy, dz
    '''
    # shift by dxyz
    xyz = oxgs.getCoords()
    dxyz = np.zeros(xyz.shape)
    dxyz[:,0] = dx
    dxyz[:,1] = dy
    dxyz[:,2] = dz

    # shifted copy
    oxgs_shifted = oxgs.copy()
    oxgs_shifted.setCoords(xyz+dxyz)

    return oxgs_shifted

def expand_ice(oxgs,n=1,dx=0,dy=0,dz=0):
    '''
    expands ice by along (dx,dy,dz) n times
    '''
    expanded = oxgs.copy()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                expanded += shift_structure(oxgs,dx=i*dx,dy=j*dy,dz=k*dz)

    print 'Crystal expanded: %dA x %dA x %dA '%(dx*n,dy*n,dz*n)
    return expanded

def iceIh_parameters():
    # taken from http://www1.lsbu.ac.uk/water/hexagonal_ice.html
    file_name = 'ice1h'
    density_Ih = 0.917
    density = density_Ih*10**(-24)/((2*1.0079+15.999)/(6.022*10**23))

    dOOa = 4.5181
    dOOb = np.sqrt(3.)*dOOa
    dOOc = 7.3560

    dx = dOOb*3
    dy = dOOc*3
    dz = 6*dOOa

    return file_name, density, dx, dy, dz

file_name, density, dx, dy, dz = iceIh_parameters()
n_expand = 5

save_path = '../analysis/'
data_path = '../data/'
data_file = '%s%s.pdb'%(data_path,file_name)

wtr1 = parsePDB(data_file)
oxgs1 = wtr1.select('name O1')
oxgs2 = wtr1.select('name O2')
#oxgs3 = wtr1.select('name O3')
#oxgs4 = wtr1.select('name O4')
#oxgs5 = wtr1.select('name O5')
#oxgs6 = wtr1.select('name O6')
#oxgs =  oxgs1.copy()+oxgs2.copy()+oxgs3.copy()+oxgs4.copy()+oxgs5.copy()+oxgs6.copy()
oxgs =  oxgs1.copy()
xyz = oxgs.getCoords()


fig = plt.figure(figsize=[6,5])
fig3d = Axes3D(fig)
fig3d.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='red')
#x_range = (Na + 1)*np.ceil(aa.max())/2.
#y_range = (Nb + 1)*np.ceil(aa.max())/2.
#z_range = (Nc + 1)*np.ceil(ac.max())/2.
fig3d.set_xlim3d(-15, 15)
fig3d.set_ylim3d(-15, 15)
fig3d.set_zlim3d(-15, 15)
fig3d.set_xlabel(r'x $[\AA]$')
fig3d.set_ylabel(r'y $[\AA]$')
fig3d.set_zlabel(r'z $[\AA]$')
plt.show()
plt.close(fig)

# molecular form factor
def affapprox(Q,a1,b1,a2,b2,a3,b3,a4,b4,c):
    return a1*np.exp(-b1*Q**2) + a2*np.exp(-b2*Q**2) + a3*np.exp(-b3*Q**2) + a4*np.exp(-b4*Q**2) + c

def aff1s(Q,z): #1s approximation
    if z==1:
        a = 0.53
    else:
        a = 0.53/(z-0.3)
    return 1 / (1+(Q*a/2)**2)**2

Q = np.arange(0,2,0.01)
ffo = affapprox(Q,3.0485,13.2771,2.2868,5.7011,1.5463,0.3239,0.8670,32.9089,0.2508)
ffh = aff1s(Q,1)
ql1 = Q*0.9584
ql2 = Q*1.5151
ffmol=ffo**2 + 2*ffh**2 + 4*ffo*ffh*np.sin(ql1)/ql1 + 2*ffh*ffh*np.sin(ql2)/ql2


def ffmol2(q):
    Q=q
    a1,b1,a2,b2,a3,b3,a4,b4,c=3.0485,13.2771,2.2868,5.7011,1.5463,0.3239,0.8670,32.9089,0.2508
    ffo=a1*np.exp(-b1*Q**2) + a2*np.exp(-b2*Q**2) + a3*np.exp(-b3*Q**2) + a4*np.exp(-b4*Q**2) + c
    a = 0.53
    ffh=1 / (1+(Q*a/2)**2)**2
    ql1 = Q*0.9584
    ql2 = Q*1.5151
    return ffo**2 + 2*ffh**2 + 4*ffo*ffh*np.sin(ql1)/ql1 + 2*ffh*ffh*np.sin(ql2)/ql2

def sn(q,rn):#q = 3dim vector; rn = n*3 matrix
    out = 0*complex(0,1) #initialize
    dp = np.dot(rn,q) #q*r dot-product
    out = np.sum(np.exp(dp*complex(0,1))) #lattice sum
    out /= len(rn[:,0]) #normalize by the number of unit cells
    return (out*np.conjugate(out)) #/ len(rn[:,0])**2


# Q array
n_pix = 100 #number of pixels along each dimension
dd = 100 #50 # sample-detector distance in units of pixel size
q = np.zeros([n_pix**2,3])
#wavelength = 2.4797 # photon energy 5 kev in units of Angstrom
wavelength = 1.2398 # photon energy 10 kev in units of Angstrom

k_in = np.array([0,1,0])

idx = 0
k_y = dd 
for dx in range(n_pix):
    k_x = dx - n_pix/2
    for dz in range(n_pix):
        k_z = dz - n_pix/2
        k_out = k_x,k_y,k_z
        kabs = np.sqrt(k_x**2+k_y**2+k_z**2)
        k_out/=kabs
        
        for i in range(3):
            q[idx,i] =  k_in[i]-k_out[i]
        idx += 1
        
Q_norm = np.sqrt(q[:,0]**2+q[:,1]**2+q[:,2]**2)*(2.*np.pi/wavelength)
Q2d = Q_norm.reshape((n_pix,n_pix))
Q_x = Q2d[:,n_pix/2]
Q_y = Q2d[n_pix/2,:]

f_atom_2d = ffmol2(Q_norm)
f_atom_2d = f_atom_2d.reshape((n_pix,n_pix))

fig=plt.figure(figsize=[8,3])
plt.subplot(1,2,1)
plt.imshow(Q2d, interpolation='nearest',cmap='jet',origin='lower right')
plt.xlabel(r'$Q_x [\AA^{-1}]$',size=15)
plt.ylabel(r'$Q_y [\AA^{-1}]$',size=15)
plt.title('Q map')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(f_atom_2d,interpolation='nearest',cmap='jet',origin='lower right',extent=[-Q_x[0],Q_x[0],-Q_y[0],Q_y[0]])
plt.xlabel(r'$Q_x [\AA^{-1}]$',size=15)
plt.ylabel(r'$Q_y [\AA^{-1}]$',size=15)
plt.title('Form Factor')
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close(fig)

# lattice sum
lmbda = np.array([wavelength]) #Angstrom
intensity = np.zeros([n_pix**2,len(lmbda)])
q_2d = np.zeros([n_pix**2])

idx = 0
for l in lmbda:
    ql = (2*np.pi/l)*q
    for q_pix in range(len(q[:,0])):
        intensity[q_pix,idx] = sn(ql[q_pix],xyz)
    idx += 1

intensity = np.max(intensity,axis=1)
intensity_2d = intensity.reshape((n_pix,n_pix))

#plt.figure(figsize=[6,4])
fig=plt.figure(figsize=[8,3])
plt.subplot(1,2,1)
# lineaer scale
plt.imshow(np.rot90(intensity_2d),interpolation='nearest',cmap='jet',origin='lower right',extent=[-Q_x[0],Q_x[0],-Q_y[0],Q_y[0]])
# log scale
#plt.imshow(np.log(np.rot90(intensity_2d)),interpolation='nearest',cmap='jet',origin='lower right',extent=[-Q_x[0],Q_x[0],-Q_y[0],Q_y[0]])
plt.colorbar()
#plt.title('Lattice sum (log scale)')
plt.title('Lattice sum')
plt.xlabel(r'$Q_x [\AA^{-1}]$',size=15)
plt.ylabel(r'$Q_y [\AA^{-1}]$',size=15)

plt.subplot(1,2,2)
# linear scale
plt.imshow(np.rot90(intensity_2d*f_atom_2d),vmax=1,interpolation='nearest',cmap='jet',origin='lower right',extent=[-Q_x[0],Q_x[0],-Q_y[0],Q_y[0]])
# log scale
#plt.imshow(np.log(np.rot90(intensity_2d*f_atom_2d)),vmax=1,interpolation='nearest',cmap='jet',origin='lower right',extent=[-Q_x[0],Q_x[0],-Q_y[0],Q_y[0]])
#plt.title('Total scattering (log scale)')
plt.title('Total scattering')
plt.colorbar()
plt.xlabel(r'$Q_x [\AA^{-1}]$',size=15)
plt.ylabel(r'$Q_y [\AA^{-1}]$',size=15)
plt.tight_layout()
plt.show()
plt.close(fig)
