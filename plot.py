from equation import quat_eqns
from equation import ter_eqns

import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import glob


########colorcode for Okabe colorblind palette

col1= "#E69F00"
col2= "#009E73"
col3= "#0072B2"
col4=  "#CC79A7"

###PARAMETERS

const_p = {
	
"gamma_i" : 8.84*60,  #RNA production
"gamma_j" : 8.84*60, #RNA production
"dg_i" : 0.18*60, #RNA degrdation
"dg_j" : 0.18*60, #RNA degrdation
"m_t" : 100, #total MCP_soxS
"d_t" : 100, #total dCAS9
"Di_t" : 30, #total activation site
"Dj_t" : 30, #total inihibition site
"ui" : 0.0,# 0.000023  #leaky expression RNA
"uj" : 0.0,# 0.000023  #leaky expression RNA
"K_i" : 100,#4000, #GFP activation factor
"K_j" : 5,#800, #GFP basal expression factor
"dG" : 0.0196*60  #GFP degrdation
}

dynamic_p = {
	"k_j" : 1, #inhibition complex fromation
	"q_j" : 1, #inhibition DNA binding
	"k_i" : 1, #activation complex fromation
	"q_i" : 1 #activation DNA binding
}

p = {**const_p, **dynamic_p}
dt = 0.01


ARA=[0,10]
AHL=[0,10]

#One plot
out_arr=np.zeros((2,len(ARA),len(AHL)))


p['k_i']=1
p['k_j']=100000
p['q_i']=1
p['q_j']=100

p['d_t']=10000

pqj = p['q_j'] #save value

for j,ja in enumerate(AHL):
	for i,ia in enumerate(ARA):
		p['q_j'] =pqj #restore value
		i_arr=np.zeros(5) #ARA,x,g,c,C,D
		j_arr=np.zeros(5) #AHL,x,g,c,C,D 
		sys_arr=np.zeros(3) #dcas9,soxs, GFP
		sys_arr[0]=p['d_t']
		sys_arr[0]=p['m_t']
		i_arr[0]=ia
		j_arr[0]=ja
		for t in range(10000):
		    #i_arr, j_arr, sys_arr = quat_eqns(i_arr, j_arr, sys_arr, p, dt)
		    i_arr, j_arr, sys_arr = ter_eqns(i_arr, j_arr, sys_arr, p, dt)
		out_arr[0,j,i]=sys_arr[2]


		i_arr=np.zeros(5) #ARA,x,g,c,C,D
		j_arr=np.zeros(5) #AHL,x,g,c,C,D 
		sys_arr=np.zeros(3) #dcas9,soxs, GFP
		sys_arr[0]=p['d_t']
		sys_arr[0]=p['m_t']
		i_arr[0]=ia
		j_arr[0]=ja
		p['q_j']=0 #off target
		for t in range(10000):
		    #i_arr, j_arr, sys_arr = quat_eqns(i_arr, j_arr, sys_arr, p, dt)
		    i_arr, j_arr, sys_arr = ter_eqns(i_arr, j_arr, sys_arr, p, dt)
		out_arr[1,j,i]=sys_arr[2]

h1 = np.concatenate(out_arr[0])
h2 = np.concatenate(out_arr[1])
h=np.append(h1,0)
h=np.append(h,h2)
plt.bar(np.arange(len(h)), h, color=[col1,col2,col3,col4,col1])
plt.title("kij = " + str(p['k_i']) + "/" + str(p['k_j'])+ " qij = " +str(p['q_i']) +"/" + str(pqj), size=4)
plt.ylim(0,3000)
for t in h:
	print(t)
plt.show()
stop
#####plot ratio kij, qij
l=12 #4
ratio1=np.logspace(1,6,l)
ratio2=np.logspace(1,6,l)
r=np.ones(len(ratio1)-1)
ra1=np.append(r,ratio1)
ra2=np.append(np.flip(ratio1),r)

#store GFP values in one array
tt=len(ra1)*len(ra2)
out_arr=np.zeros((len(ra1),len(ra2),2,len(ARA),len(AHL)))

fig, axs = plt.subplots(len(ra1),len(ra2),constrained_layout = True, figsize=(14, 8))
c=0
for ri,r1 in enumerate(ra1):
	for rj,r2 in enumerate(ra1):

		p['k_i']=r1
		p['k_j']=ra2[ri]

		p['q_i']=r2
		p['q_j']=ra2[rj]
		pqj = p['q_j'] #save value



		for j,ja in enumerate(AHL):
			for i,ia in enumerate(ARA):
				p['q_j'] =pqj #restore value
				i_arr=np.zeros(5) #ARA,x,g,c,C,D
				j_arr=np.zeros(5) #AHL,x,g,c,C,D 
				sys_arr=np.zeros(3) #dcas9,soxs, GFP
				sys_arr[0]=p['d_t']
				sys_arr[0]=p['m_t']
				i_arr[0]=ia
				j_arr[0]=ja
				for t in range(10000):
				    #i_arr, j_arr, sys_arr = quat_eqns(i_arr, j_arr, sys_arr, p, dt)
				    i_arr, j_arr, sys_arr = ter_eqns(i_arr, j_arr, sys_arr, p, dt)
				out_arr[ri,rj,0,j,i]=sys_arr[2]


				i_arr=np.zeros(5) #ARA,x,g,c,C,D
				j_arr=np.zeros(5) #AHL,x,g,c,C,D 
				sys_arr=np.zeros(3) #dcas9,soxs, GFP
				sys_arr[0]=p['d_t']
				sys_arr[0]=p['m_t']
				i_arr[0]=ia
				j_arr[0]=ja
				p['q_j']=0 #off target
				for t in range(10000):
				    #i_arr, j_arr, sys_arr = quat_eqns(i_arr, j_arr, sys_arr, p, dt)
				    i_arr, j_arr, sys_arr = ter_eqns(i_arr, j_arr, sys_arr, p, dt)
				out_arr[ri,rj,1,j,i]=sys_arr[2]

		h1 = np.concatenate(out_arr[ri,rj,0])
		h2 = np.concatenate(out_arr[ri,rj,1])
		h=np.append(h1,0)
		h=np.append(h,h2)
		axs[ri,rj].bar(np.arange(len(h)), h, color=[col1,col2,col3,col4,col1])
		axs[ri,rj].set_title("kij = " + str(p['k_i']) + "/" + str(p['k_j'])+ " qij = " +str(p['q_i']) +"/" + str(pqj), size=4)
		axs[ri,rj].set_ylim(0,3000)
		c=c+1


#plt.savefig("kij_qij_ter.pdf")
#plt.savefig("kij_qij_quat.pdf")

plt.show()

fig, axs = plt.subplots(2,4,constrained_layout = True)#), figsize=(14, 8))

axs[0,0].imshow(out_arr[:,:,0,0,0],vmin=0,vmax=2500, cmap="Oranges")
axs[0,1].imshow(out_arr[:,:,0,0,1],vmin=0,vmax=2500, cmap="Greens")
axs[0,2].imshow(out_arr[:,:,0,1,0],vmin=0,vmax=2500, cmap="Blues")
axs[0,3].imshow(out_arr[:,:,0,1,1],vmin=0,vmax=2500, cmap="Purples")

axs[1,0].imshow(out_arr[:,:,1,0,0],vmin=0,vmax=2500, cmap="Oranges")
axs[1,1].imshow(out_arr[:,:,1,0,1],vmin=0,vmax=2500, cmap="Greens")
axs[1,2].imshow(out_arr[:,:,1,1,0],vmin=0,vmax=2500, cmap="Blues")
axs[1,3].imshow(out_arr[:,:,1,1,1],vmin=0,vmax=2500, cmap="Purples")
plt.savefig("heatmap_ter_qij_kij2.pdf")
plt.show()



"""
##### for cas9 and soxS

p['k_i']=1
p['k_j']=100000

p['q_i']=1
p['q_j']=100
pqj = p['q_j'] #save value

qty=np.logspace(0,4,5)

#store GFP values in one array
out_arr=np.zeros((2,len(ARA),len(AHL)))

fig, axs = plt.subplots(len(qty),len(qty),constrained_layout = True, figsize=(14, 8))

for ri,r1 in enumerate(qty):
	for rj,r2 in enumerate(qty):
		p["m_t"]= r1
		p["d_t"]= r2

		for j,ja in enumerate(AHL):
			for i,ia in enumerate(ARA):
				p['q_j'] =pqj #restore value
				i_arr=np.zeros(5) #ARA,x,g,c,C,D
				j_arr=np.zeros(5) #AHL,x,g,c,C,D 
				sys_arr=np.zeros(3) #dcas9,soxs, GFP
				sys_arr[0]=p['d_t']
				sys_arr[0]=p['m_t']
				i_arr[0]=ia
				j_arr[0]=ja
				for t in range(10000):
				    i_arr, j_arr, sys_arr = quat_eqns(i_arr, j_arr, sys_arr, p, dt)
				    #i_arr, j_arr, sys_arr = ter_eqns(i_arr, j_arr, sys_arr, p, dt)
				out_arr[0,j,i]=sys_arr[2]


				i_arr=np.zeros(5) #ARA,x,g,c,C,D
				j_arr=np.zeros(5) #AHL,x,g,c,C,D 
				sys_arr=np.zeros(3) #dcas9,soxs, GFP
				sys_arr[0]=p['d_t']
				sys_arr[0]=p['m_t']
				i_arr[0]=ia
				j_arr[0]=ja
				p['q_j']=0 #off target
				for t in range(10000):
				    i_arr, j_arr, sys_arr = quat_eqns(i_arr, j_arr, sys_arr, p, dt)
				    #i_arr, j_arr, sys_arr = ter_eqns(i_arr, j_arr, sys_arr, p, dt)
				out_arr[1,j,i]=sys_arr[2]

		h1 = np.concatenate(out_arr[0])
		h2 = np.concatenate(out_arr[1])
		h=np.append(h1,0)
		h=np.append(h,h2)
		axs[ri,rj].bar(np.arange(len(h)), h, color=[col1,col2,col3,col4,col1])
		axs[ri,rj].set_title("dCas9 = " + str(r2) + "MCP-SoxS = " + str(r1), size=6)
		axs[ri,rj].set_ylim(0,3000)


plt.savefig("cas9_soxS_ter2.pdf")
#plt.savefig("cas9_soxS_quat.pdf")

plt.show()
"""