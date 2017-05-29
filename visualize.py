import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def plot_training_data(fun, dim, nplot, heat=True):
	n2 = np.ceil(np.sqrt(nplot))
	n1 = np.ceil(nplot/n2)
	
	if dim==1: 
		x = np.linspace(-1,1,100).reshape(1,-1, dim)
		y = fun(x)
		plt.figure(figsize=(5*n1,3*n2))
		for i in range(nplot):
			plt.subplot(n1,n2,i+1)
			plt.plot(x.flatten(),y[i])
		plt.show()
	if dim==2:
		xx1 = np.linspace(-1,1,100)
		xx2 = np.linspace(-1,1,100)
		x = np.array(np.meshgrid(xx1,xx2)).T.reshape(1,-1,dim)
		y = fun(x)
		if not heat: 
			XX1, XX2 = np.meshgrid(xx1,xx2)			
			fig = plt.figure(figsize=(20,15))
			for i in range(nplot):
				ax1 = fig.add_subplot(n1,n2,1+i, projection='3d')
				ax1.plot_surface(XX1, XX2, y[i].reshape(100,100), cmap=cm.coolwarm, 
									   linewidth=0, antialiased=False)
			plt.show()
		if heat:
			plt.figure(figsize=(20,15))
			for i in range(nplot):
				plt.subplot(n1,n2,1+i)
				plt.imshow(y[i].reshape(100,100), cmap='hot', interpolation='nearest', extent=[-1,1,1,-1])
				plt.colorbar()
				plt.xlim([-1,1])
				plt.ylim([-1,1])
			plt.show()
			
def plot_result(fun, dim, nplot, samples):
	n2 = np.ceil(np.sqrt(nplot))
	n1 = np.ceil(nplot/n2)
	n_steps = samples.shape[1]
	
	if dim == 1:  
		xx = np.linspace(-1,1,200).reshape(1,-1, dim)
		yy = fun(xx)
		samples_y = fun(samples)
		
		plt.figure(figsize=(20,15))		
		for i in range(nplot):        
			plt.subplot(n1,n2,i+1)
			plt.plot(xx.flatten(),yy[i],linewidth=3)
			plt.scatter(samples[i],samples_y[i],color="k")
			for j in range(n_steps):
				plt.text(samples[i,j], samples_y[i,j], str(j), color="red", fontsize=12)			
		plt.show()	
	if dim == 2:
		xx1 = np.linspace(-1,1,100)
		xx2 = np.linspace(-1,1,100)
		xx = np.array(np.meshgrid(xx1,xx2)).T.reshape(1,-1,dim)
		yy = fun(xx)

		plt.figure(figsize=(20,15))
		for i in range(nplot):
			plt.subplot(n1,n2,i+1)
			plt.imshow(yy[i].reshape(100,100), cmap='hot', interpolation='nearest', extent=[-1,1,1,-1])
			#plt.colorbar()
			plt.scatter(samples[i,:,1], samples[i,:,0], color = "b", marker = "x")
			for j in range(n_steps):
				plt.text(samples[i,j,1], samples[i,j,0], str(j), color="green", fontsize=12)
			plt.xlim([-1,1])
			plt.ylim([-1,1])
		plt.show()

def visualize_learning(train_loss_list, test_loss_list, train_fmin_list, test_fmin_list):		
	plt.figure(figsize=(20,8))
	plt.subplot(2,2,1)
	plt.plot(train_loss_list)
	plt.title("Training Error")
	plt.subplot(2,2,2)
	plt.plot(test_loss_list)
	plt.title("Validation Error")
	plt.subplot(2,2,3)
	plt.plot(train_fmin_list)
	plt.title("Training Minimum")
	plt.subplot(2,2,4)
	plt.plot(test_fmin_list)
	plt.title("Validation Minimum")
	plt.show()