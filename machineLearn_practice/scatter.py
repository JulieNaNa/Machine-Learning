import array
import matplotlib.pyplot as plt
import numpy as np

def scatter(x,y,l):
	f=[40]*len(l)
	c=np.array(f)*np.array(l)
	#print type(c)
	
	#s=15*array.array('B',l)
	#print type(s)
	fig = plt.figure()
	subplot=fig.add_subplot(111)
	subplot.scatter(x,y,c,c)
	#subplot.scatter(x,y)
	plt.show()
