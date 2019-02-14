import numpy as np
import matplotlib.pyplot as plt

# line1 attributes ==> A1*X = c1
A1 = np.array([2,1])
c1 = np.array([3])

# line2 attributes ==> A2*X = c2
A2 = np.array([1,-1])
c2 = np.array([1])

# point of tangency
P = np.array([[1],[-1]])

#plot of required tangent on the circle

# center of the circle is the intersection of the lines
def center():
	A = np.vstack((A1,A2))
	D = np.vstack((c1,c2))
 
 # parallel lines don't intersect
	if(np.linalg.det(A) == 0):
		raise Exception('The given lines are parallel and hence dont intersect')
 # AX = D ==> X = inv(A)*D 
	else:
		X = np.matmul(np.linalg.inv(A),D)
		return X 

# circle attributes
C = center()     #centre
CP = C-P         #radius vector
radius = np.linalg.norm(CP)   #radius magnitude

def eqn_tan(x):
	#slope of tangent = -(1/slope of line CP)
	slope_tan = -(CP[0])/(CP[1])
	return (slope_tan*(x - P[0])) + P[1] 

x = np.linspace(-1,8,500)
#tangent equation
y = eqn_tan(x)

#plot of tangent at point of tangency
plt.plot(x,y,'--b',P[0],P[1],'om')

#plot of circle
k=plt.Circle((C[0],C[1]), radius , color='r')
plt.gcf().gca().add_artist(k)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('equal')
plt.show()
plt.savefig('plot.jpg')
