import numpy as np
import matplotlib.pyplot as plt
import math

# Equation of circle

# line1 attributes ==> A1*X = c1
A1 = np.array([3.0,-4.0])
c1 = np.array([7.0])

# line2 attributes ==> A2*X = c2
A2 = np.array([2.0,-3.0])
c2 = np.array([5.0])

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

#center		 
c = center()

#given area in question
area = 49*math.pi

#area = pi*(r^2)
#radius
r = np.sqrt(area/math.pi)

# (x.T)*x - 2c.T*x + c.T*c - r^2 = 0
# General equation of circle
# c - center of circle
# r - radius of circle

def circle():
	print('x.T*x - 2c.T*x + c.T*c - r^2 = 0')
	print('where')
	print('c - ',c)
	print('r - ',r)
	

def line(slope, intercept):
	y = (slope*x) + intercept
	return y
   
x = np.linspace(-10,10,500)

#slopes of given lines
slope1 = -A1[0]/A1[1]
slope2 = -A2[0]/A2[1]

#intercepts of given lines
intercept1 = c1[0]/A1[1]
intercept2 = c2[0]/A2[1]

#line eqations
y1 = line(slope1,intercept1)
y2 = line(slope2,intercept2)

a = [1,1]
b = [-1,6]

#circle equation
circle()

#plot of given lines and circle center
plt.plot(x,y1,'b',x,y2,'k',c[0],c[1],'om')

  
#plot of circle
k=plt.Circle((c[0],c[1]), r , color='r')
plt.gcf().gca().add_artist(k)

plt.text(1,-2,'c (1,-1)')
plt.plot(a,b,'--k')
plt.text(-0.5, 2.5*0.5, 'r = 7')
plt.xlim(-12,12)
plt.ylim(-12,12)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('equal')
plt.show()
plt.savefig('plot.jpg')	
	
	
