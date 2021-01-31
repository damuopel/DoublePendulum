import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import gif

# Pendulum Lenghts and masses
L1, L2 = 1, 1
m1, m2 = 1, 1
# Gravity
g = 9.81

def deriv(y, t):
    theta1, z1, theta2, z2 = y
    
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
    
    z1dot = (m2*g*np.sin(theta2)*c-m2*s*(L1*z1**2*c+L2*z2**2)-(m1+m2)*g*np.sin(theta1))/(L1*(m1+m2*s**2))
    z2dot = ((m1+m2)*(L1*z1**2*s-g*np.sin(theta2)+g*np.sin(theta2)*c)+m2*L2*z2**2*s*c)/(L2*(m1+m2*s**2))
    
    return z1, z1dot, z2, z2dot

# Time spacing
tmax, dt = 30, 0.01
t = np.arange(0,tmax+dt,dt)

# Initial Conditions
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])
# y0 = np.array([np.pi/2, 0, np.pi/2, 0])

# Numerical Integration
y = odeint(deriv, y0, t)

theta1, theta2 = y[:,0], y[:,2]

x1 = L1*np.sin(theta1)
y1 = -L1*np.cos(theta1)
x2 = x1 + L2*np.sin(theta2)
y2 = y1 - L2*np.cos(theta2)

@gif.frame
def plot(i):
    r = 0.05
    plt.figure()
    ax = plt.gca()
    plt.plot([0,x1[i],x2[i]],[0,y1[i],y2[i]],lw=2,c='k')
    c1 = plt.Circle((0,0),r/2)
    c2 = plt.Circle((x1[i],y1[i]),r,color='r')
    c3 = plt.Circle((x2[i],y2[i]),r,color='b')
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)
    
    ax.set_xlim(-L1-L2-r,L1+L2+r)
    ax.set_ylim(-L1-L2-r,L1+L2+r)
    ax.set_aspect('equal')
    
    plt.axis('off')

fps = 10
di = int(1/fps/dt)
frames = []
for i in np.arange(0,t.size,di):
    frame = plot(i)
    frames.append(frame)
      
gif.save(frames, 'Test.gif', duration=1000/fps)