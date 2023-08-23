import numpy as np
import cmath
from brian2 import *

defaultclock.dt = 0.01*ms

Cm = 1*uF # /cm**2
gL = 50*nsiemens
EL = -65*mV
EE = 0*mV
alpha1 = 280*Hz
alpha2 = 280*Hz
sigma1 = 21.2*cm
sigma2 = 21.2*cm
fi1 = 260/360*2*pi
fi2 = 100/360*2*pi
x1 = 90*cm
x2 = 110*cm
vel = 40*cm/second
k = 0*1/cm
f = 8*Hz
eqs = '''
x = vel*t :metre
dv/dt = (gL*(EL-v)+gE*(EE-v))/Cm : volt
A1 = alpha1*exp(-((x-x1)**2)/(2*sigma1**2)):Hz
A2 = alpha2*exp(-((x-x2)**2)/(2*sigma2**2)):Hz
r3 = Atot*cos(2*pi*f*t+fi)+A1+A2 :Hz
fi = arctan((A1*sin(fi1)+A2*sin(fi2))/(A1*cos(fi1)+A2*cos(fi2))):1
Atot = sqrt(A1**2+A2**2+2*A1*A2*cos(fi1-fi2)):Hz
dgE/dt = -gE/(2*ms)+r3/660*mS :siemens

'''
#r1 = A1*(cos(2*pi*f*t+fi1)+1) :Hz
#r2 = A2*(cos(2*pi*f*t+fi2)+1) :Hz

neuron = NeuronGroup(1, eqs, threshold ='v>-52*mV' , reset = 'v = -65*mV', method='exponential_euler')
neuron.v = -70*mV
neuron.gE = 0.1*mS
M = StateMonitor(neuron, 'v', record=0)

run(5000*ms, report='text')

plot(M.t/ms, M[0].v/mV)
show()

