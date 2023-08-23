import numpy as np
from brian2 import *

start_scope()

defaultclock.dt = 0.01*ms

gs = 1*mS
Is = 5*mA
#Es = -10*mV
tau_rise = 1*ms
tau_decay = 10*ms
t0 = 1*ms
h0 = siemens/second**2
t_peak =t0+ tau_rise*tau_decay/(tau_decay-tau_rise)*log(tau_decay/tau_rise)
f = 1/(-exp(-(t_peak-t0)/tau_rise)+exp(-(t_peak-t0)/tau_decay))

Cm = 1*uF # /cm**2
#Iapp = 2*uA
gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 9*msiemens

eqs = '''
dv/dt = (-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL)+Iapp)/Cm : volt
m = alpha_m/(alpha_m+beta_m) : 1
alpha_m = 0.1/mV*10*mV/exprel(-(v+35*mV)/(10*mV))/ms : Hz
beta_m = 4*exp(-(v+60*mV)/(18*mV))/ms : Hz
dh/dt = 5*(alpha_h*(1-h)-beta_h*h) : 1
alpha_h = 0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
beta_h = 1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
dn/dt = 5*(alpha_n*(1-n)-beta_n*n) : 1
alpha_n = 0.01/mV*10*mV/exprel(-(v+34*mV)/(10*mV))/ms : Hz
beta_n = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
tau:second
Iapp:amp
'''

G = NeuronGroup(2, eqs, threshold='v>-50*mV', reset='v = -70*mV',  method='exponential_euler')
G.Iapp = [2*uA, 0*uA]
G.tau = [10, 100]*ms
G.v = -70*mV
G.h = 1
#threshold='v>-50*mV', reset='v = -70*mV',
# Comment these lines out to see what happens without Synapses
S = Synapses(G, G, model = '''
w:volt
dg/dt = -g/tau_decay +h_syn: siemens
dh_syn/dt = -h_syn/tau_rise+h0 : siemens/second
''', on_pre='''
v_post +=w
w=Is/g
''', method ='exponential_euler')
#S.w = 
S.connect(i=0, j=1)
#dh_syn/dt = -h_syn/tau_rise+h0 : siemens/second
#+h_syn
#dg/dt = -g/tau_decay+h:1
#dh/dt = -h/tau_rise+h0:1
#g=gs*f*(exp(-(t-t0)/tau_dec)-exp(-(t-t0)/tau_rise)):siemens
M = StateMonitor(G, 'v', record=True)

run(100*ms)

plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend();
show()
