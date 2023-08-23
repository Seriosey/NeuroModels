from brian2 import *
import numpy as np

start_scope()


tauP = 30*ms
tauI = 3*ms
Rm = 33*Mohm
Aexc = 2*nA
Aahp = -2*nA
tau_ahp = 17*ms
Agaba = -20*nA
tau_gaba = 3*ms
Acur = 1*nA
tau_cur = 5*ms
Ibasal = 0.5*nA
sigma = 32
#sigma = 5
Imax = 5*nA
Vrest = -65*mV
theta = 145

# @check_units(t = 1, result = 1)
# def H(t):
#     if t> 0: return 1
#     else: return 0
#
# Igaba = lambda t: Agaba*H(1-t/tau_gaba)*(1)
# Iahp = lambda t: Aahp*H(1-t/tau_ahp)*(1)

#@check_units(theta_max = 1, theta=1, sigma = 1, result=1)
#def G(theta_max, theta, sigma):
#    return np.exp(-(theta - theta_max**2)/(2*sigma**2))

_Noise = 0 #np.random.normal(loc=0, scale=1, size= 100)

eqsP = '''
# dVp/dt = (-Vp + Vrest +Rm*(Iexc))/tauP : volt 
dVp/dt = (-Vp + Vrest +Rm*(Iexc  + Iahp + Igaba))/tauP : volt #(event-driven)
Iexc = Ibasal + Imax*(exp(-(theta - theta_max)**2/(2*sigma**2)) + _Noise) : ampere (constant over dt)
#Igaba =  Agaba*int(t>0*ms)*(1-t/tau_gaba)*int((1-t/tau_gaba)>0)*(1) : ampere (constant over dt)
Igaba : ampere
Iahp =  Aahp*int(t>0*ms)*(1-t/tau_ahp)*int((1-t/tau_ahp)>0)*(1) : ampere  (constant over dt)
theta_max : 1
# theta : 1
# sisgma : 1
# v:1
'''

#dv/dt = (i-v)/taup : 1 (unless refractory)
#i:1

eqsI = '''
w:1
#dVi/dt = -Vi/tauI : volt (unless refractory)
Vi: volt

# dVp/dt = (-Vp + Vrest +Rm*(Iexc  + Iahp + Igaba))/tauP : volt
# Iexc = Ibasal + Imax*(exp(-(theta - theta_max)**2/(2*sigma**2)) + _Noise) : ampere (constant over dt)
# Igaba =  Agaba*int(t>0*ms)*(1-t/tau_gaba)*int((1-t/tau_gaba)>0)*(1) : ampere (constant over dt)
# Iahp =  Aahp*int(t>0*ms)*(1-t/tau_ahp)*int((1-t/tau_ahp)>0)*(1) : ampere  (constant over dt)
# theta_max : 1
# I:1
'''

N = 100

Pyramids = NeuronGroup(N, eqsP, threshold='Vp > -50*mV', reset='Vp = -65*mV', refractory=5*ms)
Interneurons = NeuronGroup(1, eqsI, threshold='Vi> -40*mV', reset='Vi = -55*mV', refractory=5*ms, method='exact')
#
X = np.random.normal(1, 0.3, 1000)
# count, bins, _ = plt.hist(X, N, density=True)
#print(count)
#Pyramids.I = count
Pyramids.theta_max = [95 + i for i in range(N)]
Pyramids.Vp = Vrest*np.ones(N)
Interneurons.Vi = -55*mV

#model =  '''#w_s : volt
             #dVi/dt = -Vi/tauI : volt (unless refractory, event-driven)'''Vp_post -= 10*mV
S1 = Synapses(Interneurons, Pyramids,
             on_pre= '''
                                  
             Igaba_post -=  0.5*nA''', #on_post = 'Vi_pre +=15*mV',
            delay = 0*ms)

S2 = Synapses(Pyramids,Interneurons, #model = '''dVi/dt = -Vi/tauI : volt (event-driven) ''',
             on_pre= 'Vi_post += 0.5*mV'
                     , #on_post = 'Vp_pre -=20*mV',
             delay = 2*ms)

S1.connect()
S2.connect()
#S.w_s = '10*mV'
S1.delay = '0*ms'
S2.delay = '2*ms'

statemon = StateMonitor(Pyramids, 'Vp', record=range(0, N))
statemon_I = StateMonitor(Pyramids, ['Iexc', 'Igaba', 'Iahp'], record=range(0, N, 20))
spikemon = SpikeMonitor(Pyramids)
statemon_Int = StateMonitor(Interneurons, 'Vi', record=0)



tfinal = 10*ms
run(tfinal, report='text')

plot(statemon_Int.t/ms, statemon_Int.Vi[0]/mV, label= 'Vi')
show()

# print(statemon_I.Iahp.shape)
#plot(statemon_I.t/ms, np.abs(statemon_I.Iahp[:].T), label='Iahp')
plot(statemon_I.t/ms, statemon_I.Igaba[:].T, label='Igaba')
#plot(statemon_I.t/ms, np.abs(statemon_I.Iexc[:].T), label='Iexc')
legend()
show()
# yscale('log')
for i in range(0, N, 9):
    plot(statemon.t/ms, statemon.Vp[i]/mV, label=f'{i+1}')
legend()
show()

# plot(statemon_Int.t/ms, np.abs(statemon_Int.Vi[0]/mV), label='Vi')
# show()

# plot(statemon.t/ms, statemon.Vp[0]/mV)
# # for t in spikemon.t:
# #     axvline(t/ms, ls='--', c='C1', lw=3)
# xlabel('Time (ms)')
# ylabel('v');

# print(spikemon.i.shape)
# scatter(spikemon.t / ms, spikemon.i[:], marker="_", color="k", s=10)
# xlim(0, tfinal / ms)
# ylim(0, N)
# ylabel("neuron number")
# set_yticks(np.arange(0, len(N), 1))
# spines['right'].set_visible(False)
# spines['top'].set_visible(False)
# text(500, 400, 'excitatory', backgroundcolor='w', color='k', ha='center')
# axhline(Ne, color="k")
# text(500, 900, 'inhibitory', backgroundcolor='w', color='k', ha='center')

# show()

# eqs = '''
# dV/dt = (2-V)/(10*ms) : 1
# '''
# threshold = 'V>1'
# reset = 'V = 0'
# G = NeuronGroup(100, eqs, threshold=threshold, reset=reset)
# G.V = rand(len(G))
# M = StateMonitor(G, True, record=range(5))
# run(100*ms)
# plot(M.t, M.V.T)
# show()