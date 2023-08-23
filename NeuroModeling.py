import numpy as np
from brian2 import *


defaultclock.dt = 0.01*ms

C_pr = 3 * uF
VL_pr = 0.0 * mV
VK_pr = -15 * mV
VNa_pr = 120 * mV
VCa_pr = 140 * mV
Vsyn_pr_pr = 60 * mV  # Reverse potential for pyramid <=> pyramid interation
g_leak_pr = 0.1 * mS
g_Na_pr = 30 * mS
g_KDR_pr = 15 * mS
g_KAHP_pr = 0.8 * mS
g_Ca_pr = 10 * mS
g_KC_pr = 15 * mS
g_C_pr = 1.0 * mS
g_NMDA_pr_pr = 0.0 * uS  # Synaptic weight for NMDA pyramid <=> pyramid interaction
g_AMPA_pr_pr =  0.1 * uS #  0.01 * uS  # Synaptic weight for AMPA pyramid <=> pyramid interaction
p_pr = 0.5

# Is_ext_pr = 0.0 * uA # 1.0 * uA  # External current on soma of pyramids 2.5

Id_ext_pr = 0.0 * uA  # External current on dendrits of pyramids
sigmas = 1 * mV  # Parameters of the Gaussian noise for soma and dendrites
tausgauss = 1 * ms
sigmad = 1 * mV
taudgauss = 1 * ms

eqs = '''
# Soma
dvs_pr/dt = (-g_leak_pr*(vs_pr - VL_pr) - g_Na_pr*(m_inf_pr**2)*h_pr*(vs_pr - VNa_pr) -
    g_KDR_pr*n_pr*(vs_pr - VK_pr) + (g_C_pr/p_pr)*(vd_pr - vs_pr) + Is_ext_pr/p_pr)/C_pr +
    sigmas*xi*sqrt(tausgauss)/tausgauss : volt
dn_pr/dt = (n_inf_pr - n_pr)/taun_pr : 1
dh_pr/dt = (h_inf_pr - h_pr)/tauh_pr : 1
m_inf_pr = alpha_m_pr/(alpha_m_pr + beta_m_pr) : 1
n_inf_pr = alpha_n_pr/(alpha_n_pr + beta_n_pr) : 1
h_inf_pr = alpha_h_pr/(alpha_h_pr + beta_h_pr) : 1
taun_pr = 1/(alpha_n_pr + beta_n_pr) : second
tauh_pr = 1/(alpha_h_pr + beta_h_pr) : second
alpha_m_pr = 1.28*((13.1*mV - vs_pr)/(4*mV))/(exp((13.1*mV - vs_pr)/(4*mV)) - 1)/ms : Hz
beta_m_pr = 1.4*((vs_pr - 40.1*mV)/(5*mV))/(exp((vs_pr - 40.1*mV)/(5*mV)) - 1)/ms : Hz
alpha_h_pr = 0.128*exp((17*mV - vs_pr)/(18*mV))/ms : Hz
beta_h_pr = 4/(1 + exp((40*mV - vs_pr)/(5*mV)))/ms : Hz
alpha_n_pr = 0.08*((35.1*mV - vs_pr)/(5*mV))/(exp((35.1*mV - vs_pr)/(5*mV))- 1)/ms: Hz
beta_n_pr = 0.25*exp((0.5*mV - 0.025*vs_pr)/mV)/ms : Hz

# Is_ext_pr_timed = Is_ext_pr * exp(-0.1 * t**2 / ms**2) :  amp

# Dendrits
dvd_pr/dt = (-g_leak_pr*(vd_pr-VL_pr) - g_Ca_pr*(s_pr**2)*(vd_pr - VCa_pr) -
    g_KAHP_pr*q_pr*(vd_pr - VK_pr) - g_KC_pr*c_pr*chi_pr*(vd_pr - VK_pr) +
    g_C_pr/((1 - p_pr))*(vs_pr - vd_pr) + (Isyn_pr_pr + Isyn_wb_pr + Id_ext_pr)/(1 - p_pr))/C_pr +
    sigmad*xi_*sqrt(taudgauss)/taudgauss : volt
ds_pr/dt = (s_inf_pr - s_pr)/taus_pr : 1
dc_pr/dt = (c_inf_pr - c_pr)/tauc_pr : 1
dq_pr/dt = (q_inf_pr - q_pr) /tauq_pr : 1
dCa_pr/dt = (-0.13*(g_Ca_pr)*(s_pr**2)*(vd_pr - VCa_pr)/uA - 0.075*Ca_pr)/ms : 1
s_inf_pr = alpha_s_pr/(alpha_s_pr + beta_s_pr) : 1
c_inf_pr = alpha_c_pr/(alpha_c_pr + beta_c_pr) : 1
q_inf_pr = alpha_q_pr/(alpha_q_pr + beta_q_pr) : 1
taus_pr = 1/(alpha_s_pr + beta_s_pr) : second
tauc_pr = 1/(alpha_c_pr + beta_c_pr) : second
tauq_pr = 1/(alpha_q_pr + beta_q_pr) : second
alpha_s_pr = 1.6/(1 + exp(-0.072*(vd_pr - 65*mV)/mV))/ms : Hz
beta_s_pr = 0.1*((vd_pr - 51.1*mV)/(5*mV))/(exp((vd_pr - 51.1*mV)/(5*mV)) - 1)/ms : Hz
alpha_c_pr = (exp((vd_pr - 10*mV)/(11*mV)) - exp((vd_pr - 6.5*mV)/(27*mV)))/18.975/ms*int(vd_pr <= 50*mV) +
    2.0*exp((6.5*mV - vd_pr)/(27*mV))/ms *int(vd_pr > 50*mV) : Hz
beta_c_pr = 2.0*exp((6.5*mV - vd_pr)/(27*mV))/ms - alpha_c_pr : Hz
alpha_q_pr = (0.00002*Ca_pr *int(0.00002*Ca_pr < 0.01) + 0.01*int(0.00002*Ca_pr >= 0.01))/ms : Hz
beta_q_pr = 0.001/ms : Hz
chi_pr = Ca_pr/250*int(Ca_pr/250 < 1) + 1*int(Ca_pr/250 >= 1) : 1
vNMDA_pr_pr : 1          # Artificial variable for summing postsynaptic voltage of pyramids for NMDA synapse
vNMDA_wb_pr : 1        # Artificial variable for summing postsynaptic voltage of interneurons for NMDA synapse
vAMPA_pr_pr : 1        # Artificial variable for summing postynaptic voltage of pyramids for AMPA synapse
vAMPA_wb_pr : 1        # Artificial variable for summing postynaptic voltage of interneurons for AMPA synapse
Isyn_pr_pr : amp       # Synaptic current from pyramids
Isyn_wb_pr : amp       # Synaptic current from interneurons
Is_ext_pr  : amp 
'''
#alpha_s = 1.6/(1+exp(-0.072/mV*(Vd-65*mV)))/ms:Hz
#beta_s = 0.02(Vd-51.1*mV)/exp((Vd-51.1*mV)/(5*mV)-1)
#alpha_c = 2*exp((6.5*mV-Vd)/(27*mV)
#beta_c = 2*exp((6.5*mV-Vd)/(27*mV)-alpha_c
#alpha_q = min((0.00002)*Ca
#dCa/dt = -0.13*gCa*s**2(Vd-ECa)-0.075Ca
#s = alpha_s/(alpha_s+beta_s) : 1
#threshold='v>-48*mV', reset='v = -70*mV', refractory=3*ms,
#+gC(Vd-Vs)
#dVd/dt = (-gNa*m**3*h*(Vd-ENa)-gK*n**4*(Vd-EK)-gL*(Vd-EL)-gC(Vd-Vs)+Iappd)/Cm : volt

G_pr = NeuronGroup(1, eqs, threshold='vs_pr > 20*mV', reset='vs_pr=0*mV', refractory=10*ms, method='euler')


G_pr.vs_pr = '(rand() - 25)*mV'
G_pr.vd_pr = -4.5 * mV
G_pr.h_pr = 0.999
G_pr.n_pr = 0.001
G_pr.s_pr = 0.009
G_pr.c_pr = 0.007
G_pr.q_pr = 0.01
G_pr.Ca_pr = 0.2


Is_ext = np.zeros(1, dtype=np.float64)
Is_ext[0:10] += 2.0
G_pr.Is_ext_pr = Is_ext * uA


G_pr_non_active = NeuronGroup(1, eqs, threshold='vs_pr > 20*mV', reset='vs_pr=-10*mV', refractory=2*ms, method='euler')
G_pr_non_active.vs_pr = '(50*rand() - 25)*mV'
G_pr_non_active.vd_pr = -4.5 * mV
G_pr_non_active.h_pr = 0.999
G_pr_non_active.n_pr = 0.001
G_pr_non_active.s_pr = 0.009
G_pr_non_active.c_pr = 0.007
G_pr_non_active.q_pr = 0.01
G_pr_non_active.Ca_pr = 0.2
G_pr_non_active.Is_ext_pr = 0.0 * uA

# Simulation
StMs_pr = StateMonitor(G_pr, 'vs_pr', record=True)
StMd_pr = StateMonitor(G_pr, 'vd_pr', record=True)

SpM_pr = SpikeMonitor(G_pr)



StMs_pr_non_active = StateMonitor(G_pr_non_active, 'vs_pr', record=True)
StMd_pr_non_active = StateMonitor(G_pr_non_active, 'vd_pr', record=True)

run(300*ms, report='text')

# Graphical output
figure(figsize=(14, 4))
axhline(VL_pr / mV, ls='-', c='lightgray', lw=3) # horisontal line at the level VL_pr

plot(StMs_pr.t / ms, StMs_pr.vs_pr.T / mV)
xlabel('Time (ms)')
ylabel('vs_pr (mV)');


figure(figsize=(14, 4))
plot(SpM_pr.t / ms, SpM_pr.i, '.r')
xlabel('Time (ms)')
ylabel('Neuron index')



show()




