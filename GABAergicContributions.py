import numpy as np
from brian2 import *

g_max = 0.1 * mS/cm**2
w_s = 0.5
w_s_np = 1
Ch_s = 0.005 *uF/cm**2
Ch_p = 0.01 *uF/cm**2
Ch_d = 0.02 *uF/cm**2

#g_syn = (w_s * w + W)* g_max
W3 = 0;
wec_to_pcAMPA=1.4 + W3 
waac_to_pc =1.0
W1 = 0;
wca3_to_pcAMPA =2.4 + W1 
wbc_to_pc= 0.1
wec_to_pcNMDA =1.4 + W3
wbc_to_bsc =20
wca3_to_pcNMDA =2.4 + W1 
wbsc_to_pc =0.3
wec_to_aac =0.9 
wpc_to_bsc=0
wca3_to_aac= 0.8 
wbsc_to_bc= 0.5
wec_to_bc=0.8 
wolm_to_pc=0.5
wca3_to_bc =0.8 
wpc_to_olm =1.1
wca3_to_bsc= 2 
wivy_to_pc =0.15
wsep360_to_aac =10 
wpc_to_ivy =1
wsep360_to_bc=10 
wec_to_ngl =3
wsep180_to_bsc =8 
wngl_to_pc =0.8
wsep180_to_olm= 30 
wbc_to_pc= 0.1
wOLM_to_NGL= 1,500

Cm =1*uF/cm**2
gA_dend =12*mS/cm**2
gL =0.1*mS/cm**2
asap =0.001
VL =270*mV
psi = psi_p =30*mV#?
g_axon_soma = g_coup=1.125*mS/cm**2
g_soma_axon = g_coup=1.125*mS/cm**2
g_soma_dendprox= g_coup=1.125*mS/cm**2
g_dendprox_denddist= g_coup=1.125*mS/cm**2

inact= 72*mV
gNa_soma =30 *mS/cm**2
inact2 =0.11*mV
gNa_axon =100 *mS/cm**2
inact3 =2
VNa =60*mV
inact4 =64*mV
gNa_dend =30*mS/cm**2
inact5 =1*mV
natt =0 
gKdr_axon =20*mS/cm**2
T =23*K
gKdr_soma =14*mS/cm**2
gA_soma =7.5*mS/cm**2
gKdr_d= 14*mS/cm**2
gmAHP= 25*mS/cm**2
Vk= 280*mV
qhat =1 
qma =0.00048
qmb= 0.28 
β_s =0.083*1/ms
gCaLs =7*mS/cm**2
β_d =0.083*1/ms
gCaLd =25*mS/cm**2 
s =0
VCa =140*mV
nonc =6
Ca2p =2 
Ca_tau =1,000
ϕ_s =0.1 
s1 =0
ϕ_d =0.1 
s2 =40
ksi_0s =0.05 
s3 =3.6
ksi_0d= 0.07 
Mg2p =2#Mg2+
k= 7#j
tauMNa_d = 0.1#*ms
tauHNa_d = 0.5#*ms
tauN_d = 2.2#*ms
tau_Td = 29#*ms
#psi =ς
#ksi = χ

c_p =5 
num_b= 1
tau_p= 500 
num_c= 1
tau_v= 10 
num_d= 1
tau_A= 5 
num_e= 5
tau_B= 40 
CmHC= 4*uF/cm**2
tau_D= 250 
CmHN= 4*uF/cm**2
tau_w= 500 
CnHC= 0.6*uF/cm**2
α_w= 0.8 *1/ms
CnHN= 3*uF/cm**2
β_w= 0.6 *1/ms
θ_c= 2
aw= 0.3 
θ_d= 2.6
pw_a= -0.1 
θ_e= 0.55
dw= 0.05 
σ_c= -0.05
pw_d=-0.002 
σ_d =-0.01
num_a= 10 
σ_e= -0.02
cd= 4

Cm= 1 *uF/cm**2
gKdr_axon =23*mS/cm**2 
gL =0.18 *mS/cm**2 
gKdrOLM= 36*mS/cm**2 
VL =260 *mV
Vk_OLM =277*mV
gNa= 150 *mS/cm**2 
#Vk =290*mV
gNa_OLM= 120 *mS/cm**2 
gNaP =2.5*mS/cm**2 
#VNa= 55 *mV
gh= 1.5*mS/cm**2 
VNa_OLM= 50 *mV
VNaP= 50*mV
gL_OLM= 0.3 *mS/cm**2 
Vh =220*mV
VL_OLM =254.4 *mV
gA =10*mS/cm**2 

gCa_NMDA =25*mS/cm**2  
VCa_NMDA =140*mV
gNMDA =0.3*mS/cm**2 
VNMDA =0*mV
gAMPA =0.05*mS/cm**2  
VAMPA =0*mV
gGABA =0.05*mS/cm**2  
VGABA =275*mV

aAAC2PC =5*1/ms
bAAC2PC =0.01*1/ms
aBC2PC =5 *1/ms
bBC2PC =0.015*1/ms
aBSC2PC =5 *1/ms
bBSC2PC =0.01*1/ms
aOLM2PC =5 *1/ms
bOLM2PC =0.01*1/ms
aIVY2PC =1 *1/ms
bIVY2PC=0.0015*1/ms
aBC2BC =3.5 *1/ms
bBC2BC =0.18*1/ms
aBC2BSC =3.5 *1/ms
bBC2BSC =0.18*1/ms
aBSC2BC =3.5 *1/ms
bBSC2BC =0.18*1/ms
aNGL2PC =5 *1/ms
bNGL2PC =0.015*1/ms
aPC2BC= 20 *1/ms
bPC2BC =0.19*1/ms
aPC2BSC= 20 *1/ms
bPC2BSC= 0.19*1/ms
aPC2AAC =20 *1/ms
bPC2AAC =0.19*1/ms
aPC2IVY =20 *1/ms
bPC2IVY= 0.19*1/ms
aPC2OLM= 20 *1/ms
bPC2OLM =0.19*1/ms
aOLM2NGL =5 *1/ms
bOLM2NGL =0.01*1/ms

Iin = 5*uA
gKA_d = 12*mS/cm**2#gA_dend
Eh = -44*mV#?
Vhalf = -50*mV#?
Vhalft = -50*mV#?
k_l = 7#?
wsep_to_ivy = 12#?
a0 = q = t = l = gmt = 1#?
#tauDNa_pd = max(0.1 , (0.00333 * exp(0.0024 * (Vpd + 60) * Q))/(1 + exp(0.0012 * (Vpd + 60) * Q)))
#tauDNa_dd = max(0.1 , (0.00333 * exp(0.0024 * (Vdd + 60) * Q))/(1 + exp(0.0012 * (Vdd + 60) * Q)))
H = 1#?
Tper = 0.250*ms
VK = -90*mV

def H(x):
    if x>0: return 1
    else: return 0


def LeakageCurrent(V):
    return gL*(V-VL)

eqs ='''
#PCs
Cm*dVax/dt = IL + INa_ax + IKdr_axon + Iaxon_coup + Ipc_syn_sum_ax + Iin :volt*farad/meter**2
Cm*dVs/dt = IL + INa_s + IKdr_soma + IKA_s + Im_AHP + ICaL_s + Ih_s + Isoma_coup + Ipc_syn_sum_s + Iin :volt*farad/meter**2
Cm*dVpd/dt = IL + INa_pd + IKdr_pd + IKA_pd + ICaL_pd + Ih_pd + Idendprox_coup + Ipc_syn_sum_pd + Iin_pd :volt*farad/meter**2
Cm*dVdd/dt = IL + INa_dd + IKdr_dd + IKA_dd + ICaL_dd + Ih_dd + Idenddist_coup + Ipc_syn_sum_dd + Iin_dd :volt*farad/meter**2

Iaxon_coup = g_axon_soma * (Vs - Vax) :ampere
Isoma_coup = g_soma_axon * (Vax  - Vs) + g_soma_dendprox * (Vpd - Vs):ampere
Idendprox_coup = g_soma_dendprox * (Vs - Vpd) + g_denddist_dendprox *(Vdd-Vpd):ampere
Idenddist_coup = g_dendprox_denddist* (Vpd-Vdd):ampere

IL_ax = LeakageCurrent(Vax):ampere
IL_s = gL*(Vs-VL):ampere
IL_pd = gL*(Vpd-VL):ampere
IL_dd = gL*(Vdd-VL):ampere

INa_ax = -gNa_axon * (MNa_ax**2) * HNa_ax * (Vax-VNa):ampere
INa_s = -gNa_soma * (MNa_s**2) * HNa_s * (Vs-VNa):ampere
INa_pd = -gNa_dend * (MNa_pd**2) * HNa_pd * DNa_pd * (Vpd - VNa):ampere
INa_dd = -gNa_dend * (MNa_dd**2) * HNa_dd * DNa_dd * (Vdd - VNa):ampere

IKA_s = -gKA_d * Apd * Bd * (Vs - VK):ampere
IKA_pd = -gKA_d * Apd * Bd * (Vpd - VK):ampere
IKA_dd = -gKA_d * Add * Bd * (Vdd - VK):ampere

IKdr_axon = -gKdr_axon * Nax * (Vax - VK):ampere
IKdr_soma = -gKdr_soma * Ns * (Vs - VK):ampere
IKdr_pd = -gKdr_d * Npd**2 * (Vpd - VK):ampere
IKdr_dd = -gKdr_d * Ndd**2 * (Vdd - VK):ampere

ImAHP = -gmAHP * Qm * (Vs - VK):ampere

ICaL_s = -gCaLs * S_s * g_hk*(1/(1 + ksi_s)):ampere
ICaL_pd = -gCaLd * (S_pd**3) * T_pd * (Vpd - VCa):ampere                    #S and T are not depending on Ca concentration
ICaL_dd = -gCaLd * (S_dd**3) * T_dd * (Vdd - VCa):ampere                    

Ih_s = -gh * tt_s* (Vs - Eh):ampere
Ih_pd = -gh * tt_pd * (Vpd - Eh):ampere
Ih_dd = -gh * tt_dd * (Vdd - Eh):ampere

Iin_pd =  -g_syn_ca3nmda2pc * sNMDA * mNMDA * (Vpd - VNMDA) -g_syn_ca3ampa2pc * sAMPA * (Vpd - VAMPA) :ampere

Iin_dd = -g_syn_ec_nmda2pc * sNMDA * mNMDA * (Vdd - VNMDA) -g_syn_ec_ampa2pc * sAMPA * (Vdd - VAMPA) :ampere

MNa_s = αM_s/(αM_s + βM_s):1
αM_s = 0.32 * (-46.9-Vs/mV )/(exp((-46.9*mV - Vs )/(4.0*mV)) - 1.0):1 
βM_s = 0.28 * (Vs/mV + 19.9)/(exp((Vs + 19.9*mV)/(5.0*mV)) - 1.0):1

MNa_ax = αM_ax/(αM_ax + βM_ax):1
αM_ax = 0.32 * (-46.9-Vax/mV )/(exp((-46.9*mV - Vax )/(4.0*mV)) - 1.0):1
βM_ax = 0.28 * (Vax/mV + 19.9)/(exp((Vax + 19.9*mV)/(5.0*mV)) - 1.0):1

dHNa_s/dt = αH_s - (αH_s + βH_s) * HNa_s:1
αH_s = 0.128 * exp((-43*mV - Vs )/(18*mV)):1
βH_s = 4/(1 + exp((-20*mV - Vs )/(5*mV)):1

dHNa_ax/dt = αH_ax - (αH_ax + βH_ax) * HNa_ax:1
αH_ax = 0.128 * exp((-43*mV - Vax )/(18*mV)):1
βH_ax = 4/(1 + exp((-20*mV - Vax )/(5*mV)):1

dMNa_pd/dt = (MinfNa_pd - MNa_pd)/tauMNa_d:1
MinfNa_pd = 1/(1+exp((-Vpd-40*mV)/(3*mV))):1
dMNa_dd/dt = (MinfNa_dd - MNa_dd)/tauMNa_d:1
MinfNa_dd = 1/(1+exp((-Vdd-40*mV)/(3*mV))):1
dHNa_pd/dt = (HinfNa_pd - HNa_pd)/tauHNa_d:1
HinfNa_pd = 1/(1 + exp((Vpd + 45*mV)/(3*mV))):1
dHNa_dd/dt = (HinfNa_dd - HNa_dd)/tauHNa_d:1
HinfNa_dd = 1/(1 + exp((Vdd + 45*mV)/(3*mV))):1
dDNa_pd/dt = (DinfNa_pd -DNa_pd)/tauDNa_d:1
DinfNa_pd = (1 + natt * exp((Vpd + 60*mV)/(2*mV)))/(1 + exp((Vpd + 60*mV)/(2*mV))):1       #natt=0? p1617, table A2, and equation for D∞ below table
dDNa_dd/dt = (DinfNa_dd -DNa_dd)/tauDNa_d:1
DinfNa_dd = (1 + natt * exp((Vdd + 60*mV)/(2*mV)))/(1 + exp((Vdd + 60*mV)/(2*mV))):1
tauDNa_pd = max(0.1 , (0.00333 * exp(0.0024 * (Vpd/mV + 60) * Q))/(1 + exp(0.0012 * (Vpd/mV + 60) * Q))):1 
tauDNa_dd = max(0.1 , (0.00333 * exp(0.0024 * (Vdd/mV + 60) * Q))/(1 + exp(0.0012 * (Vdd/mV + 60) * Q))):1
Q = 96480*K/(8.315 * (273.16*K+ T)):1

dApd/dt = (Ainf_pd - Apd)/tauA_pd:1
Ainf_pd = 1/(1 + Aα_pd):1
dAdd/dt = (Ainf_dd - Add)/tauA_dd:1
Ainf_dd = 1/(1 + Aα_dd):1
Aα_pd = exp(asap * psi_pd*(Vpd + 1*mV)/mV * Q):1
Aα_dd = exp(asap * psi_dd*(Vdd + 1*mV)/mV * Q):1
Aβ_pd = exp(0.00039 * Q * (Vpd + 1*mV)/mV * psi2_pd):1
Aβ_dd = exp(0.00039 * Q * (Vdd + 1*mV)/mV * psi2_dd):1
tauA_pd = max(Aβ_pd/((1 + Aα_pd) * Q*T/K * 0.1), 0.1):1
tauA_dd = max(Aβ_dd/((1 + Aα_dd) * Q*T/K * 0.1), 0.1):1
psi_s=-1.5 - (1/(1 + exp((Vs/mV + psi_p)/5))):1
psi_pd=-1.5 - (1/(1 + exp((Vpd/mV + psi_p)/5))):1
psi_dd=-1.5 - (1/(1 + exp((Vdd/mV + psi_p)/5))):1
psi2_pd=-1.8 - (1/(1 + exp((Vpd/mV + 40)/5))):1
psi2_dd=-1.8 - (1/(1 + exp((Vdd/mV + 40)/5))):1

dBd/dt = (Binf_d - Bd)/tauB_d:1
Binf_d = 0.3 + 0.7/(1 + exp(inact2 * (Vs + inact)/mV**2 * Q)):1
tauB_d = k * max(inact3 * (Vs + inact4)/mV**2, inact5/mV):1

dNs/dt = αNs-(αNs + βNs) * Ns:1
αNs = 0.016 * (-24.9 - Vs/mV )/(exp((24.9 - Vs/mV )/5) -1):1
βNs = 0.25 * exp(-1 - 0.025 * Vs/mV ):1

dNax/dt = αNax-(αNax + βNax) * Nax:1
αNax = 0.016 * (-24.9 - Vax/mV )/(exp((24.9 - Vax/mV)/5) -1):1
βNax = 0.25 * exp(-1 - 0.025 * Vax/mV ):1

dNpd/dt = (Ninf_pd - Npd)/tauN_d:1
Ninf_pd = 1/(1 + exp((-Vpd/mV - 42)/2):1
dNdd/dt = (Ninf_dd - Ndd)/tauN_d:1
Ninf_dd = 1/(1 + exp((-Vdd/mV - 42)/2):1

dQm/dt = (Qminf - Qm)/tauQm:1
Qminf = qhat * Qmα * tauQm:1
Qmα = qma * ksi_s/(0.001 * ksi_s + 0.18 * exp(-1.68 * Vs/mV * Q)):1
Qmβ = (qmb * exp(-0.022 * Vs/mV * Q))/(exp(-0.022 * Vs/mV * Q)+ 0.001 * ksi_s):1
tauQm = 1/(Qmα + Qmβ ):1

dtt_s/dt = (tt_inf_s - tt_s/)tau_tt_s:1
tt_inf_s = 1/(1 + exp(-(Vs -Vhalf)/mV)/k_l):1
tau_tt_s = exp(0.0378*psi_s*gmt*(Vs - Vhalft)/mV)/q*t*l *q*10**((T/K-33)/10) * a0*t * (1 + a_tt_s):1
a_tt_s = exp(0.00378*psi_s*(Vs -Vhalft)/mV):1

dtt_pd/dt = (tt_inf_pd - tt_pd/)tau_tt_pd:1
tt_inf_pd = 1/(1 + exp(-(Vpd -Vhalf)/mV)/k_l):1
tau_tt_pd = exp(0.0378*psi_pd*gmt*(Vpd - Vhalft)/mV)/q*t*l *q*10**((T-33)/10) * a0*t * (1 + a_tt_pd):1              #gmt, a0, q, t, l? 
a_tt_pd = exp(0.00378*psi_pd*(Vpd -Vhalft)):1                                                                                                                           #- не определены значения в статье
                                                                                                                                                                                                            #первые три уравнения стр. 1618
dtt_dd/dt = (tt_inf_dd - tt_dd)/tau_tt_dd:1
tt_inf_dd = 1/(1 + exp(-(Vdd -Vhalf)/mV)/k_l):1
tau_tt_dd = exp(0.0378*psi_dd*gmt*(Vdd - Vhalft)/mV)/q*t*l *q*10**((T-33*K)/10) * a0*t * (1 + a_tt_dd):1
a_tt_dd = exp(0.00378*psi_dd*(Vdd -Vhalft)/mV):1

dS_s/dt = (Sinf -  S_s)/tau_s:1
Sinf = α_s/(α_a + β_s):1
tau_s = 1/(5 * (α_s+ β_s)):1
α_s= -0.055 * (Vs/mV + 27.01)/(exp((-Vs/mV - 27.01)/3.8) - 1):1
β_s = 0.94 * exp((-Vs/mV - 63.01)/17)
xx = 0.0853 * (273.16 + T/K)/2:1                                                                                                                
f = (1 - Vs/mV/(2*xx)) * f2+(Vs/mV/(xx*(exp(Vs/mV/xx) - 1))) * f3:1
f2 = H(0.0001-abs(Vs/mV/xx)):1
f3 = H(abs(Vs/mV/xx) - 0.0001):1
g_hk = -xx * (1 - ((ksi_s/Ca) * exp(Vs/mV/xx))) * f :1                                          #Ca? equation right before eq.A17, 1618
                                                                                                                                                #Ca2+ = 2 : 1617, table A2, first column,  (Ca_tau = 1000).
                                                                                                                                                 #Written that "the Ca2+ concentrations in the soma
                                                                                                                                                 #and dendrites are given by ksi eqs:"
                                                                                                                                                 #So Ca2+ concentration is 2, or it's given by ksi eqs?
                                                                                                                                                 #Ca determines g_hk, g_hk det. ICaL_s, which det. ksi_s, 
                                                                                                                                                 #but where is Ca conc. changing, and why it's influences only soma comp, but not dendrites?
                                                                                                                                                 #W and so g_syn is determed. by ksi_pd, ksi_dd but not ksi_s

dksi_s/dt = ϕ_s * ICaL_s - (β_s * (ksi_s - ksi_0s)) + (ksi_pd - ksi_s)/Ca_tau - (β_s/nonc) * ksi_s**2:1
dksi_pd/dt = ϕ_d * (ICaL_pd + ICa_NMDA) - β_d * (ksi_pd - ksi_0d) - (β_d/nonc)*ksi_pd**2 - buff *ksi_pd:1             #buff? eqs (A18), (A19), p1618
dksi_dd/dt = ϕ_d * (ICaL_dd + ICa_NMDA) - β_d * (ksi_dd - ksi_0d) - (β_d/nonc)*ksi_dd**2 - buff *ksi_dd:1

dS_pd/dt = (Sinfpd - S_pd)/tau_pd:1                                                                                              
Sinfpd = 1/(1 + exp(-Vpd/mV - 37)):1                                                                                                        
dS_dd/dt = (Sinfdd - S_dd)/tau_dd:1
Sinfdd = 1/(1 + exp(-Vdd/mV - 37)):1
tau_pd = s3 + s1/(1 + exp(Vpd/mV + s2)):1
tau_dd = s3 + s1/(1 + exp(Vdd/mV + s2)):1
dT_pd/dt = (Tinfpd - T_pd)/tau_Tp:1
Tinfpd = 1/(1 + exp((Vpd/mV + 41)/0.5)):1
dT_dd/dt = (Tinfdd - T_dd)/tau_Td:1
Tinfdd = 1/(1 + exp((Vdd/mV + 41)/0.5)):1
tau_Td = 29:1


dP_pd/dt = (ϕ_a_pd - c_p*A_pd*P_pd)/tau_p:1
dP_dd/dt = (ϕ_a_dd - c_p*A_dd*P_dd)/tau_p:1
dVdet_pd/dt = (ϕ_b_pd - Vdet_pd )/tau_V:1
dVdet_dd/dt = (ϕ_b_dd - Vdet_dd )/tau_V:1
dA_pd/dt = (ϕ_c_pd - A_pd )/tau_A:1
dA_dd/dt = (ϕ_c_dd - A_dd )/tau_A:1
dB_pd/dt = (ϕ_e_pd - B_pd - c_d*B_pd*Vdet_pd)/tau_b:1
dB_dd/dt = (ϕ_e_dd - B_dd - c_d*B_dd*Vdet_dd)/tau_b:1
dD_pd/dt = (ϕ_d_pd - D_pd )/tau_D:1
dD_dd/dt = (ϕ_d_dd - D_dd )/tau_D:1
dW_pd/dt = (α_w/((1 + exp((P_pd - aw)/pw_a)) - β_w/(1 + (exp((D_pd - dw)/pw_d)) - W_pd )/tau_w):1
dW_dd/dt = (α_w/((1 + exp((P_dd - aw)/pw_a)) - β_w/(1 + (exp((D_dd - dw)/pw_d)) - W_dd )/tau_w):1
ϕ_a_pd = num_a * ((ksi_pd/CmHC)**CmHN)/(1 + (ksi_pd/CmHC)**CmHN):1
ϕ_a_dd = num_a * ((ksi_dd/CmHC)**CmHN)/(1 + (ksi_dd/CmHC)**CmHN):1
ϕ_b_pd = num_b * ((ksi_pd/CnHC)**CnHN)/(1 + (ksi_pd/CnHC)**CnHN):1
ϕ_b_dd = num_b * ((ksi_dd/CnHC)**CnHN)/(1 + (ksi_dd/CnHC)**CnHN):1
ϕ_c_pd = num_c/(1 + exp((ksi_pd - θ_c)/σ_c)):1
ϕ_c_dd = num_c/(1 + exp((ksi_dd - θ_c)/σ_c)):1
ϕ_d_pd = num_d/(1 + exp((B_pd - θ_d)/σ_d)):1
ϕ_d_dd = num_d/(1 + exp((B_dd - θ_d)/σ_d)):1
ϕ_e_pd = num_e/(1 + exp((A_pd - θ_e)/σ_e)):1
ϕ_e_dd = num_e/(1 + exp((A_dd - θ_e)/σ_e)):1
'''

eqsBC = '''
#BC, AAC, BSC, IVY
Cm*dVbc/dt = IL_bc + INa_bc + IK_dr_bc + IA_bc + Iin_bc + Isyn_sum_bc :volt*farad/meter**2

INa_bc = gNa*(m_bc**3)*h_bc*(Vbc - 50*mV):ampere             
IK_dr_bc = gK_dr*(n_bc**4)*(Vbc + 90*mV ):ampere
IA_bc = gA*a_bc*b_bc*(Vbc + 90*mV):ampere
Iin_bc = -g_syn_ca3ampa2bc * sAMPA * (Vbc - VAMPA) -g_syn_ec_ampa2bc * sAMPA * (Vbc - VAMPA) -g_syn_ms2bc * sGABA * (Vbc - VGABA): ampere

dm_bc/dt = α_m_bc*(1-  m_bc) - β_m_bc*m_bc:1
α_m_bc = 0.1*(Vbc/mV + 40)/(1 - exp((Vbc/mV +40)/10):1
β_m_bc = 4 * exp(-(Vbc/mV+65)/18):1
dh_bc/dt = α_h_bc*(1-  h_bc) - β_h_bc*h_bc:1
α_h_bc = 0.07*exp(-(Vbc/mV +65)/20):1
β_h_bc =1 / (1+ exp(-(Vbc/mV+35)/10)):1

dn_bc/dt = α_n_bc*(1-  n_bc) - β_n_bc*n_bc:1
α_n_bc = 0.01*(Vbc/mV + 55)/(1 - exp(-(Vbc/mV +55)/10):1
β_n_bc = 0.125*exp(-(Vbc/mV+65)/80):1

da_bc/dt = α_a_bc*(1-  a_bc) - β_a_bc*a_bc:1
α_a_bc = 0.02*(13.1-Vbc/mV)/(exp((Vbc/mV -40.1)/10)-1):1
β_a_bc = 0.0175*(Vbc/mV-40.1)/(exp(-(Vbc/mV-40.1)/10)-1):1

db_bc/dt = α_b_bc*(1-  b_bc) - β_b_bc*b_bc:1
α_b_bc = 0.0016*exp((-Vbc/mV -13)/18):1
β_b_bc = 0.05/(1+ exp(-(10.1-Vbc/mV)/5)):1
'''

eqsAAC = '''
Cm*dVaac/dt = IL_aac + INa_aac + IK_dr_aac + IA_aac + Iin_aac + Isyn_sum_aac :volt*farad/meter**2

INa_aac = gNa*(m_aac**3)*h_aac*(Vaac - ENa):ampere
IK_dr_aac = gK_dr*(n_aac**4)*(Vaac - EK ):ampere
IA_aac = gA*a_aac*b_aac*(Vaac - EK):ampere
Iin_aac = -g_syn_ca3ampa2aac * sAMPA * (Vaac - VAMPA) -g_syn_ec_ampa2aac * sAMPA * (Vaac - VAMPA) -g_syn_ms2aac * sGABA * (Vaac - VGABA) :ampere

dm_aac/dt = α_m_aac*(1-  m_aac) - β_m_aac*m_aac:1
α_m_aac = 0.1*(Vaac + 40)/(1 - exp((Vaac/mV +40)/10):1
β_m_aac = 4 * exp(-(Vaac/mV+65)/18):1

dh_aac/dt = α_h_aac*(1-  h_aac) - β_h_aac*h_aac:1
α_h_aac = 0.07*exp(-(Vaac/mV +65)/20):1
β_h_aac =1 / (1+ exp(-(Vaac/mV+35)/10)):1

dn_aac/dt = α_n_aac*(1-  n_aac) - β_n_aac*n_aac:1
α_n_aac = 0.01*(Vaac/mV + 55)/(1 - exp(-(Vaac/mV +55)/10):1
β_n_aac = 0.125*exp(-(Vaac/mV+65)/80):1

da_aac/dt = α_a_aac*(1-  a_aac) - β_a_aac*a_aac:1
α_a_aac = 0.02*(13.1-Vaac/mV)/(exp((Vaac/mV -40.1)/10)-1):1
β_a_aac = 0.0175*(Vaac/mV-40.1)/(exp(-(Vaac/mV-40.1)/10)-1):1

db_aac/dt = α_b_aac*(1-  b_aac) - β_b_aac*b_aac:1
α_b_aac = 0.0016*exp((-Vaac/mV -13)/18):1
β_b_aac = 0.05/(1+ exp(-(10.1-Vaac/mV)/5)):1
'''

eqsBSC = '''
Cm*dVbsc/dt = IL_bsc + INa_bsc + IK_dr_bsc + IA_bsc + Iin_bsc + Isyn_sum_bsc :volt*farad/meter**2

INa_bsc = gNa*(m_bsc**3)*h_bsc*(Vbsc - ENa):ampere
IK_dr_bsc = gK_dr*(n_bsc**4)*(Vbsc - EK ):ampere
IA_bsc = gA*a_bsc*b_bsc*(Vbsc - EK):ampere
Iin_bsc = -g_syn_ca3ampa2bsc * sAMPA * (Vbsc - VAMPA)  -g_syn_ms2bsc * sGABA * (Vbsc - VGABA) :ampere

dm_bsc/dt = α_m_bsc*(1-  m_bsc) - β_m_bsc*m_bsc:1
α_m_bsc = 0.1*(Vbsc/mV + 40)/(1 - exp((Vbsc/mV +40)/10):1
β_m_bsc = 4 * exp(-(Vbsc/mV+65)/18):1

dh_bsc/dt = α_h_bsc*(1-  h_bsc) - β_h_bsc*h_bsc:1
α_h_bsc = 0.07*exp(-(Vbsc/mV +65)/20):1
β_h_bsc =1 / (1+ exp(-(Vbsc/mV+35)/10)):1

dn_bsc/dt = α_n_bsc*(1-  n_bsc) - β_n_bsc*n_bsc:1
α_n_bsc = 0.01*(Vbsc/mV + 55)/(1 - exp(-(Vbsc/mV +55)/10):1
β_n_bsc = 0.125*exp(-(Vbsc/mV+65)/80):1

da_bsc/dt = α_a_bsc*(1-  a_bsc) - β_a_bsc*a_bsc:1
α_a_bsc = 0.02*(13.1-Vbsc/mV)/(exp((Vbsc/mV -40.1)/10)-1):1
β_a_bsc = 0.0175*(Vbsc/mV -40.1)/(exp(-(Vbsc/mV -40.1)/10)-1):1

db_bsc/dt = α_b_bsc*(1-  b_bsc) - β_b_bsc*b_bsc:1
α_b_bsc = 0.0016*exp((-Vbsc/mV -13)/18):1
β_b_bsc = 0.05/(1+ exp(-(10.1-Vbsc/mV)/5)):1
'''

eqsIVY = '''
Cm*dVivy/dt = IL_ivy + INa_ivy + IK_dr_ivy + IA_ivy + Iin_ivy + Isyn_sum_ivy :volt*farad/meter**2

INa_ivy = gNa*(m_ivy**3)*h_ivy*(Vivy - 50*mV):ampere
IK_dr_ivy = gK_dr*(n_ivy**4)*(Vivy + 90*mV):ampere                        
IA_ivy = gA*a_ivy*b_ivy*(Vivy + 90*mV):ampere
Iin_ivy = -g_syn_ms2bsc * sGABA * (Vbsc - VGABA) :ampere



dm_ivy/dt = α_m_ivy*(1-  m_ivy) - β_m_ivy*m_ivy:1
α_m_ivy = 0.1*(Vivy/mV + 40)/(1 - exp((Vivy +40)/10):1
β_m_ivy = 4 * exp(-(Vivy/mV+65)/18):1

dh_ivy/dt = α_h_ivy*(1-  h_ivy) - β_h_ivy*h_ivy:1
α_h_ivy = 0.07*exp(-(Vivy/mV +65)/20):1
β_h_ivy =1 / (1+ exp(-(Vivy/mV+35)/10)):1

dn_ivy/dt = α_n_ivy*(1-  n_ivy) - β_n_ivy*n_ivy:1
α_n_ivy = 0.01*(Vivy/mV + 55)/(1 - exp(-(Vivy/mV +55)/10):1
β_n_ivy = 0.125*exp(-(Vivy/mV+65)/80):1

da_ivy/dt = α_a_ivy*(1-  a_ivy) - β_a_ivy*a_ivy:1
α_a_ivy = 0.02*(13.1-Vivy/mV)/(exp((Vivy/mV -40.1)/10)-1):1
β_a_ivy = 0.0175*(Vivy/mV-40.1)/(exp(-(Vivy/mV-40.1)/10)-1):1

db_ivy/dt = α_b_ivy*(1-  b_ivy) - β_b_ivy*b_ivy:1
α_b_ivy = 0.0016*exp((-Vbsc/mV -13)/18):1
β_b_ivy= 0.05/(1+ exp(-(10.1-Vbsc/mV)/5)):1
'''
eqsNGL = '''
Cm*dVngl/dt = IL_ngl + INa_ngl + IK_dr_ngl + Iin_ngl + Isyn_sum_ngl :volt*farad/meter**2

INa_ngl = gNa*(m_ngl**3)*h_ngl*(Vngl - ENa):ampere
IK_dr_ngl = gK_dr*(n_ngl**4)*(Vngl - EK ):ampere
Iin_ngl = -g_syn_ec_ampa2bsc * sAMPA * (Vbsc - VAMPA) :ampere


dm_ngl/dt = α_m_ngl*(1-  m_ngl) - β_m_ngl*m_ngl:1
α_m_ngl = 0.1*(Vngl/mV + 40)/(1 - exp((Vngl/mV +40)/10):1
β_m_ngl = 4 * exp(-(Vngl/mV+65)/18):1
dh_ngl/dt = α_h_ngl*(1-  h_ngl) - β_h_ngl*h_ngl:1
α_h_ngl = 0.07*exp(-(Vngl/mV +65)/20):1
β_h_ngl =1 / (1+ exp(-(Vngl/mV+35)/10)):1
dn_ngl/dt = α_n_ngl*(1-  n_ngl) - β_n_ngl*n_ngl:1
α_n_ngl = 0.01*(Vngl/mV + 55)/(1 - exp(-(Vngl/mV +55)/10):1
β_n_ngl = 0.125*exp(-(Vngl/mV+65)/80):1
'''

eqsOLM = '''
Cm*dVolm/dt = IL_olm + INa_olm + IK_dr_olm + INaP +Ih_olm+ Iin_olm + Isyn_sum_olm :volt*farad/meter**2

INa_olm = gNa*(m_olm**3)*h_olm*(Volm - ENa):ampere
IK_dr_olm= gK_dr*(n_olm**4)*(Volm + 90*mV ):ampere
INaP = -gNaP*m_po*(Volm - VNa):ampere
Ih_olm = -g_h(0.65*λ_fo + 0.35*λ_so)(Volm - Vh):ampere
Iin_olm = -g_syn_ms2bsc * sGABA * (Vbsc - VGABA) :ampere

dm_olm/dt = α_m_olm*(1-  m_olm) - β_m_olm*m_olm:1
α_m_olm = 0.1*(Volm/mV + 40)/(1 - exp((Volm/mV +40)/10):1
β_m_olm = 4 * exp(-(Volm/mV+65)/18):1
dh_olm/dt = α_h_olm*(1-  h_olm) - β_h_olm*h_olm:1
α_h_olm = 0.07*exp(-(Volm/mV +65)/20):1
β_h_olm =1 / (1+ exp(-(Volm/mV+35)/10)):1
dn_olm/dt = α_n_olm*(1-  n_olm) - β_n_olm*n_olm:1
α_n_olm = 0.01*(Volm/mV + 55)/(1 - exp(-(Volm/mV +55)/10):1
β_n_olm = 0.125*exp(-(Volm/mV+65)/80):1

dm_po/dt = α_mpo*(1-  m_po) - β_mpo*m_po:1
α_mpo = 1/(0.15*(1 + exp(-(V/mV +38)/6.5)):1
β_mpo = exp(-(V/mV+38)/6.5)/(0.15*(1 + exp(-(V/mV +38)/6.5)):1

dλ_fo/dt = (λfinf*-λ_fo)/tau_λf:1
λfinf = 1/(1 + exp((Volm/mV +79.2)/9.78):1
tau_λf = 0.51/(exp((Volm/mV -1.7)/10)+ exp(-(Volm/mV +340)/52))):1
dλ_so/dt = (λsinf*-λ_so)/tau_λs:1
λsinf = 1/(1 + exp((Volm/mV +2.83)/15.9)**58:1                                                      #**58?
tau_λs = 5.6/(exp((Volm/mV -1.7)/14)+ exp(-(Volm/mV +260)/43)))+1:1
'''
eqsSynInput = '''
#ICa_NMDA = -g_syn * sNMDA * mCa_NMDA * (Vd - VCa_NMDA):ampere
#NMDA = -g_syn * sNMDA * mNMDA * (Vd - VNMDA) :ampere
#IAMPA = -g_syn * sAMPA * (Vd - VAMPA) :ampere
#IGABA = -g_syn * sGABA * (Vd - VGABA):ampere                                                  

mNMDA = 1/(1 + 0.3 * Mg2p * exp(-0.062 * Vd/mV)):1
mCa_NMDA = 1/(1 + 0.3 * Mg2p* exp(-0.124 * Vd/mV)):1

dsNMDArise/dt = 20 * (1-  sNMDAfast  sNMDAslow ) * Fpre - (1/2) * sNMDArise:1
dsNMDAfast/dt = 20 * (0.527 - sNMDAfast) * Fpre - (1/10) * sNMDAfast:1
dsNMDAslow/dt = 20 * (0.473 - sNMDAslow ) * Fpre -(1/45) * sNMDAslow:1

dsAMPArise/dt = 20 * (1 - sAMPAfast  sAMPAslow ) * Fpre - (1/0.58) * sAMPArise:1
dsAMPAfast/dt = 20 * (0.903 - sAMPAfast) * Fpre - (1/7.6) * sAMPAfast:1
dsAMPAslow/dt = 20 * (0.097 - sAMPAslow ) * Fpre - (1/25.69) * sAMPAslow:1

dsGABArise/dt = 20 * (1 - sGABAfast  sGABAslow ) * Fpre - (1/1.18) * sGABArise:1
dsGABAfast/dt = 20 * (0.803 - sGABAfast) * Fpre - (1/8.5) * sGABAfast:1
dsGABAslow/dt = 20 * (0.197 - sGABAslow ) * Fpre - (1/30.01) * sGABAslow:1

Fpre = H(t - 1)*(H(sin(2*pi * (t  -2)/Tper)) * (1-  H(sin(2pi * (t - 1)/Tper)))):1              

Ipc_syn_sum_ax = Isyn_aac2pc:ampere
Ipc_syn_sum_s = Isyn_bc2pc:ampere
Ipc_syn_sum_pd = Isyn_ca3ampa2pc_pd+Isyn_ca3nmda2pc_pd+ Isyn_bsc2pc_pd+Isyn_ivy2pc_pd:ampere
Ipc_syn_sum_dd = Isyn_ec_ampa2pc_dd+Isyn_ec_nmda2pc_dd+ Isyn_ngl2pc_dd+Isyn_olm2pc_dd:ampere

Isyn_sum_bc = Isyn_ec_ampa2bc+Isyn_ca3ampa2bc+Isyn_ms2bc+Isyn_bsc2bc:ampere
Isyn_sum_aac = Isyn_ec_ampa2aac+Isyn_ca3ampa2aac +Isyn_ms2aac:ampere
Isyn_sum_bsc = Isyn_ca3_ampa2bsc+Isyn_bc2bsc+Isyn_ms2bsc:ampere
Isyn_sum_ivy = Isyn_pc_ampa2ivy+Isyn_ms2ivy:ampere
Isyn_sum_ngl = Isyn_ec_ampa2ngl+Isyn_olm2ngl:ampere
Isyn_sum_olm =Isyn_pc_ampa2olm+Isyn_ms2olm:ampere

Isyn_aac2pc = waac_to_pc * DA * g_max* s * (Vaac +75*mV):ampere                

Isyn_bc2pc= wbc_to_pc * DA * g_max* s * (Vbc +75*mV):ampere

Isyn_ca3ampa2pc_pd= (w_s * wca3_to_pcAMPA + W_pd) * g_max* s * (V - 0*mV):ampere            #V it's Vca3, ec, ms, how is it defined?
Isyn_ca3ampa2pc_dd= (w_s * wca3_to_pcAMPA + W_dd) * g_max* s * (V - 0*mV):ampere            #w_s, wca3_to_pcAMPA - constants, 's' is changing always, 
Isyn_ca3ampa2pc_dd= (w_s * wca3_to_pcAMPA + W_dd) * g_max* s * (V - 0*mV):ampere            #depending on Fpre 
Isyn_ca3nmda2pc_pd= (w_s * wca3_to_pcNMDA + W_pd) * g_max* s * (V - 0*mV):ampere            #W_pd and W_dd are depending on Ca, so only W 
Isyn_ca3nmda2pc_dd= (w_s * wca3_to_pcNMDA + W_dd) * g_max* s * (V - 0*mV):ampere            #shoud be changed when spike arrives? 
Isyn_bsc2pc_pd= (w_s * wbsc_to_pc + W_pd) * g_max* s * (Vbsc +75*mV):ampere
Isyn_bsc2pc_dd= (w_s * wbsc_to_pc + W_dd) * g_max* s * (Vbsc +75*mV):ampere
Isyn_ivy2pc_pd= (w_s * wivy_to_pc + W_pd)* DA * g_max* s * (Vivy +75*mV):ampere
Isyn_ivy2pc_dd= (w_s * wivy_to_pc + W_dd)* DA * g_max* s * (Vivy +75*mV):ampere


Isyn_ec_ampa2pc_pd = (w_s * wec_to_pcAMPA + W_pd) * g_max* s * (V - 0*mV):ampere
Isyn_ec_ampa2pc_dd= (w_s * wec_to_pcAMPA + W_dd) * g_max* s * (V - 0*mV):ampere
Isyn_ec_nmda2pc_pd= (w_s * wec_to_pcNMDA + W_pd)* g_max* s * (V - 0*mV):ampere
Isyn_ec_nmda2pc_dd= (w_s * wec_to_pcNMDA + W_dd)* g_max* s * (V - 0*mV):ampere
Isyn_ngl2pc_pd= (w_s * wngl_to_pc + W_pd) * g_max* s * (Vngl +75*mV):ampere
Isyn_ngl2pc_dd= (w_s * wngl_to_pc + W_dd) * g_max* s * (Vngl +75*mV):ampere
Isyn_olm2pc_pd= (w_s * wolm_to_pc + W_pd) * g_max* s * (Volm +75*mV) :ampere
Isyn_olm2pc_dd= (w_s * wolm_to_pc + W_dd) * g_max* s * (Volm +75*mV) :ampere

Isyn_ec_ampa2bc= wec_to_bc * DA * g_max* s * (V - 0*mV) :ampere
Isyn_ca3ampa2bc= wca3_to_bc * DA * g_max* s * (V - 0*mV) :ampere
Isyn_ms2bc= wsep360_to_bc * DA * g_max* s * (V +75*mV) :ampere
Isyn_bsc2bc= wbsc_to_bc * DA * g_max* s * (Vbsc +75*mV) :ampere

Isyn_ec_ampa2aac= wec_to_aac * DA * g_max* s * (V - 0*mV) :ampere
Isyn_ca3ampa2aac= wca3_to_aac * DA * g_max* s * (V - 0*mV) :ampere
Isyn_ms2aac= wsep360_to_aac * DA * g_max* s * (V +75*mV) :ampere

Isyn_ca3_ampa2bsc= wca3_to_bsc * DA * g_max* s * (V - 0*mV) :ampere
Isyn_bc2bsc= wbc_to_bsc * DA * g_max* s * (Vbc +75*mV) :ampere
Isyn_ms2bsc= wsep180_to_bsc * DA * g_max* s * (V +75*mV) :ampere

Isyn_pc_ampa2ivy= wpc_to_ivy * DA * g_max* s * (V - 0*mV) :ampere
Isyn_ms2ivy= wsep_to_ivy * DA * g_max* s * (V +75*mV) :ampere

Isyn_ec_ampa2ngl= wec_to_ngl * DA * g_max* s * (V - 0*mV) :ampere
Isyn_olm2ngl= wolm_to_ngl * DA * g_max* s * (V +75*mV) :ampere

Isyn_pc_ampa2olm= wpc_to_olm * DA * g_max* s * (V - 0*mV) :ampere
Isyn_ms2olm= wsep180_to_olm * DA * g_max* s * (V +75*mV) :ampere




wec_to_pcAMPA=1.4 + W3 :1               #W1, W3? are not defined
wca3_to_pcAMPA =2.4 + W1 :1
wec_to_pcNMDA =1.4 + W3:1
wca3_to_pcNMDA =2.4 + W1 :1

#Isyn = g_syn * s * (V - Erev) :ampere
#g_syn = w * DA * g_max :
#g_syn = (w_s * w + W)* g_max: siemens/metre**2
#ds/dt = α *F*(1 - s) - β * s :1
#F = 1/(1 + exp((-Vpreu+θ)/2)):1
'''



morpho = Soma(diameter=30*um)
morpho.axon = Cylinder(length=100*um, diameter=1*um, n=10)
morpho.proximal_dendrite = Cylinder(length=50*um, diameter=2*um, n=5)
morpho.distal_dendrite = Cylinder(length=50*um, diameter=2*um, n=5)

place_cell = SpatialNeuron(morphology=morpho, model=eqs, Cm=1*uF/cm**2, Ri=100*ohm*cm) #reset, threshold?

PC = NeuronGroup(4, model = eqs)

BC = NeuronGroup(1, model = eqsBC)

AAC = NeuronGroup(1, model = eqsAAC)

BSC = NeuronGroup(1, model = eqsBSC)

IVY = NeuronGroup(4, model = eqsIVY)

NGL = NeuronGroup(4, model = eqsNGL)

OLM = NeuronGroup(1, model = eqsOLM)

BP = Synapses(BC, PC, model = '''Isyn_bc2pc= wbc_to_pc * DA * g_max* s * (Vbc - Erev):ampere (event-driven)
''', on_pre='ge+=we')        #how to change I?
BP.Connect()
AP = Synapses(AAC, PC, model = '''Isyn_aac2pc= waac_to_pc * DA * g_max* s * (Vaac - Erev):ampere (event-driven)
''', on_pre='ge+=we')# which ion's Erev?
AP.Connect()
BSP = Synapses(BSC, PC, model = '''Isyn_bsc2pc= wbsc_to_pc * DA * g_max* s * (Vbsc - Erev):ampere (event-driven)
''', on_pre='ge+=we')
BSP.Connect()
IP = Synapses(IVY, PC, model = '''Isyn_ivy2pc= wivy_to_pc * DA * g_max* s * (Vivy - Erev):ampere (event-driven)
''')
IP.Connect()
NP = Synapses(NGL, PC, model = '''Isyn_ngl2pc= wngl_to_pc * DA * g_max* s * (Vngl - Erev):ampere (event-driven)
''', on_pre='ge+=we')
NP.Connect()
OP = Synapses(OLM, PC, model = '''Isyn_olm2pc= wolm_to_pc * DA * g_max* s * (Volm - Erev):ampere (event-driven)
''', on_pre='ge+=we')
OP.Connect()
BSB = Synapses(BSC, BC, model = '''Isyn_bsc2bc= wbsc_to_bc * DA * g_max* s * (Vbsc - Erev):ampere (event-driven)
''', on_pre='ge+=we')
BSB.Connect()
BBS = Synapses(BC, BSC, model = '''Isyn_bc2bsc= wbc_to_bsc * DA * g_max* s * (Vbc - Erev):ampere (event-driven)
''', on_pre='ge+=we')
BBS.Connect()
ON = Synapses(OLM, NGL, model = '''Isyn_olm2ngl= wolm_to_ngl * DA * g_max* s * (Vngl - Erev):ampere (event-driven)
''', on_pre='ge+=we')
ON.Connect()
PO = Synapses(PC, OLM, model = '''Isyn_pc2olm= wpc_to_olm * DA * g_max* s * (Vax - Erev):ampere (event-driven)
''', on_pre='ge+=we')
PO.Connect()

StMs = StateMonitor(PC, 'Vs', record=True)
StMax = StateMonitor(PC, 'Vax', record=True)

SpM = SpikeMonitor(PC)

run(9000*ms, report='text')

# Graphical output
figure(figsize=(14, 4))
axhline(VL / mV, ls='-', c='lightgray', lw=3) # horisontal line at the level VL_pr

plot(StMs.t / ms, StMs.Vs_pr.T / mV)
xlabel('Time (ms)')
ylabel('Vs (mV)');


figure(figsize=(14, 4))
plot(SpM.t / ms, SpM.i, '.r')
xlabel('Time (ms)')
ylabel('Neuron index')



show()