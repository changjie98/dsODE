import SFP
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import welch

# Parameters of the integration
dt = 0.01 / 1000  # [s]  Integration time step
T = 2  # [s]  Simulation life time
N_V = 1000  # Numbert of grid point in potential discretization


Vt,Vr,Vmin=20.,0.,-3*20.                 #[mV] Trheshold,Reset and minimum potential
TauV=0.02                                #[s]  Membrane time constant
TauD=0.001                               #[s]  Characteristic time of exponential delay distribution
delta=0.005                            #[s]  Constant axonal delay
tref=4/1000                                   #[s]  Absolute refracory time

Mu0,Sig0=22,2.6652                       #[mV] Mean and variace of synaptic current multiplied by TauV and sqrt(TauV)
R0=20                                    #[Hz] Stationary firing rate
N=1000                                  #     Number of neurons, Possible choices at the moment N:1000,10000,100000
K=500                                     #     Mean synaptic contacts per neuron
J=12/K                                   #[mV] Strenght of synaptic coupling


# Initialize routine for finite size noise given the number of neuron
IntegrateEta = SFP.InitializeEta(N)

# Store parameters in dictionary
Net = {'dt': dt, 'Vt': Vt, 'Vr': Vr, 'Vmin': Vmin, 'N_V': N_V, 'K': K, 'J': J,
       'Mu0': Mu0, 'R0': R0, 'Sig0': Sig0, 'TauV': TauV, 'IExt': 0.0, 'delta': delta, 'tref': tref,
       'Life': T, 'N': N, 'TauD': TauD, 'delay_type': 3, 'gC': 0.0, 'AlphaC': 0.0, "TauC": 1.0}

# Readjust the external current to have the stationary firing rate at R0
Net['MuExt'], Net['SigExt'] = SFP.ExternalCurrent(Net)

# Initialize the grid for the Fokker-Planck
grid = SFP.Grid(V_0=Net['Vmin'], V_1=Net['Vt'], V_r=Net['Vr'], N_V=Net['N_V'])


def Simulate(Net):
    T, dt = Net["Life"], Net["dt"]
    tref, delta = Net["tref"], Net["delta"]
    TauV, TauD = Net["TauV"], Net["TauD"]

    Steps = int(T / dt)
    rN = np.zeros(Steps)
    r = np.zeros_like(rN)
    t = np.linspace(0, T, Steps)
    nd = int(delta / dt)
    n_ref = int(tref / dt)

    # Initialize probability distribution
    Net['fp_v_init'] = 'delta'
    p = SFP.initial_p_distribution(grid, Net)
    r_d = rN[nd]
    Eta, u1, u2 = 0, 0, 0
    ts = time.time()
    for n in range(nd + n_ref, Steps - 1):
        # Update the mean and variance of the synaptic current
        mu_tot = K * J * r_d + Net['MuExt']
        sigma_tot = np.sqrt(K * J ** 2 * r_d + Net['SigExt'] ** 2)
        int_ref = np.sum(r[n - n_ref:n] * dt)

        # Generate finite-soze noise via Markovian embedding
        Z = np.random.randn()
        Eta, u1, u2 = IntegrateEta(dt, Z, Eta, u1, u2, r[n], N, mu_tot * TauV, sigma_tot * TauV)
        InPut = {'r_d': r_d, 'rND': rN[n - nd], 'p': p, 'mu_tot': mu_tot, 'sigma_tot': sigma_tot, 'TauD': TauD,
                 'Eta': Eta, 'dt': dt, 'TauV': TauV, 'int_ref': int_ref, 'rNref': rN[n - n_ref]}

        # Integrate the F.P equation for one step
        r_d, rN[n + 1], r[n + 1], p = SFP.IntegrateFP(InPut, grid)
    te = time.time()
    print('Integration done in  %3d s' % (te - ts))
    return t, rN


t, rN = Simulate(Net)
'''
fs = 1 / Net["dt"]
Ind = np.abs(t - 1).argmin()
nFreq = 90000
f, Pxx_den = welch(rN[Ind:], fs, nperseg=nFreq, return_onesided=True)
F = np.logspace(0, 3, 1000)
S = np.interp(F, f, Pxx_den / 2)
plt.loglog(F, S, "-k")
plt.xlabel("f [Hz]")
plt.ylabel(r"$S_{\nu N}$")
plt.xlim(2, 1000);
'''
# 平滑化滤波，窗口为10ms
window_size = int(10 / (dt * 1000))  # 10ms对应的样本数
rN_smooth = np.convolve(rN, np.ones(window_size)/window_size, mode='same')

# 去掉负值
rN_smooth[rN_smooth < 0] = 0
plt.plot(t, rN_smooth,label = 'Fokker-Planck model')
x1 = sum(rN_smooth)
print(sum(rN_smooth))
plt.xlabel("time [s]")
plt.ylabel("rN")
#plt.show()
























#Gain Function fast and robust by M. Mattia
import numpy as np
import mpmath as mp
DawsonF=lambda x: (mp.sqrt(mp.pi)/2)*mp.exp(-(x**2))*mp.erfi(x)

def G(w):
	if(w>=-3):
		return mp.exp(w**2)*(mp.erf(w)+1)
	else:
		f=lambda y: mp.exp(-y**2)*mp.exp(2*y*w)
		return (2/mp.sqrt(mp.pi))*mp.quad(f,[0,5])

MFPT1=lambda a,b: mp.sqrt(mp.pi)*mp.quad(G,[a,b])

def MFPT2(a,b):
	MFPTfrom0=lambda b: mp.sqrt(mp.pi)*DawsonF(b)*G(b) - 2*mp.quad(DawsonF,[0,b])
	MFPTto0=lambda a: -MFPTfrom0(a)
	return MFPTfrom0(b)+MFPTto0(a)

def MFPT3(a,b):
	f=lambda w : mp.exp(-(w**2))*(mp.exp(2*b*w)-mp.exp(2*a*w))/w
	return mp.quad(f,[0,mp.inf])

def MFPT(a,b):
	if(b<=0):
		return MFPT3(a,b)

	else:
		if(b>15):
			return mp.inf

		else:
			if(a<=0):
				return MFPT3(a,0)+MFPT2(0,b)

			else:
				return MFPT2(a,b)

            
def Phi(mu,sig,Tarp,TauV,Vr,Vt):
    xr=(Vr-mu)/sig
    xt=(Vt-mu)/sig
    return np.array((1/(Tarp+TauV*MFPT(xr,xt))),dtype=float)


import numpy as np
import matplotlib.pyplot as plt
import nest
import time
from scipy import stats
import time




def InitializeNet(nest,Net,Process,WN):

    # Network parameters
    
    Vthr     =   Net['Vt']             # Spike emission threshold (mV)
    Vres     =   Net['Vr']             # After-spike reset potential (mV)
    TauV     =   Net['TauV']*1000      # Membrane potential decay constant (ms)
    Tau0     =   Net['Tau0']*1000      # Refractory period (ms)
    Cm       =   500.                  # Membrane capacitance (pF)

    N        =   Net['N']  			   # Number of (recurrent) neurons in the network, it is an integer.
    
    Jrec  = Net['J']                   # Efficacy of the synapses from recurrent neurons (mV)
    DJ=Net['DJ']
    Krec  = Net['K']                   # Number of recurrent synapes received by a neuron

    # Simulation parameters
    dt           = Net['dt']*1000       # Integration time step (ms)
    Delay    =dt                        # Minimum spike transmission delay (ms)
    ThreadNum    = Process             # Number of simultaneous threads to launch (depends on the machine).
    TauD=Net['TauD']*1000              #[ms]
    
    Iext=0.0

    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": ThreadNum})
    nest.SetKernelStatus({"resolution": dt, "print_time": False})

    # Initialize the random number generators
    msd = int(np.fabs(time.process_time()*1000))
    N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
    nest.SetKernelStatus({"grng_seed" : msd+N_vp})
    nest.SetKernelStatus({"rng_seeds" : range(msd+N_vp+1, msd+2*N_vp+1)})
 
        
    # Network building: create the neurons composing the network
    neurons_exc = nest.Create("iaf_psc_delta", N, [{"V_m": Vres, "E_L": 0.0,
                                                    "C_m": Cm, "tau_m": TauV,
                                                    "V_th": Vthr, "V_reset": Vres,
                                                    "I_e": Iext, "t_ref": Tau0}])
    KExt=Net["KExt"]

    if WN:
        noise = nest.Create('noise_generator')
        mean = Cm*(Net["MuExt"]/1000.) # Infinitesimal mean (pA)
        std  = Cm *Net["SigExt"]*np.sqrt(1/(1000*dt)) # Infinitesimal SD/sqrt(dt) (pA)


        nest.SetStatus(noise, [{'mean': mean, 'std': std,
                                           'start': 0.,'dt':float(dt),'origin': 0.}])
        nest.Connect(noise, neurons_exc)
    else:
        KExt=Net["KExt"]
        JExt=Net["JExt"]
        NuExt=Net["NuExt"]

        noise=nest.Create("poisson_generator",
                    params={"rate": float(KExt*NuExt),'origin':0., 'start':0.});
        nest.Connect(noise, neurons_exc, syn_spec={'model': 'static_synapse',
                                                           'delay': dt,'weight': float(JExt)})



    # Spike detector to save simulated network activity
    spikes_detector_exc = nest.Create("spike_detector")
    nest.SetStatus(spikes_detector_exc,[{"label": 'Spikes',
                                         "withtime": True, "withgid":   False,
                                         "to_file":  False, "to_memory": True,
                                         "close_after_simulate": False}])
    conn_dict = {"rule": "all_to_all"}
    syn_dict = {"model": "static_synapse"}
    nest.Connect(neurons_exc, spikes_detector_exc, conn_dict, syn_dict)


    # Connecting recurrent neurons
    DelayDetails={"distribution": "exponential_clipped", "lambda": (1/TauD),"low":dt, "high": float('inf')}

    conn_dict_exc = {"rule": 'fixed_indegree', 'indegree': int(Krec)}
    #syn_dict_exc = {"model": "static_synapse", "weight": Jrec,"delay":DelayDetails}
    WeightsDict= {"distribution": "normal","mu": Jrec, "sigma": Jrec*DJ}

    syn_dict_exc = {"model": "static_synapse", "weight":WeightsDict,"delay":DelayDetails}
    if Krec>0:
        nest.Connect(neurons_exc, neurons_exc, conn_dict_exc, syn_dict_exc)


    return spikes_detector_exc,neurons_exc,noise

def Rate(spikes_detector_exc,Life,DeltaT,N):
    Spikes=nest.GetStatus(spikes_detector_exc)[0]["events"]["times"]
    xbins=np.arange(0,Life,DeltaT)
    counts,edges=np.histogram(Spikes,xbins)
    rate=counts/(N*DeltaT*0.001) #[Hz]
    t= edges[:-1]
    return t,rate


## Network parameters
dt=0.01/1000 #[s]
K=500 #Average number of synaptic contacts
N=1000 # Number of neurons
J=12/K # Average synaptic strenght
θ,H=20.0,0.0
ntrheads=1 # !!! Set here your number of cpu core!!


τ,τ0 = 0.02,dt #[s]
τ0 = dt
τD=0.005 #[s] # mean axonal delay

#External current poisson process::
μ0=Mu0/τ
σ0=2.6652/np.sqrt(τ)
ν0=Phi(μ0*τ,σ0*np.sqrt(τ),τ0,τ,H,θ)#[Hz]
print("Stationary firing rate: "+str(ν0))

#Change external current to keep stationary firing rate at ν0
μExt=μ0-K*J*ν0
σExt=np.sqrt(σ0**2 -K*(J**2)*ν0)
CExt=5000
JExt=(σExt**2)/μExt
νExt=μExt/(CExt*JExt)

Net={"Vt":θ,"Vr":H,"K":K,"N":N,"J":J,"DJ":0.,"MuExt":μExt,"SigExt":σExt,"KExt":CExt,"JExt":JExt,"NuExt":νExt,
      "TauV":τ,"Tau0":τ0,"TauD":τD,"dt":dt}

ts=time.time()
SD,NP,IX=InitializeNet(nest,Net,ntrheads,True)
Life=2000 #[ms]
Δt=4 #[ms]
# Simulate
nest.Simulate(Life)
print(str(time.time() -ts))

t,ν=Rate(SD,Life,Δt,Net["N"])
print(x1/1000)
print(sum(ν)*4)

plt.plot(t/1000,ν*0.4,"-k",label = 'LIF model')
print(t.size)
x1 = x1/1000
x2 = sum(ν)*0.4
print(x1/10)
print(x2/10)
plt.legend(prop = {'size':20})
plt.title(f"firing rate error {round((x1-x2)/x2,4)*100}%")
plt.xlabel("t [s]")
plt.ylabel("νHz]")
plt.show()




