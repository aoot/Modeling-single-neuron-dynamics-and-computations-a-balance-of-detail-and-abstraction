#/usr/bin python
import sys
import os
import nest
import nest.topology as topo
import numpy
import pickle
import time
import stim
import fncts
import argparse
import random
from mpi4py import MPI

#print "NEST VERSION", nest.version()

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#--------------------------------- SIMULATION PARAMETERS -----------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('-t','--timestamp')
args=parser.parse_args()

sim = dict(
fname      = "",
N_vp 	=  0,		#total number of virtual processes is set automatically
res	=    0.1, 	#time resolution of simulation
tinterval    = 10000.0,	#recording interval for synaptic weights in ms **10000
runs 	=  689, #689 for 3, 424 full 2 runs 262 for run with sp. 162 = 27mins +1 SP 100,# runs*tinterval=tsim  (for runs=120 --> total time=20min) **12
n_stn = 1000, #number of stn neurons 1000 is best for testing. Equal for both. Had 100
n_gpe = 1000, #number of gpe neurons

res_glo_sync =   10,  #resolution of global sync-msr in ms 
rec_ls = False,  #record voxel based local sync (large sim n_stn >1000) 
                #or pairwise sync (small sim n_stn <=1000)
res_loc_sync_v = 100, #resolution of voxel based local sync in ms
res_loc_sync =   10,  #number of datapoints per tinterval for single phase recording

rec_vm = False, #record mean membrane potential
res_vm =   10., # resoultion of mean membrane potential recording

rec_isi = False, #record interspike intervals

#Seed Python and NEST random number generators
msd = 723545987, #789545327,#24624636736,#789545327,#, #master seed
#msd = 837939, #seed 1
#msd = 5006947, #seed 2
#msd = 6656469, #seed 3
#msd = 7107975, #seed 4
#msd = 3557262, #seed 5
#msd = 7895364, #seed 6
#msd = 503780, #seed 7
#msd = 5325028, #seed 8
#msd = 9302260, #seed 9
#msd = 1802117, #seed 10
#msd = 1767643, #seed 11
#msd = 5100918, #seed 12
#msd = 4229506, #seed 13
#msd = 5142601, #seed 14
#msd = 4256732, #seed 15
#msd = 9351747, #seed 16
#msd = 9979664, #seed 17
#msd = 8536993, #seed 18
#msd = 3066907, #seed 19
#msd = 1474920, #seed 20


#-------------------------------------------------------------------------------------------#
#-------------------------------- CR STIMULATION -------------------------------------------#
#-------------------------------------------------------------------------------------------#

# GENERAL FIXIED STIMULATION PARAMETERS (we could potential use different for CR or PS)
stim_amp   = -3.3,#-2.0,#-3.3, 		#(mA) stim amplitude
stim_width = 200.0,		#(us) pulse width
stim_ps    = 8,			#integer Pulse Shape reverse order: <0 
stim_gap   = 0.0, 		#(ms) gap between pulse anodic and cathodic pulse
stim_ibf   = 130.0,		#(Hz) intra burst frequency --> Tp=1000*(1/130)=7.69ms
stim_npulses = 4,		#integer number of pulses per burst
stim_fburst  = 8.0,		#(Hz) burst frequency --> fburst=1000*(1/T), T=125ms here -- 3ON+2OFF=600ms 
###

stim_cr_on= True,  # ** switch stimulation on/off 
stim_cur= True,    # ** current controlled stim
stim_cr_simple= False,

#--> usage: per_sequence(start,stop,amp,width,ps,gap,ibf,npulses,fburst,cy_on,cy_off) BUT is the same practilly function as for CR!!!!
stim_periodic = True, # ** 25.04.2017 - PERIODIC STIMULATION (or PS)
stim_start_per = 120000.0,	#(ms) start and stop time of stimulation for Periodic Stimulation **
stim_stop_per  = 270000.0,	#(ms) **
stim_cy_on_per   = 5,	#integer cycles on for Periodic Stimulation
stim_cy_off_per  = 0,	#integer cycles off for Periodic Stimulation

#--> usage: cr_sequence(start,stop,amp,width,ps,gap,ibf,npulses,fburst,cy_on,cy_off) --> This is the standard function actually
stim_cr_rand = True, # ** 25.04.2017 - Coordinated Recet Stimulation  (or CR)
stim_start = 420000.0,	#(ms) start and stop time of stimulation **
stim_stop  = 1020000.0,	#(ms) **
stim_cy_on   = 3,		#integer cycles on
stim_cy_off  = 2,		#integer cycles off

#distance dependence of current controlled stim
stim_c_cur   = 0.02,
stim_a_cur   = 1.8,
stim_tau_cur = 0.7,
stim_p_center_cur=1.0,
stim_sigma_cur=1.5, 	#electrode contact length in mm
stim_delay  = 4.0,

#mask radius
stim_radius_cur = 2.0,
stim_radius_syn = 1.5,

#simulation times for structural plasticity
sp_enabled = False, #Disables structural plasticity in the simulations
sim_structural_plasticity = 1000000,#1000000, #**
start_sp = 163, #163
stop_sp = 263, #263
end_epoch = 2630000,#2630000,
#Uncomment if stimulation times>2
start_sp_2 = 426, 
stop_sp_2= 526,
end_epoch_2=5260000,
stimulation_times = 3,
short_follow_iterations =True,

#electrode coordinates
pos_electrode = [[0.,-3.,0.],[0.,-1.,0.],[0.,1.,0.],[0.,3.,0.]],
#[[0.,-1.,0.]],# [[0.,3.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,-3.,0.]],

#-------------------------------------------------------------------------------------------#
#----------------------------- CONNECTION PARAMETERS ---------------------------------------#
#-------------------------------------------------------------------------------------------#

#initial synaptic weights
#gaussian distributed synaptic weights
#w_ss_ij = offset_g_ss +- sigma_g_ss
#mean:offset_g_ss, standarddeviation: sigma_g_ss

gaussian_weights = True, #gaussian distributed initial weights

#decline of connection strength with increasing distance
#w_ss_ij = offset_g_ss + g_ss*exp(-distance_ij/dec_g_ss)
#mean values for connection strength
offset_g_ss =  0.002, #0.0025 #0.018
offset_g_sg =  0.006,#0.005 last, 0.006 orig,#0.003,#0.002,
offset_g_gs = -0.003,#-0.002 last, -0.003 orig,#-0.001,
offset_g_gg = -0.0025,#-0.0025,#-0.0025,

#standard deviations for connection strength
sigma_g_ss = 0.000125, #0.001 5% of offset
sigma_g_sg = 0.0003, #0.003
sigma_g_gs = 0.00015,
sigma_g_gg = 0.000125,

g_ss =  0.0,
g_sg =  0.0,
g_gs =  0.0,	
g_gg =  0.0, 

dec_g_ss =  1.25,
dec_g_sg = 12.0,
dec_g_gs = 12.0,
dec_g_gg =  2.5,

#connectivity values: Probabilities of connection
con_ss = 0.07, #Gillies and Willshaw, Proc. R. Soc. Lond. B. 265:2101-2109 (1998) 
con_sg = 0.02, #Baufreton et al, J Neurophysiol 102(1):532-545 (2009)
con_gs = 0.02, #Baufreton et al, J Neurophysiol 102(1):532-545 (2009)
con_gg = 0.01, #Sadek et al, J Neurosci 27(24):6352-6362 (2007)

#connections
#decline of connection probability with increasing distance
#wss_ij = offset_c_ss + c_ss*exp(-distance_ij/dec_c_ss)
dist_con_prob = True,

offset_c_ss = 0.0,
offset_c_sg = 1.0,
offset_c_gs = 1.0,
offset_c_gg = 0.0,

c_ss = 1.0,
c_sg = 0.0,
c_gs = 0.0,
c_gg = 1.0,

dec_c_ss =  0.5, #Kita et al. J Compa Neur 25:245-257 (1983), Logbook p. 167
dec_c_sg = 12.0,
dec_c_gs = 12.0,
dec_c_gg =  0.63, #Sadek et al, J Neurosci 27(24):6352-6362 (2007), Logbook p.167

#connection delays 
#Holgado et al, J Neurosci 30(37):12340-12352 (2010)
delay_ss = 4.0,
delay_sg = 6.0, # inconsistent with Paper & thesis the value in code I have it is 6.0 while in the paer 4.
delay_gs = 6.0, # same here
delay_gg = 4.0,

#-------------------------------------------------------------------------------------------#
#--------------------------------- STDP PARAMETERS -----------------------------------------#
#-------------------------------------------------------------------------------------------#

stdp_on = True, #switch STDP on/off
# This is True for the populations where STDP will be on during the simulation
stdp_ss = True,
stdp_sg = False,
stdp_gs = False,
stdp_gg = False,
#Potentiation
#delta_w_p=(w/Wmax)+(lambda*(1.0-(w/Wmax)**mu_plus)*kplus*exp(dt/tau_plus));
#return delta_w_p < 1.0 ? delta_w_p * Wmax: Wmax;
#Depression
#delta_w_d=(w/Wmax)-(alpha*lambda*(w/Wmax)**mu_minus)*kminus*exp(dt/tau_minus));
#return delta_w_d > 0.0 ? delta_w_d * Wmax : 0.0;
#Implementation: cf. nestkernel/archiving_node.h, models/stdp_connection.h
#theoretical background: cf. Morrison et al, Neural Computation 19:1437-1467 (2007)
#                            Morrison et al, Biol Cybern 98(6):459-478 (2008)

#dep time const are set in postsyn. neurons
stdp_tau_minus_stn = 27.5,#27.5,  
stdp_tau_minus_gpe = 22.5,

stdp_Wmax_ss     = 0.02, #0.017max syn weight. Important to keep the balance.
stdp_tau_plus_ss = 12.0, #12.0pot time const can be set for each synapse
stdp_alpha_ss 	 = 1.4, #1.4 #1.1 last,  #1.25b 1.1 from 2.0, 16 Feb. 2017 Thanos to get the closest results. Critical for stable steup.	
stdp_lambda_ss   = 0.002, #0.002
stdp_mu_plus_ss  = 0.0,
stdp_mu_minus_ss = 0.0,

stdp_Wmax_sg     = 0.0021,#0.0015, #max syn weight
stdp_tau_plus_sg = 10.0, #pot time const can be set for each synapse
stdp_alpha_sg 	 = 0.64,
stdp_lambda_sg   = 0.001,
stdp_mu_plus_sg  = 0.0,
stdp_mu_minus_sg = 0.0,

stdp_Wmax_gs     = -0.0011, #max syn weight
stdp_tau_plus_gs = 18.0, #pot time const can be set for each synapse
stdp_alpha_gs 	 = 0.64,
stdp_lambda_gs   = 0.001,
stdp_mu_plus_gs  = 0.0,
stdp_mu_minus_gs = 0.0,

stdp_Wmax_gg     = -0.0026, #max syn weight
stdp_tau_plus_gg = 18.0, #pot time const can be set for each synapse
stdp_alpha_gg 	 = 0.64,
stdp_lambda_gg   = 0.001,
stdp_mu_plus_gg  = 0.0,
stdp_mu_minus_gg = 0.0,


#-------------------------------------------------------------------------------------------#
#------------------------------- EXTERNAL INPUT --------------------------------------------#
#-------------------------------------------------------------------------------------------#
#noise_rate (Hz)
noise_rate_stn 	 =   20.0,
noise_weight_stn =   0.2,
noise_delay_stn  =   4.0,
noise_rate_gpe 	 =   40.0,
noise_weight_gpe =   0.2,
noise_delay_gpe  =   4.0,

#amplitudes of constant input to STN and GPe
I_e_stn =  0.0,
I_e_gpe = -7.0,

#-------------------------------------------------------------------------------------------#
#-------------------------- SINGLE NEURON PARAMETERS ---------------------------------------#
#-------------------------------------------------------------------------------------------#

#min & max for initial membrane potentials (uniform dist)
vm_min_stn = -100.0,
vm_max_stn =  -20.0,
vm_min_gpe = -100.0,
vm_max_gpe =  -20.0,

#equilibrium potentials and conductances for ion channels
#gaussian dist around mean, std=10%
#standard deviation percentage e.g. sigma_E_K_stn = E_K_stn * pinit_std
pinit_std = 0.05,

#STN
E_K_stn         = -80.0,
E_Na_stn        =  55.0,
E_Ca_stn        = 140.0,
E_L_stn         = -60.0,
sigma_E_K_stn   =   8.0,
sigma_E_Na_stn  =   5.5,
sigma_E_Ca_stn  =  14.0,
sigma_E_L_stn   =   0.1, # 6.0 instead of 0.1 --> Thanos 15.02.2017

g_K_stn         = 45.0,
g_Na_stn        = 37.5,
g_Ca_stn        =  0.5,
g_T_stn         =  0.5,
g_ahp_stn       =  9.0,
g_L_stn         =  2.25,
sigma_g_K_stn   =  4.5,
sigma_g_Na_stn  =  3.75,
sigma_g_Ca_stn  =  0.05,
sigma_g_T_stn   =  0.05,
sigma_g_ahp_stn =  0.9,
sigma_g_L_stn   =  0.225,

#GPe
E_K_gpe         = -80.0,
E_Na_gpe        =  55.0,
E_Ca_gpe        = 120.0,
E_L_gpe         = -55.0,
sigma_E_K_gpe   =   8.0,
sigma_E_Na_gpe  =   5.5,
sigma_E_Ca_gpe  =  12.0,
sigma_E_L_gpe   =   0.1, # 5.5 instead 0.1, Thanos 15.02.2017

g_K_gpe         =  30.0,
g_Na_gpe        = 120.0,
g_Ca_gpe        =   0.15,
g_T_gpe         =   0.5,
g_ahp_gpe       =  30.0,
g_L_gpe         =   0.1,
sigma_g_K_gpe   =   3.0,
sigma_g_Na_gpe  =  12.0,
sigma_g_Ca_gpe  =   0.015,
sigma_g_T_gpe   =   0.05,
sigma_g_ahp_gpe =   3.0,
sigma_g_L_gpe   =   0.01,

#Parameters for synaptic currents
#reciprocal time constants
beta_ex =    1.0,
beta_in =    0.3,
#reversal potentials for inhib currents
E_gg    =  -80.0, 
E_gs    = -100.0,

#refractory periods
ref_gpe = 3.0,
ref_stn = 3.0,


#-------------------------------------------------------------------------------------------#
#--------------------------------- 3D TOPOLOGY ---------------------------------------------#
#-------------------------------------------------------------------------------------------#

#STN / GPe Topology connection masks
tp_ctr_stn = numpy.array([0.0, 0.0, 0.0]),
tp_ext_stn = numpy.array([30.0,30.0,30.0]),
tp_ctr_gpe = numpy.array([12.0, -4.7, 3.0]),
tp_ext_gpe = numpy.array([50.0, 50.0, 50.0]),
tp_mask_all_ctr = numpy.array([8.0,-3.7,3.0]),
tp_mask_all_ll	 = numpy.array([-50.0,-50.0,-20.0]),
tp_mask_all_ur  = numpy.array([50.0,50.0,20.0]),
tp_mask_ls_ll_1mm = numpy.array([-.5,-.5,-.5]),
tp_mask_ls_ur_1mm = numpy.array([.5,.5,0.5]),
#-------------------------------------------------------------------------------------------#
#----------------------------------- SIM CHAR ----------------------------------------------#
#-------------------------------------------------------------------------------------------#

#init values for analysis parameters (change after simulation)
#mean firing rates
rate_stn   = 0.0,
rate_gpe   = 0.0,

#initial mean syn weights
in_synw_ss = 0.0,
in_synw_sg = 0.0,
in_synw_gs = 0.0,
in_synw_gg = 0.0,

t_sim	 = 0.0, #simulation time (s)
t_node   = 0.0, #node creation time (s)
t_wire   = 0.0, #wiring time (s)
t_init   = 0.0, #init time (s)
t_comp   = 0.0, #runtime for simulation (s)
t_realtf = 0.0, #realtime factor
t_gather = 0.0, #data gathering from mpi procs (s)
t_data   = 0.0, #data processing at sim end (s)

#standard deviations
t_node_sd   = 0.0,
t_wire_sd   = 0.0,
t_init_sd   = 0.0,
t_comp_sd   = 0.0,
t_gather_sd = 0.0,

#synchronistaion measure
t_sync = 0.0,
sync_r1_stn = 0.0,
sync_r2_stn = 0.0,
sync_r3_stn = 0.0,
sync_r4_stn = 0.0,
sync_r1_gpe = 0.0,
sync_r2_gpe = 0.0,
sync_r3_gpe = 0.0,
sync_r4_gpe = 0.0,

data_path="/p/scratch/cslns/slns009/datajuwels/"+args.timestamp+"/",
par_path="par/",
topo_path="/p/scratch/cslns/slns009/datajuwels/topo/",
ls_path="loc_sync/"
) # end of sim parameter dictionary


#-------------------------------------------------------------------------------------------#
#-------------------- STN - Excitatory synaptic elements of excitatory neurons -------------#
#-------------------------------------------------------------------------------------------#
# Defining the update rate in strcuture for the whole network. In simulation steps.
nest.SetStructuralPlasticityStatus({'structural_plasticity_update_interval': 1000,})

# We will us the standard values for Tau and Beta constants of the sp framework.
# 4weeks = 2419200000 ms. We want this compressed into ~2 hours of simulation. Make x25 smaller the growth rate
# Excitatory synaptic elements of excitatory neurons
growth_curve_e_e = {'growth_curve': "gaussian",
            'growth_rate': 0.0000, # (elements/ms)
            'continuous': False,
            'eta': -0.005,# Ca2+
            'eps': 0.03, # Ca2+
        }
growth_curve_e_i = {'growth_curve': "gaussian",
            'growth_rate': 0.0000, # # (elements/ms)
            'continuous': False,
            'eta': -0.005,# Ca2+
            'eps': 0.03, # Ca2+
        }

growth_curve_i_e = {'growth_curve': "gaussian",
            'growth_rate': 0.0000, # (elements/ms)
            'continuous': False,
            'eta': -0.005,# Ca2+
            'eps': 0.075, # Ca2+
        }

growth_curve_i_i = {'growth_curve': "gaussian",
            'growth_rate': 0.0000, # (elements/ms)
            'continuous': False,
            'eta': -0.005,# Ca2+
            'eps': 0.075, # Ca2+
        }

synaptic_elements_e = {'Den_ex': growth_curve_e_e,'Axon_ex': growth_curve_e_e, 'Axon_in': growth_curve_e_i,}
synaptic_elements_i = {'Den_in': growth_curve_i_e, 'Den_inin': growth_curve_i_i,'Axon_inin': growth_curve_i_i}


#set connectivity values dependent on number of neurons
if sim["n_stn"] <= 1000:
	sim["con_ss"] = 0.7 
	sim["con_sg"] = 0.2 #0.2 
if sim["n_gpe"] <= 1000:
	sim["con_gs"] = 0.2 
	sim["con_gg"] = 0.1 #0.1 

#set standard deviation for initial conditions
sim["sigma_E_K_stn"]    =  abs(sim["E_K_stn"] * sim["pinit_std"])
sim["sigma_E_Na_stn"]   =  abs(sim["E_Na_stn"] * sim["pinit_std"])
sim["sigma_E_Ca_stn"]   =  abs(sim["E_Ca_stn"] * sim["pinit_std"])

sim["sigma_g_K_stn"]    =  abs(sim["g_K_stn"] * sim["pinit_std"])
sim["sigma_g_Na_stn"]   =  abs(sim["g_Na_stn"] * sim["pinit_std"])
sim["sigma_g_Ca_stn"]   =  abs(sim["g_Ca_stn"] * sim["pinit_std"])
sim["sigma_g_T_stn"]    =  abs(sim["g_T_stn"] * sim["pinit_std"])
sim["sigma_g_ahp_stn"]  =  abs(sim["g_ahp_stn"] * sim["pinit_std"])
sim["sigma_g_L_stn"]    =  abs(sim["g_L_stn"] * sim["pinit_std"])

sim["sigma_E_K_gpe"]    =  abs(sim["sigma_E_K_gpe"] * sim["pinit_std"])
sim["sigma_E_Na_gpe"]   =  abs(sim["sigma_E_Na_gpe"] * sim["pinit_std"])
sim["sigma_E_Ca_gpe"]   =  abs(sim["sigma_E_Ca_gpe"] * sim["pinit_std"])

sim["sigma_g_K_gpe"]    =  abs(sim["sigma_E_K_gpe"] * sim["pinit_std"])
sim["sigma_g_Na_gpe"]   =  abs(sim["sigma_g_Na_gpe"] * sim["pinit_std"])
sim["sigma_g_Ca_gpe"]   =  abs(sim["sigma_g_Ca_gpe"] * sim["pinit_std"])
sim["sigma_g_T_gpe"]    =  abs(sim["sigma_g_T_gpe"] * sim["pinit_std"])
sim["sigma_g_ahp_gpe"]  =  abs(sim["sigma_g_ahp_gpe"] * sim["pinit_std"])
sim["sigma_g_L_gpe"]    =  abs(sim["sigma_g_L_gpe"] * sim["pinit_std"])
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------- NEST SETUP -----------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

nest.ResetKernel()#
#set nest output to warnings/errors only
if sim["rec_vm"]:
	nest.sli_run("M_ERROR setverbosity")
else:
	nest.sli_run("M_WARNING setverbosity")
#import Terman-Rubin Model neuron into nest
#nest.Install("STNGPeGPi")

#parser = argparse.ArgumentParser()
#parser.add_argument('-t','--timestamp')
#args=parser.parse_args()
sim["fname"] = args.timestamp

#setup MPI communication
comm = MPI.COMM_WORLD
sim["N_vp"] = comm.Get_size()
rank = nest.Rank()

format_string='{0:0'+str(len(str(sim["N_vp"])))+'}'

#print simulation start and estimated end
if os.path.isdir(sim["data_path"])==False:
	os.system("mkdir "+sim["data_path"])
if os.path.isdir(sim["topo_path"])==False:
	os.system("mkdir "+sim["topo_path"])
if os.path.isdir(sim["ls_path"])==False:
	os.system("mkdir "+sim["ls_path"])

if rank == 0:
	print "simulation running ..."
	print "  --> start: ", time.strftime("%H:%M")
	if sim["N_vp"] == 24:
		rtf = 7.0
		build = 80.0
		nodec = 0.2
	elif sim["N_vp"] >1000:
		rtf = 3.0
		build = 120.0
		nodec = 0.2
	#print "  --> estimated runtime (s): ",(sim["tinterval"]*sim["runs"]/1000.*rtf+nodec+3.+build)
#record global synchronization
	f_glo_sync_name = sim["data_path"]+sim["fname"]+"-glo_sync.txt"
	f_glo_sync = open(f_glo_sync_name,"w")
#record mean membrane potential
	if sim["rec_vm"]:
		f_vm_name = sim["data_path"]+sim["fname"]+"-vm.txt"
		f_vm = open(f_vm_name,"w")
#record local voxel based synchronization
f_loc_sync=[]
if sim["rec_ls"]:
	if rank == 0 and sim["n_stn"]>1000:
		f_ls_name = sim["data_path"]+sim["fname"]+"-loc_sync-vox"
		f_ls_r1_1mm = open(f_ls_name+"_r1_1mm.txt","w")
		f_ls_r2_1mm = open(f_ls_name+"_r2_1mm.txt","w")
		f_ls_r3_1mm = open(f_ls_name+"_r3_1mm.txt","w")
		f_ls_r4_1mm = open(f_ls_name+"_r4_1mm.txt","w")
		f_ls_p_1mm = open(f_ls_name+"_p_1mm.txt","w") #record phases

	if sim["n_stn"]<=1000:
#or pairwise synchronization
		part="stn"
		for l in range(1,sim["runs"]+1):
			timestep=str(int(l*sim["tinterval"]))
			fls=open(sim["ls_path"]+"local_sync-"+part+"_t="+timestep+"-"+format_string.format(rank)+".dat","w")
			f_loc_sync.append(fls)



#load 3D neuron coordinates
if sim["stim_cr_on"]:
	pos_stn = fncts.load_coord(sim["par_path"]+"bachus_le_stn_neurons_"+str(sim["n_stn"]/1000)+"k.txt",sim["n_stn"])
else:
	pos_stn = fncts.load_coord(sim["par_path"]+"bachus_le_stn_neurons_wo_el_"+str(sim["n_stn"]/1000)+"k.txt",sim["n_stn"])

pos_gpe = fncts.load_coord(sim["par_path"]+"bachus_le_gpe_neurons_"+str(sim["n_gpe"]/1000)+"k.txt",sim["n_gpe"])

#record interspike intervals
if sim["rec_isi"]:
	f_isi_stn_name = sim["data_path"]+"isi-stn-"+format_string.format(rank)+".dat"
	f_isi_stn = open(f_isi_stn_name,"w")
	f_isi_gpe_name = sim["data_path"]+"isi-gpe-"+format_string.format(rank)+".dat"
	f_isi_gpe = open(f_isi_gpe_name,"w")

#init random number generators
pyrngs = [numpy.random.RandomState(s) for s in range(sim["msd"],sim["msd"]+sim["N_vp"])]

#set simulator properties
nest.SetKernelStatus({'print_time': False,
		      'total_num_virtual_procs': sim["N_vp"],
		      #'local_num_threads': 4,
			  'resolution': sim["res"],
			  'grng_seed': sim["msd"]+sim["N_vp"],
			  'rng_seeds': range(sim["msd"]+sim["N_vp"]+1,sim["msd"]+2*sim["N_vp"]+1),
			  'overwrite_files': True,
		      'data_path': sim["data_path"]})  # 'print_time': False


#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#--------------------------------- NODE CREATION -------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

node_creation_start = time.time()

#-------------------------------------------------------------------------------------------#
#-------------------------------- CREATE NEURONS -------------------------------------------#
#-------------------------------------------------------------------------------------------#

nest.SetDefaults("terub_neuron_stn",
		{"t_ref":sim["ref_stn"],
		"I_e":sim["I_e_stn"],
		 "tau_syn_in": 1.0/sim["beta_in"],
		 "tau_syn_ex":1.0/sim["beta_ex"],
		 "E_gs":sim["E_gs"],
		 "tau_minus":sim["stdp_tau_minus_stn"]})

nest.SetDefaults("terub_neuron_gpe",
		{"t_ref": sim["ref_gpe"],
		 "I_e": sim["I_e_gpe"],
		 "tau_syn_in": 1.0/sim["beta_in"],
		 "tau_syn_ex": 1.0/sim["beta_ex"],
		 "E_gg": sim["E_gg"],
		 "tau_minus":sim["stdp_tau_minus_gpe"]})

stn = topo.CreateLayer({"extent": sim["tp_ext_stn"],
			"center": sim["tp_ctr_stn"],
			"positions": pos_stn,
			"edge_wrap": False,
			"elements": "terub_neuron_stn"})

gpe = topo.CreateLayer({"extent": sim["tp_ext_gpe"],
			"center": sim["tp_ctr_gpe"],
			"positions": pos_gpe,
			"edge_wrap": False,
			"elements": "terub_neuron_gpe"})

stn_neurons = nest.GetNodes(stn, local_only=True)[0]
nest.SetStatus(stn_neurons, 'synaptic_elements', synaptic_elements_e)
gpe_neurons = nest.GetNodes(gpe, local_only=True)[0]
nest.SetStatus(gpe_neurons, 'synaptic_elements', synaptic_elements_i)


stn_node_info = nest.GetStatus(stn_neurons, ['global_id', 'vp', 'local'])
local_stn_nodes  = [(gid, vp) for gid, vp, islocal in stn_node_info if islocal]
n_rec_local_stn = numpy.sum(nest.GetStatus(stn_neurons,"local"))
sn_rate_stn={}
for gid,vp in local_stn_nodes:
	nest.SetStatus([gid], {"V_m":pyrngs[vp].uniform(sim["vm_min_stn"],sim["vm_max_stn"]),
				"g_K":pyrngs[vp].normal(loc=sim["g_K_stn"],scale=sim["sigma_g_K_stn"]),
				"g_Na":pyrngs[vp].normal(loc=sim["g_Na_stn"],scale=sim["sigma_g_Na_stn"]),
		        "g_Ca":pyrngs[vp].normal(loc=sim["g_Ca_stn"],scale=sim["sigma_g_Ca_stn"]),
				"g_T":pyrngs[vp].normal(loc=sim["g_T_stn"],scale=sim["sigma_g_T_stn"]),
				"g_ahp":pyrngs[vp].normal(loc=sim["g_ahp_stn"],scale=sim["sigma_g_ahp_stn"]),
				"g_L":pyrngs[vp].normal(loc=sim["g_L_stn"],scale=sim["sigma_g_L_stn"]),
				"E_K":pyrngs[vp].normal(loc=sim["E_K_stn"],scale=sim["sigma_E_K_stn"]),
				"E_Na":pyrngs[vp].normal(loc=sim["E_Na_stn"],scale=sim["sigma_E_Na_stn"]),
				"E_Ca":pyrngs[vp].normal(loc=sim["E_Ca_stn"],scale=sim["sigma_E_Ca_stn"]),
				"E_L":pyrngs[vp].normal(loc=sim["E_L_stn"],scale=sim["sigma_E_L_stn"])
				})


gpe_node_info = nest.GetStatus(gpe_neurons, ["global_id","vp","local"])
local_gpe_nodes = [(gid,vp) for gid,vp,islocal in gpe_node_info if islocal]
n_rec_local_gpe = numpy.sum(nest.GetStatus(gpe_neurons,"local"))
sn_rate_gpe={}
for gid,vp in local_gpe_nodes:
	nest.SetStatus([gid], {"V_m":pyrngs[vp].uniform(sim["vm_min_gpe"],sim["vm_max_gpe"]),
				"g_K":pyrngs[vp].normal(loc=sim["g_K_gpe"],scale=sim["sigma_g_K_gpe"]),
				"g_Na":pyrngs[vp].normal(loc=sim["g_Na_gpe"],scale=sim["sigma_g_Na_gpe"]),
				"g_Ca":pyrngs[vp].normal(loc=sim["g_Ca_gpe"],scale=sim["sigma_g_Ca_gpe"]),
				"g_T":pyrngs[vp].normal(loc=sim["g_T_gpe"],scale=sim["sigma_g_T_gpe"]),
				"g_ahp":pyrngs[vp].normal(loc=sim["g_ahp_gpe"],scale=sim["sigma_g_ahp_gpe"]),
				"g_L":pyrngs[vp].normal(loc=sim["g_L_gpe"],scale=sim["sigma_g_L_gpe"]),\
				"E_K":pyrngs[vp].normal(loc=sim["E_K_gpe"],scale=sim["sigma_E_K_gpe"]),
				"E_Na":pyrngs[vp].normal(loc=sim["E_Na_gpe"],scale=sim["sigma_E_Na_gpe"]),
				"E_Ca":pyrngs[vp].normal(loc=sim["E_Ca_gpe"],scale=sim["sigma_E_Ca_gpe"]),
				"E_L":pyrngs[vp].normal(loc=sim["E_L_gpe"],scale=sim["sigma_E_L_gpe"])
				})



#-------------------------------------------------------------------------------------------#
#-------------------------------- CREATE DEVICES--------------------------------------------#
#-------------------------------------------------------------------------------------------#

#setup spike detectors for local synchronization
if sim["rec_ls"] and sim["n_stn"]>1000:
	sd_layer_1mm = topo.CreateLayer({"extent": sim["tp_ext_stn"],
									"center": sim["tp_ctr_stn"],
									"positions": fncts.sd_coord_1mm(),
									"edge_wrap": False,
									"elements": "spike_detector"})
	sd_ls_nodes_1mm = nest.GetNodes(sd_layer_1mm)[0]

#setup spike detector
spikes = nest.Create("spike_detector",2,
		[{"label":"spikes-stn", "to_file":True, "to_screen":False},
		{"label":"spikes-gpe", "to_file":True, "to_screen":False}])
spikes_stn = spikes[:1]
spikes_gpe = spikes[1:] 
spikes_stn2 = nest.Create("spike_detector",1,
                [{"label":"spikes-stn2", "to_file":True, "to_screen":False}])
#setup noise generators
noise_stn = nest.Create("poisson_generator")
noise_gpe = nest.Create("poisson_generator")
nest.SetStatus(noise_stn,"rate",sim["noise_rate_stn"])
nest.SetStatus(noise_gpe,"rate",sim["noise_rate_gpe"])

#setup stimulation devices
#delay between contacts in ms
cr_dly = 1.0 / (sim["stim_fburst"]*len(sim["pos_electrode"]))*1000.0

#current controlled stimulation
if sim["stim_cur"] == True:
	electrode = topo.CreateLayer({"extent": sim["tp_ext_stn"],
					"center": sim["tp_ctr_stn"],
					"positions": sim["pos_electrode"],
					"edge_wrap": False,
					"elements": "step_current_rec"})
	contacts = nest.GetNodes(electrode)[0]
	cr_times = [[] for i in range(len(contacts))]
	cr_amp = []
	#stimulation_events is the number of complete stimulation cycles we want to have. We start with 2.
	per_times = []
	crsimple_times = []
	crrand_times = []
	per_amp = []
	crrand_amp = []
	crsimple_amp = []
	if sim["stim_periodic"]:
		per_times,per_amp=stim.cr_sequence(sim["stim_start_per"], sim["stim_stop_per"],
			 sim["stim_amp"], sim["stim_width"], 
			 sim["stim_ps"], sim["stim_gap"], 
			 sim["stim_ibf"], sim["stim_npulses"],sim["stim_fburst"], 
			 sim["stim_cy_on_per"], sim["stim_cy_off_per"],
			 len(contacts))	


	if sim["stim_cr_rand"]:
		crrand_times,crrand_amp=stim.rand_cr_sequence(sim["stim_start"], sim["stim_stop"],
							 sim["stim_amp"], sim["stim_width"], 
							 sim["stim_ps"], sim["stim_gap"], 
							 sim["stim_ibf"], sim["stim_npulses"],
							 sim["stim_fburst"], 
							 sim["stim_cy_on"], sim["stim_cy_off"],
							 len(contacts),pyrngs[rank])


	#else:
	if sim["stim_cr_simple"]:
		crsimple_times,crsimple_amp=stim.cr_sequence(sim["stim_start"]+cr_dly*float(i), 
						sim["stim_stop"]+cr_dly*float(i),
						 sim["stim_amp"], sim["stim_width"], 
						 sim["stim_ps"], sim["stim_gap"], 
						 sim["stim_ibf"], sim["stim_npulses"],
						 sim["stim_fburst"], 
						 sim["stim_cy_on"], sim["stim_cy_off"],
						 len(contacts))
	for i in range(len(contacts)):
		if len(per_times)>0:
			cr_times[i].extend(per_times[i])
		if len(crrand_times)>0:
			cr_times[i].extend(crrand_times[i])
		if len(crsimple_times) >0:
			cr_times[i].extend(crsimple_times[i])
	cr_amp.extend(per_amp)
	cr_amp.extend(crrand_amp)
	cr_amp.extend(crsimple_amp)
        cramp_len = len(cr_amp)
        for j in range(1,sim["stimulation_times"]):
            if sim["short_follow_iterations"]:
                 if j == 1:
                     cr_amp.extend(cr_amp[:int(float(cramp_len)*(7./12.))])
                 else:
                     cr_amp.extend(cr_amp[:int(float(cramp_len)*(6./12.))])
            else:
                 cr_amp.extend(cr_amp)
        cr_len = len(cr_times[0])
	for i in range(len(contacts)):
            for j in range(1,sim["stimulation_times"]):
                if sim["short_follow_iterations"]:
                    if j == 1:
                        cr_times[i].extend([x+(sim["end_epoch"]*j) for x in cr_times[i][:int(float(cr_len)*(7./12.))]]) #7.6
                    else:
                        cr_times[i].extend([x+(sim["end_epoch"]*j) for x in cr_times[i][:int(float(cr_len)*(6./12.))]]) #6.5
                else:
                    cr_times[i].extend([x+(sim["end_epoch"]*j) for x in cr_times[i]])
	    nest.SetStatus([contacts[i]], {"amplitude_times": cr_times[i],	
						"amplitude_values":cr_amp})
		
#setup voltmeter to investigate time course of mean membrane potentials
if sim["rec_vm"]:
	vm = nest.Create("multimeter",2,
			[{"record_from":["V_m"], "to_accumulator":True,"interval":sim["res_vm"]},
			{"record_from":["V_m"],"to_accumulator":True,"interval":sim["res_vm"]}])
	vm_stn = vm[:1]
	vm_gpe = vm[1:]

node_creation_stop = time.time()

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#----------------------------------- WIRING ------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

wiring_start = time.time()


#-------------------------------------------------------------------------------------------#
#-------------------------------- WIRE NEURONS ---------------------------------------------#
#-------------------------------------------------------------------------------------------#
#We are not using a static synapse. We are modifying stdp synapses.
#Publish the plastic synapses to the Structural plasticity algorithm by linking them to the synaptic elements.

if sim["gaussian_weights"]:
	w_ss = {'normal':{'mean':sim["offset_g_ss"],'sigma':sim["sigma_g_ss"]}}
	w_sg = {'normal':{'mean':sim["offset_g_sg"],'sigma':sim["sigma_g_sg"]}}
	w_gs = {'normal':{'mean':sim["offset_g_gs"],'sigma':sim["sigma_g_gs"]}}
	w_gg = {'normal':{'mean':sim["offset_g_gg"],'sigma':sim["sigma_g_gg"]}}
else:
	w_ss = {'exponential':{'c':sim["offset_g_ss"],'a':sim["g_ss"],'tau':sim["dec_g_ss"]}}
	w_sg = {'exponential':{'c':sim["offset_g_sg"],'a':sim["g_sg"],'tau':sim["dec_g_sg"]}}
	w_gs = {'exponential':{'c':sim["offset_g_gs"],'a':sim["g_gs"],'tau':sim["dec_g_gs"]}}
	w_gg = {'exponential':{'c':sim["offset_g_gg"],'a':sim["g_gg"],'tau':sim["dec_g_gg"]}}

if sim["dist_con_prob"]:
	kernel_ss = {'exponential':{'c':sim["offset_c_ss"],'a':sim["c_ss"],'tau':sim["dec_c_ss"]}}
	kernel_sg = {'exponential':{'c':sim["offset_c_sg"],'a':sim["c_sg"],'tau':sim["dec_c_sg"]}}
	kernel_gs = {'exponential':{'c':sim["offset_c_gs"],'a':sim["c_gs"],'tau':sim["dec_c_gs"]}}
	kernel_gg = {'exponential':{'c':sim["offset_c_gg"],'a':sim["c_gg"],'tau':sim["dec_c_gg"]}}
else:
	kernel_ss = 1.0
	kernel_sg = 1.0
	kernel_gs = 1.0
	kernel_gg = 1.0


if sim["stdp_on"] == True:
	if sim["stdp_ss"] == True:
		param_ss = {"weight": sim["offset_g_ss"], "Wmax": sim["stdp_Wmax_ss"],
		"alpha":sim["stdp_alpha_ss"], "lambda":sim["stdp_lambda_ss"],
		"mu_plus":sim["stdp_mu_plus_ss"], "mu_minus":sim["stdp_mu_minus_ss"],
		"tau_plus":sim["stdp_tau_plus_ss"]}
		nest.CopyModel("stdp_synapse", "ss_synapse",params=param_ss)
	else:
		nest.CopyModel("static_synapse", "ss_synapse")
	#We are only changing these synapses during the simulation

	if sim["stdp_sg"] == True:
		param_ss = {"Wmax": sim["stdp_Wmax_sg"],
		"alpha":sim["stdp_alpha_sg"], "lambda":sim["stdp_lambda_sg"],
		"mu_plus":sim["stdp_mu_plus_sg"], "mu_minus":sim["stdp_mu_minus_sg"],
		"tau_plus":sim["stdp_tau_plus_sg"]}
		nest.CopyModel("stdp_synapse", "sg_synapse",params=param_ss)
	else:
		nest.CopyModel("static_synapse", "sg_synapse")
                vals = {'weight': sim["offset_g_sg"], 'delay':sim['delay_sg'],}
                nest.SetDefaults("sg_synapse", vals)

	if sim["stdp_gs"] == True:
		param_ss = {"Wmax": sim["stdp_Wmax_gs"],
		"alpha":sim["stdp_alpha_gs"], "lambda":sim["stdp_lambda_gs"],
		"mu_plus":sim["stdp_mu_plus_gs"], "mu_minus":sim["stdp_mu_minus_gs"],
		"tau_plus":sim["stdp_tau_plus_gs"]}
		nest.CopyModel("stdp_synapse", "gs_synapse",params=param_ss)
	else:
		nest.CopyModel("static_synapse", "gs_synapse")

	if sim["stdp_gg"] == True:
		param_gg = {"Wmax": sim["stdp_Wmax_gg"],
		"alpha":sim["stdp_alpha_gg"], "lambda":sim["stdp_lambda_gg"],
		"mu_plus":sim["stdp_mu_plus_gg"], "mu_minus":sim["stdp_mu_minus_gg"],
		"tau_plus":sim["stdp_tau_plus_gg"]}
		param_gg = {"Wmax": sim["stdp_Wmax_gg"],
		"alpha":sim["stdp_alpha_gg"], "lambda":sim["stdp_lambda_gg"],
		"mu_plus":sim["stdp_mu_plus_gg"], "mu_minus":sim["stdp_mu_minus_gg"],
		"tau_plus":sim["stdp_tau_plus_gg"]}
		nest.CopyModel("stdp_synapse", "gg_synapse",params=param_gg)
	else:
		nest.CopyModel("static_synapse", "gg_synapse")
                vals = {'weight': sim["offset_g_gg"], 'delay':sim['delay_gg'],}
                nest.SetDefaults("gg_synapse", vals)

        nest.SetStructuralPlasticityStatus({'structural_plasticity_synapses':
                                   {'ss_synapse':
                                   {'model': 'ss_synapse',
                                    'post_synaptic_element': 'Den_ex',
                                    'pre_synaptic_element': 'Axon_ex',
                                   },
                                   'sg_synapse':
                                   {'model': 'sg_synapse',
                                    'post_synaptic_element': 'Den_in',
                                    'pre_synaptic_element': 'Axon_in',
                                   },
                                   'gg_synapse':
                                   {'model': 'gg_synapse',
                                    'post_synaptic_element': 'Den_inin',
                                    'pre_synaptic_element': 'Axon_inin',
                                   }
                                   }
                                   })

else:
	nest.CopyModel("static_synapse", "ss_synapse")
	nest.CopyModel("static_synapse", "sg_synapse")
	nest.CopyModel("static_synapse", "gs_synapse")
	nest.CopyModel("static_synapse", "gg_synapse")


mask_all = {'volume': {'lower_left': sim["tp_mask_all_ll"],
	 'upper_right': sim["tp_mask_all_ur"]},
	 'anchor':sim["tp_mask_all_ctr"]}

syn_elems_i = nest.GetStatus(stn_neurons, 'synaptic_elements')
sum_neurons = sum(neuron['Axon_ex']['z_connected'] for neuron in syn_elems_i)
sum_neurons = comm.gather(sum_neurons, root=0)
if nest.Rank() == 0:
    print ("Total connected before conn syn elements: " + str((sum(sum_neurons))))
    print ("\n new conns:" + str(int(sim["con_ss"]*sim["n_stn"]) * sim["n_stn"]))


#STN-STN
topo.ConnectLayers(stn, stn,
		{'connection_type': 'divergent', 
		 'allow_autapses': False,
		 'allow_multapses': False,
		 'allow_oversized_mask': True,
		 'mask': mask_all,
		 'number_of_connections':int(sim["con_ss"]*sim["n_stn"]), # WHAT do I change here??
		 'kernel': kernel_ss,
		 'weights': w_ss,
		 'delays':sim['delay_ss'],
		 'synapse_model':'ss_synapse',
                 'pre_synaptic_element':'Axon_ex',
                 'post_synaptic_element':'Den_ex'})



#STN-GPe
topo.ConnectLayers(stn, gpe,
		{'connection_type': 'divergent', 
		 'allow_autapses': False,
		 'allow_multapses': False,
 		 'allow_oversized_mask': True,
		 'allow_multapses': False,
 		 'allow_oversized_mask': True,
		 'allow_multapses': False,
 		 'allow_oversized_mask': True,
		 'mask': mask_all,
		 'number_of_connections':int(sim["con_sg"]*sim["n_gpe"]),
 		 'kernel': kernel_sg,
		 'weights': w_sg,
 		 'delays':sim['delay_sg'],
		 'synapse_model':'sg_synapse',
                 'pre_synaptic_element':'Axon_in',
                 'post_synaptic_element':'Den_in'})

#GPe-STN
topo.ConnectLayers(gpe, stn,
		{'connection_type': 'divergent', 
		 'allow_autapses': False,
		 'allow_multapses': False,
 		 'allow_oversized_mask': True,
		 'mask': mask_all,
		 'number_of_connections':int(sim["con_gs"]*sim["n_stn"]),
		 'kernel': kernel_gs,
		 'weights': w_gs,
 		 'delays':sim['delay_gs'],
		 'synapse_model':'gs_synapse'})	

#GPe-GPe
topo.ConnectLayers(gpe, gpe,
		{'connection_type': 'divergent', 
		'allow_autapses': False,
		'allow_multapses': False,
		'allow_oversized_mask': True,
		'mask': mask_all,
		'number_of_connections':int(sim["con_gg"]*sim["n_gpe"]),
		'kernel': kernel_gg,
		'weights': w_gg,
		'delays':sim['delay_gg'],
		'synapse_model':'gg_synapse',
                'pre_synaptic_element':'Axon_inin',
                'post_synaptic_element':'Den_inin'})

#-------------------------------------------------------------------------------------------#
#--------------------------------- WIRE DEVICES --------------------------------------------#
#-------------------------------------------------------------------------------------------#

#connect noise generators
nest.CopyModel("static_synapse", "noise_conn_stn", 
	{"weight":sim["noise_weight_stn"], "delay":sim["noise_delay_stn"]})
nest.CopyModel("static_synapse", "noise_conn_gpe", 
	{"weight":sim["noise_weight_gpe"], "delay":sim["noise_delay_gpe"]})
nest.Connect(noise_stn, stn_neurons, conn_spec="all_to_all", syn_spec="noise_conn_stn")
nest.Connect(noise_gpe, gpe_neurons, conn_spec="all_to_all", syn_spec="noise_conn_gpe")




#connect stimulation device
if sim["stim_cr_on"]: 
	nest.CopyModel("static_synapse","stim_synapse",{"delay":sim["stim_delay"]})
	
	if sim["stim_cur"]:
		topo.ConnectLayers(electrode, stn,
			{'connection_type': 'divergent', 
			'allow_autapses': False,
			'allow_multapses': False,
			'allow_oversized_mask': True,
			'mask': mask_all,
			'kernel': 1.0,
			'weights':{'efield':{'c': sim["stim_c_cur"], 
								 'p_center': sim["stim_p_center_cur"], 
								 'sigma': sim["stim_sigma_cur"]}},
			#'weights':{'exponential':{'c': sim["stim_c_cur"], 
			#                          'a': sim["stim_a_cur"], 
			#                          'tau': sim["stim_tau_cur"]}},
			'synapse_model':'stim_synapse'})
	else:#if sim["stim_syn"]:
		topo.ConnectLayers(electrode, stn,
			{'connection_type': 'divergent', 
			'allow_autapses': False,
			'allow_multapses': False,
			'allow_oversized_mask': True,
			'mask': mask_all,
			'kernel': 1.0,
			'weights':{'efield':{'c': sim["stim_c_cur"], 
			'sigma': sim["stim_sigma_cur"]}},
			'synapse_model':'stim_synapse'})

#connect spike detectors
nest.CopyModel("static_synapse", "spikedetector_synapse")
nest.Connect(stn_neurons, spikes_stn, conn_spec="all_to_all",syn_spec="static_synapse")
nest.Connect(gpe_neurons, spikes_gpe, conn_spec="all_to_all",syn_spec="static_synapse")

#connect spike detectors for local synchronization
if sim["rec_ls"] and sim["n_stn"]>1000:
	nest.CopyModel("static_synapse", "ls_synapse_1mm",{"weight":1.0})
	topo.ConnectLayers(stn,sd_layer_1mm,
			{'connection_type': 'convergent', 
			 'allow_autapses': False,
			 'allow_multapses': False,
		 	 'mask': {'volume':{'lower_left':sim["tp_mask_ls_ll_1mm"],'upper_right':sim["tp_mask_ls_ur_1mm"]}},
		 	 'kernel':1.0,
		 	 'synapse_model':'ls_synapse_1mm'})

	if rank ==0:
		topo.DumpLayerNodes(sd_layer_1mm,sim["data_path"]+sim["fname"]+"-sd_ls_nodes_1mm.txt")
	topo.DumpLayerConnections(stn,"ls_synapse_1mm",sim["data_path"]+"stn-sd_ls-con_1mm.dat")

#connect voltmeter
if sim["rec_vm"]:
	nest.Connect(vm_stn, stn_neurons, "all_to_all")
	nest.Connect(vm_gpe, stn_neurons, "all_to_all")

wiring_stop = time.time()

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------- INIT -----------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#used for scalability tests only!
init_start = time.time()
#nest.Simulate(sim["tinterval"])
#nest.SetStatus(spikes,{"n_events":0})
init_stop = time.time()

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#----------------------------------- SIMULATION --------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
syn_elems_i = nest.GetStatus(stn_neurons, 'synaptic_elements')
sum_neurons = sum(neuron['Axon_ex']['z_connected'] for neuron in syn_elems_i)
sum_neurons = comm.gather(sum_neurons, root=0)
sum_neuronsin = sum(neuron['Axon_in']['z_connected'] for neuron in syn_elems_i)
sum_neuronsin = comm.gather(sum_neuronsin, root=0)
if nest.Rank() == 0:
    print ("Sum neurons:" + str(sum_neurons))
    print ("Total connected syn elements: " + str((sum(sum_neurons))))
    print ("Total connected syn elements In: " + str((sum(sum_neuronsin))))

sim_start = time.time()
rate_stn = 0.
rate_gpe = 0.
rate_t = numpy.zeros(shape=(int(sim["runs"]),3),dtype="float")
r1_tav_stn = 0.
r2_tav_stn = 0.
r3_tav_stn = 0.
r4_tav_stn = 0.

r1_tav_gpe = 0.
r2_tav_gpe = 0.
r3_tav_gpe = 0.
r4_tav_gpe = 0.

if sim["rec_ls"] and sim["n_stn"]>1000:
	n_sd_ls_1mm=len(sd_ls_nodes_1mm)
	r1_ls_stn_1mm = numpy.zeros(shape=(n_sd_ls_1mm,int(sim["tinterval"]/sim["res_loc_sync_v"])),dtype="complex")
	r2_ls_stn_1mm = numpy.zeros(shape=(n_sd_ls_1mm,int(sim["tinterval"]/sim["res_loc_sync_v"])),dtype="complex")
	r3_ls_stn_1mm = numpy.zeros(shape=(n_sd_ls_1mm,int(sim["tinterval"]/sim["res_loc_sync_v"])),dtype="complex")
	r4_ls_stn_1mm = numpy.zeros(shape=(n_sd_ls_1mm,int(sim["tinterval"]/sim["res_loc_sync_v"])),dtype="complex")


#Retrieve all stn connections
tall_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")
tall_stn_conns = comm.allgather(tall_stn_conns)
all_stn_conns = []
for elem in tall_stn_conns:
    all_stn_conns.extend(elem)
if nest.Rank() == 0:
    print ("\n with getconnect Sum neurons:" + str(len(all_stn_conns)))
all_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")

for run in range(sim["runs"]):
#simulate neural system
        
	if sim["sp_enabled"] and (run == sim["start_sp_2"] or run == sim["start_sp"]):
		if nest.Rank() == 0:
                    print ("Disabling STDP and enabling SP")
		# Disable STDP
                tall_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")
                tall_stn_conns = comm.allgather(tall_stn_conns)
                all_stn_conns = []
                for elem in tall_stn_conns:
                    all_stn_conns.extend(elem)               
                if nest.Rank() == 0:
		    print(str(len(all_stn_conns))+ " conns in rank "  + str(nest.Rank()))
                #get synapses per thread
                all_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")
                try:
			nest.SetStatus(list(all_stn_conns[0:-1]), {"enabled": False})
		except:
			print ("Failed to disable stdp")
			print ("Error in Set Status:", sys.exc_info()[0])
		# Enable structural plasticity           
		nest.EnableStructuralPlasticity()
                growth_curve_e_e = {
                    'growth_rate': 0.00008, 
                    'eta': -0.005,
                    'eps': 0.03, 
                }
                growth_curve_e_i = {
                    'growth_rate': 0.00002, 
                    'eta': -0.005,
                    'eps': 0.03, 
                }
                growth_curve_i_e = {
                    'growth_rate': -0.005, 
                    'eta': -0.005,
                    'eps': 0.07, 
                }
                growth_curve_i_i = {
                    'growth_rate': 0.000,
                    'eta': -0.005,
                    'eps': 0.07,
                }
                growth_curve_e_e2 = {
                    'growth_rate': -0.005,
                    'eta': -0.005,
                    'eps': 0.03,
                }
                synaptic_elements_e = {'Den_ex': growth_curve_e_e2,'Axon_ex': growth_curve_e_e, 'Axon_in': growth_curve_e_i,}
                synaptic_elements_i = {'Den_in': growth_curve_i_e,'Den_inin': growth_curve_i_i,'Axon_inin': growth_curve_i_i}

                nest.SetStatus(stn_neurons, 'update_synaptic_elements', synaptic_elements_e)
                nest.SetStatus(gpe_neurons, 'update_synaptic_elements', synaptic_elements_i)
        if sim["sp_enabled"] and (run == sim["stop_sp_2"] or run == sim["stop_sp"]):
                if nest.Rank() == 0:
                    print ("Stop SP")
                tall_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")
                tall_stn_conns = comm.allgather(tall_stn_conns)
                all_stn_conns = []
                for elem in tall_stn_conns:
                    all_stn_conns.extend(elem)               
                if nest.Rank() == 0:
                    print(str(len(all_stn_conns)) +" conns in rank "+ str(nest.Rank()))
                tall_gpe_conns = nest.GetConnections(target=gpe_neurons, synapse_model="sg_synapse")
                tall_gpe_conns = comm.allgather(tall_gpe_conns)
                all_gpe_conns = []
                for elem in tall_gpe_conns:
                    all_gpe_conns.extend(elem)
                if nest.Rank() == 0:
                    print(str(len(all_gpe_conns)) +" conns in rank "+ str(nest.Rank()))
                growth_curve_off = { 'growth_rate': 0.0, }
                synaptic_elements_e = {'Den_ex': growth_curve_off,'Axon_ex': growth_curve_off, 'Axon_in': growth_curve_off,}
                synaptic_elements_i = {'Den_in': growth_curve_off,'Den_inin': growth_curve_off,'Axon_inin': growth_curve_off}
                nest.SetStatus(stn_neurons, 'update_synaptic_elements', synaptic_elements_e)
                nest.SetStatus(gpe_neurons, 'update_synaptic_elements', synaptic_elements_i)
                nest.DisableStructuralPlasticity()
                #get synapses per thread
                all_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")
                try:
                        nest.SetStatus(list(all_stn_conns[0:-1]), {"enabled": True})
                except:
                        print ("Error in Set Status:", sys.exc_info()[0])
        nest.Simulate(sim["tinterval"])
        if rank == 0:
	        print ("After simulate: " + str(run)+"\n")
                print ("Ca: " + str(numpy.mean(nest.GetStatus(stn_neurons,'Ca')))+ "\n")
                print ('Ca GPE:' +str(nest.GetStatus(gpe_neurons, 'Ca'))+ "\n")
        if ((run >= sim["start_sp"] and run < sim["stop_sp"]) or (run >= sim["start_sp_2"] and run < sim["stop_sp_2"])) and sim["sp_enabled"]:
                block = sim["start_sp"]
                if run >= sim["start_sp_2"]:
                    block = sim["start_sp_2"]
                syn_elems_i = nest.GetStatus(stn_neurons, 'synaptic_elements')
                sum_neurons = sum(neuron['Axon_ex']['z_connected'] for neuron in syn_elems_i)
                sum_neurons = comm.gather(sum_neurons, root=0)
                sum_neuronsin = sum(neuron['Axon_in']['z_connected'] for neuron in syn_elems_i)
                sum_neuronsin = comm.gather(sum_neuronsin, root=0)
                if rank == 0:
                   print ("Total connected syn elements: " + str((sum_neurons))+ "\n")
                   print ("Total connected syn elements in: " + str((sum_neuronsin))+ "\n")
                tall_stn_conns = nest.GetConnections(target=stn_neurons, synapse_model="ss_synapse")
                tall_stn_conns = comm.gather(tall_stn_conns, root=0)
                if rank ==0:
                   all_stn_conns = []
                   for elem in tall_stn_conns:
                       all_stn_conns.extend(elem)
                   print ("\n With getconnect Sum neurons:" + str(len(all_stn_conns))+ "\n")
#spike rate
	n_events = nest.GetStatus(spikes,"n_events")
	rate_stn += n_events[0]/(sim["tinterval"]*sim["runs"])*1000.0/n_rec_local_stn
	rate_gpe += n_events[1]/(sim["tinterval"]*sim["runs"])*1000.0/n_rec_local_gpe
	rate_t[run][0] = sim["tinterval"]*float(run+1)
	rate_t[run][1] = n_events[0]/(sim["tinterval"])*1000.0/n_rec_local_stn
	rate_t[run][2] = n_events[1]/(sim["tinterval"])*1000.0/n_rec_local_gpe
	
#sync measure
	s_stn,s_gpe = nest.GetStatus(spikes,"events")
	z_stn = zip(s_stn["senders"],s_stn["times"])
	z_gpe = zip(s_gpe["senders"],s_gpe["times"])
	if sim["rec_isi"]:
		r1_stn,r2_stn,r3_stn,r4_stn,sn_rate_stn = fncts.calc_sync(z_stn,"stn",run,sn_rate_stn,sim,f_loc_sync,f_isi_stn)
		r1_gpe,r2_gpe,r3_gpe,r4_gpe,sn_rate_gpe = fncts.calc_sync(z_gpe,"gpe",run,sn_rate_gpe,sim,f_loc_sync,f_isi_gpe)
	else:
		r1_stn,r2_stn,r3_stn,r4_stn,sn_rate_stn = fncts.calc_sync(z_stn,"stn",run,sn_rate_stn,sim,f_loc_sync)
		r1_gpe,r2_gpe,r3_gpe,r4_gpe,sn_rate_gpe = fncts.calc_sync(z_gpe,"gpe",run,sn_rate_gpe,sim,f_loc_sync)

	nest.SetStatus(spikes,{"n_events":0})
	tmp=comm.gather(r1_stn,root=0)
	if rank ==0:r1_stn = abs(numpy.sum(tmp,axis=0))/sim["n_stn"]
	tmp=comm.gather(r2_stn,root=0)
	if rank ==0:r2_stn = abs(numpy.sum(tmp,axis=0))/sim["n_stn"]
	tmp=comm.gather(r3_stn,root=0)
	if rank ==0:r3_stn = abs(numpy.sum(tmp,axis=0))/sim["n_stn"]
	tmp=comm.gather(r4_stn,root=0)
	if rank ==0:r4_stn = abs(numpy.sum(tmp,axis=0))/sim["n_stn"]

	tmp=comm.gather(r1_gpe,root=0)
	if rank ==0:r1_gpe = abs(numpy.sum(tmp,axis=0))/sim["n_gpe"]
	tmp=comm.gather(r2_gpe,root=0)
	if rank ==0:r2_gpe = abs(numpy.sum(tmp,axis=0))/sim["n_gpe"]
	tmp=comm.gather(r3_gpe,root=0)
	if rank ==0:r3_gpe = abs(numpy.sum(tmp,axis=0))/sim["n_gpe"]
	tmp=comm.gather(r4_gpe,root=0)
	if rank ==0:r4_gpe = abs(numpy.sum(tmp,axis=0))/sim["n_gpe"]

	if rank ==0:
		r1_tav_stn,r2_tav_stn,r3_tav_stn,r4_tav_stn,r1_tav_gpe,r2_tav_gpe,r3_tav_gpe,r4_tav_gpe= \
		fncts.save_glo_sync(r1_stn,r2_stn,r3_stn,r4_stn,\
		r1_gpe,r2_gpe,r3_gpe,r4_gpe,\
		r1_tav_stn,r2_tav_stn,r3_tav_stn,r4_tav_stn,\
		r1_tav_gpe,r2_tav_gpe,r3_tav_gpe,r4_tav_gpe,\
		f_glo_sync,run,sim)

#mean mebrane potential
	if sim["rec_vm"]:
		vs = nest.GetStatus(vm_stn,"events")[0]
		gather_vm_stn = comm.gather(vs["V_m"],root=0)
		vg = nest.GetStatus(vm_gpe,"events")[0]
		gather_vm_gpe = comm.gather(vg["V_m"],root=0)
		nest.SetStatus(vm,{"n_events":0})
		if rank ==0:
			 vt=vs["times"]
			 v_mean_stn = numpy.sum(gather_vm_stn,axis=0)/sim["n_stn"]
			 v_mean_gpe = numpy.sum(gather_vm_gpe,axis=0)/sim["n_gpe"]
			 for j in range(len(vt)):
				f_vm.write(str(vt[j])+"\t"+str(v_mean_stn[j])+"\t"+str(v_mean_gpe[j])+"\n")
		else:
			assert gather_vm_stn is None
			assert gather_vm_gpe is None

#voxel based local sync
	if sim["rec_ls"] and sim["n_stn"]>1000:
		for i in range(len(sd_ls_nodes_1mm)):
			s_sd = nest.GetStatus([sd_ls_nodes_1mm[i]],"events")[0]
			z_sd = zip(s_sd["senders"],s_sd["times"])
			r1_ls_stn_1mm[i,:],r2_ls_stn_1mm[i,:],r3_ls_stn_1mm[i,:],r4_ls_stn_1mm[i,:] = \
			fncts.calc_loc_sync(z_sd,run,sim)
		gather_r1_ls_stn_1mm=comm.gather(r1_ls_stn_1mm,root=0)
		gather_r2_ls_stn_1mm=comm.gather(r2_ls_stn_1mm,root=0)
		gather_r3_ls_stn_1mm=comm.gather(r3_ls_stn_1mm,root=0)
		gather_r4_ls_stn_1mm=comm.gather(r4_ls_stn_1mm,root=0)

		if rank == 0:
			fncts.save_loc_sync(gather_r1_ls_stn_1mm,gather_r2_ls_stn_1mm,\
								gather_r3_ls_stn_1mm,gather_r4_ls_stn_1mm,\
								n_sd_ls_1mm,f_ls_r1_1mm,f_ls_r2_1mm,\
								f_ls_r3_1mm,f_ls_r4_1mm,f_ls_p_1mm,run,sim)
		else:
			assert gather_r1_ls_stn_1mm is None
			assert gather_r2_ls_stn_1mm is None
			assert gather_r3_ls_stn_1mm is None
			assert gather_r4_ls_stn_1mm is None
# Record connectivity afterwards
#topo.DumpLayerConnections(stn,"ss_synapse",sim["topo_path"]+"structural_plasticity_"+str(run))
nest.DisableStructuralPlasticity()
#close datafiles
if sim["rec_isi"]:
	f_isi_stn.close()
	f_isi_gpe.close()
if sim["rec_ls"] and sim["n_stn"]<=1000:
	for l in range(0,len(f_loc_sync)):
		f_loc_sync[l].close()

#record connection data for last timestep
fncts.dump_conn(stn,gpe,sim["runs"],sim)

sim_stop = time.time()

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#--------------------------------- DATA PROCESSING -----------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
t_gather_start = time.time()
#gather mean firing rate for stn and gpe
gather_rate_stn = comm.gather(rate_stn, root=0)
gather_rate_gpe = comm.gather(rate_gpe, root=0)
gather_rate_t   = comm.gather(rate_t, root=0)

#gather mean firing rate of single neurons
gather_sn_rate_stn = comm.gather(sn_rate_stn,root=0)
gather_sn_rate_gpe = comm.gather(sn_rate_gpe,root=0)

#compute and gather computation times
t_node = node_creation_stop - node_creation_start
t_wire = wiring_stop - wiring_start
t_init = init_stop - init_start
t_comp = sim_stop - sim_start
gather_t_node = comm.gather(t_node,root=0)
gather_t_wire = comm.gather(t_wire,root=0)
gather_t_init = comm.gather(t_init,root=0)
gather_t_comp = comm.gather(t_comp,root=0)
t_gather_stop = time.time()
t_gather = t_gather_stop - t_gather_start
gather_t_gather = comm.gather(t_gather,root=0)

#shorten spike record
if sim["n_stn"]+sim["n_gpe"]>5000:
	fncts.cut_spikes(spikes,rank,format_string,sim)

#merge local sync data
if sim["rec_ls"] and sim["n_stn"]<=1000:
	if rank > 0 and rank <= sim["runs"]:
		os.chdir(sim["ls_path"])
		part = "stn"
		for run in range(1,sim["runs"]+1):
			if rank == run % sim["N_vp"]:
				t = int(sim["tinterval"])*run
				sources = "local_sync-"+part+"_t="+str(t)+"-*"
				target =  "local_sync-"+part+"_t="+str(t)+".txt"
				cmd = "cat " + sources + " | sort -k1 -n >> " + target
				os.system(cmd)
				os.system("rm "+sources)

if rank == 0:
	t_data_start = time.time()
	prec = 3 #precision of data

#close data files
	f_glo_sync.close()
	if sim["rec_vm"]: f_vm.close()
	if sim["rec_ls"] and sim["n_stn"] > 1000: 
		f_ls_r1_1mm.close()
		f_ls_r2_1mm.close()
		f_ls_r3_1mm.close()
		f_ls_r4_1mm.close()
		f_ls_p_1mm.close()

#compute mean firing rates
	sim["rate_stn"] = round(numpy.sum(gather_rate_stn) / float(sim["N_vp"]),prec)
	sim["rate_gpe"] = round(numpy.sum(gather_rate_gpe) / float(sim["N_vp"]),prec)
	fncts.save_sn_rate(gather_sn_rate_stn,"stn",sim)
	fncts.save_sn_rate(gather_sn_rate_gpe,"gpe",sim)
        rate_t = numpy.sum(gather_rate_t,axis=0) / float(sim["N_vp"])
	f_rate=open(sim["data_path"]+sim["fname"]+"-spike_rate.txt","w")
	for i in range(len(rate_t)):
		f_rate.write(str(rate_t[i][0])+"\t"+str(rate_t[i][1])+"\t"+str(rate_t[i][2])+"\n")
	f_rate.close()

#time averages of global sync
	sim["sync_r1_stn"] = round(r1_tav_stn,prec)
	sim["sync_r2_stn"] = round(r2_tav_stn,prec)
	sim["sync_r3_stn"] = round(r3_tav_stn,prec)
	sim["sync_r4_stn"] = round(r4_tav_stn,prec)

	sim["sync_r1_gpe"] = round(r1_tav_gpe,prec)
	sim["sync_r2_gpe"] = round(r2_tav_gpe,prec)
	sim["sync_r3_gpe"] = round(r3_tav_gpe,prec)
	sim["sync_r4_gpe"] = round(r4_tav_gpe,prec)

#computation times
	sim = fncts.comp_times(gather_t_node,gather_t_wire,gather_t_init,\
						   gather_t_comp,gather_t_gather,sim)

#dump neuron gids and coordinates
	topo.DumpLayerNodes(stn,sim["data_path"]+sim["fname"]+"-stn_nodes.txt")
	topo.DumpLayerNodes(gpe,sim["data_path"]+sim["fname"]+"-gpe_nodes.txt")

#data processing time
	t_data_stop = time.time()
	sim["t_data"] = t_data_stop - t_data_start

#show and dump simulation parameters
	fncts.print_simpar(sim)
	fncts.dump_simpar(sim["data_path"],sim)
else:
	assert gather_rate_stn is None
	assert gather_rate_gpe is None
	assert gather_t_node is None
	assert gather_t_wire is None
	assert gather_t_init is None
	assert gather_t_comp is None
	assert gather_t_gather is None
