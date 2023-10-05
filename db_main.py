#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:01:09 2023

@author: joao

Limitations/possible expansions:
    - real-world
    - larger fingerprints (higher DB level: multi-BS, more CSI info, etc.. )
    - ULA -> UPA
    - Other: 
        - 2D -> 3D (including angles) (unnecessary for most use-cases)
        - parallelization of measurements (compile + parallelize = easy 10Nx, N=cores)
"""
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import general_utils as gu
from RSSI_db import RSSI_Database, RSSI_Measurement_Generator

import DeepMIMO

p = { # Parameters for scene generation (ray tracing simulation)
     'freq': 3e9,
     'tx_pos': [-33,11,32],
     'tx_ori': [0,0,-18.43],
     
      # Parameters for database generation (from the ray tracing simulation)
     'bandwidth': 20e6,     # [Hz]
     'subband_size': 1e6,   # [Hz]
     'n_subbands': 20,      # = int(bandwith / subband_size)
     
     'fov': 120,            # [º] 
     'beam_res': 5,         # [º] angle difference between consecutive beams
     'n_beams': 25,         # = len(np.arange(fov/2, fov/2+.01, beam_res)
     
     'Nt_h': 64,            # TX antenna horizontal elements 
     'Nt_v': 1,             # TX antenna vertical elements 
     
     'db_n_decimals': 1,    # Number of decimal places for the DB (values in dBm)
     
     'grid_dims': [180, 120, 0],  # 3D user grid
     'cell_size': 1, # [m]
     
     # Specific: DeepMIMO
     'scenario': 'simple_street_canyon_test_rays=0p25_res=2m_3ghz',
     'scenarios_folder': 'deepmimo_scenarios', #'/media/joao/2ndStorage/DeepMIMO/Data',
     'db_save_folder': 'databases',
     
     }
#%%
# Load the default parameters
parameters = DeepMIMO.default_params()

parameters['scenario'] = p['scenario']
parameters['active_BS'] = np.array([1])
parameters['user_row_first'] = 1
parameters['user_row_last'] = 61
parameters['bs_antenna']['shape'] = np.array([1, p['Nt_h'], 1])
parameters['bs_antenna']['rotation'] =  np.array(p['tx_ori'])
parameters['ue_antenna']['shape'] = np.array([1, 1, 1])
parameters['enable_BS2BS'] = False
parameters['activate_OFDM'] = 1
parameters['OFDM']['subcarriers'] = 200
parameters['OFDM']['subcarriers_limit'] = 200
parameters['bandwidth'] = .02  # [GHz]

parameters['row_subsampling'] = 1.0
parameters['user_subsampling'] = 1.0

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = p['scenarios_folder']

ant_shape = np.array([1, p['Nt_h'], 1])
oversampling_rate = np.array([1, 1, 1])

beam_angles = gu.get_beam_angles(p['fov'], beam_res=p['beam_res'])

F1 = np.array([gu.get_steering_vec_ULA(ang, n_ele=p['Nt_h']) for ang in beam_angles])

#%% Generate dataset

dataset = DeepMIMO.generate_data(parameters)

grid_idxs_enabled = np.where(dataset[0]['user']['LoS'] != -1)[0]  # enabled UEs
n_active_ues = len(grid_idxs_enabled)

rxs = dataset[0]['user']['location'][grid_idxs_enabled]

full_dbm = np.zeros((p['n_beams'], p['n_subbands'], n_active_ues))
for ue_idx in tqdm(range(n_active_ues), desc='Computing the channel for each user'):
    chs = F1 @ dataset[0]['user']['channel'][grid_idxs_enabled[ue_idx]]
    full_linear = np.abs(np.mean(chs.squeeze().reshape((p['n_beams'], p['n_subbands'], -1)), axis=-1))
    full_dbm[:,:,ue_idx] = 20*np.log10(full_linear) + 30

full_dbm = np.around(full_dbm, p['db_n_decimals'])

#%% Create Database

y_dim = round((parameters['user_row_last'] - parameters['user_row_first']+1) * parameters['row_subsampling'])
x_dim = round(parameters['scenario_params']['user_grids'][0][-1] * parameters['user_subsampling'])
user_grid_dims = np.array([x_dim, y_dim])  # = users per row x # rows

rssi_db = RSSI_Database(dbm=full_dbm,
                        
                        n_beams=p['n_beams'],
                        beam_angles=beam_angles,
                        beam_vectors=F1, # (n_beams x ant)
                        
                        n_subbands=p['n_subbands'],
                        bandwidth=p['bandwidth'],
                        center_freq=p['freq'],
                        
                        rx_pos=rxs,
                        grid_dims=user_grid_dims,
                        grid_idxs_enabled=grid_idxs_enabled,
                        
                        scatter=True,
                        
                        tx_pos=dataset[0]['location'],
                        tx_ori=np.array(p['tx_ori'])*np.pi/180,
                        
                        save_folder=p['db_save_folder'])

# %% Save Database
# rssi_db.save(override=True)

#%% Load Database

rssi_db = RSSI_Database(n_beams=p['n_beams'], n_subbands=p['n_subbands'],
                        bandwidth=p['bandwidth'], n_ant=p['Nt_h'],
                        save_folder=p['db_save_folder'])
rssi_db.load()

#%% Plot Coverage maps for the whole database

for beam_idx in range(rssi_db.n_beams):
    for subband_idx in range(rssi_db.n_subbands):
        beam_dir = rssi_db.get_ang_of_beam(beam_idx)
        subband_freq = rssi_db.get_freq_of_subband(subband_idx) / 1e9 # [GHz]
        title = f'Beam = {beam_idx} ({beam_dir:.1f}º) | Subband = {subband_idx} ({subband_freq:.3f} GHz)'
        rssi_db.plot_coverage_map(title=title, beam_idx=beam_idx, subband_idx=subband_idx)
        break
    # break

#%% Plot Best beam in complete database

rssi_db.plot_best_beam(subband_idx=0)

#%% Plot probability functions in 2D
msr = -30
std = 3 # assumed STD != REAL WORLD STD!

for msr in np.arange(-140,-30+1)[::-1]: # probabilities from measurements [-30 to -140] dB
# for std in np.arange(1,10+1): # probabilities from increasing standard deviations
    prob_grid = rssi_db.compute_prob_grid(msr, std, beam_idx=11, subband_idx=0)
    rssi_db.plot_prob_grid(prob_grid, title=f"Measurement = {msr} dBm | Assumed $\sigma = {std}$")

#%% Generate Measurements and plot 1000 of them in time

n_samp = 10000
m_gen = RSSI_Measurement_Generator(db=rssi_db, std_def=2)

K = [3]
B = [3]
T = [i for i in range(n_samp)]

meas_list = m_gen.gen_measurement(K, B, T, pos=[0,0,0])

x_range = np.arange(n_samp)
m_vals = [m.rssi for m in meas_list]
mean_val = np.mean(m_vals)
f, ax = plt.subplots(1, 2, dpi=200, figsize=(10,6))

ax[0].set_title('Values of individual measurements')
ax[0].scatter(x_range, m_vals, s=3, label='Measurements')
ax[0].hlines(mean_val, xmin=0, xmax=n_samp, colors='orange', lw=2, label='Mean')
ax[0].set_xlabel('Measurement index')
ax[0].set_ylabel('RSSI value [dBm]')
ax[0].legend(loc='upper right', scatterpoints=5)

ax[1].set_title('Histogram of measurements')
n_samp_per_bin, *_ = ax[1].hist(m_vals, bins=20, label='Histogram')
ax[1].vlines(mean_val, ymin=0, ymax=np.max(n_samp_per_bin), colors='orange', lw=2, label='Mean')
ax[1].set_xlabel('RSSI value [dBm]')
ax[1].set_ylabel('Count')
ax[1].legend(loc='upper right')

#%% Plot best beam for each position

rssi_db.plot_best_beam(subband_idx='all')

#%% [Predetermined Measurements] Plot and Intersect probability grids for different beams/bands/times

K = [12]
B = [0] #[0, 10, 19]
T = [i for i in range(5)]

m_gen2 = RSSI_Measurement_Generator(db=rssi_db, std_def=.2)

m_gen2.gen_measurement(K, B, T, pos=[0,0,0])
prob_grid = m_gen2.comp_loc_prob(plot=True)
pos_estimate = m_gen2.estimate_loc(plot=True)
pos_error = m_gen2.comp_pos_error()
m_gen2.plot_final_result()

#%% [Real Measurements] 

NK = 3 # these are ignored in gen_real_beam_meas()
B = [0] #[0, 10, 19]
T = [1]# i for i in range(5)]

m3 = RSSI_Measurement_Generator(db=rssi_db, std_def=.2)
meas3 = m3.gen_real_beam_meas(NK, B, T, pos=[22,-25,0])
prob_grid3 = m3.comp_loc_prob(plot=True)
pos_estimate = m3.estimate_loc(plot=True)
pos_error = m3.comp_pos_error(verbose=True)

m3.plot_final_result()

#%% Test NK, NB, and NT

N_rep = 100 # repetitions of each meas
p_idx = rssi_db.get_closest_pos_idx([0,0,0])
# p_idx = rssi_db.get_closest_pos_idx([23,-25,0])

# N = 25
# # Beam Variation
# params_combo = [
#     {'NK': nk,
#       'B': [0],
#       'T': [i+1 for i in range(10)],
#       'pos': rssi_db.rx_pos[p_idx],
    
#     } for nk in range(1,N+1)]

N = 100
# # NT variation
params_combo = [
    {'NK': 1,
      'B': [0],
      'T': [i+1 for i in range(nt)],
      'pos': rssi_db.rx_pos[p_idx],
    
    } for nt in range(1,N+1)]


n_param_combos = len(params_combo)

base_results_shape = [N_rep, n_param_combos]
pos_errors = np.zeros(base_results_shape)


t = time.time()
for rep_idx in tqdm(range(N_rep), desc='Repeating experiment'):
    for combo_idx in range(n_param_combos):
        pos_errors[rep_idx, combo_idx] = gu.make_experiment(params_combo[combo_idx], 
                                                            rssi_db, std_def=2)

print(f'\nTotal time consumed = {time.time() - t:.2f}s')


avg_pos_err = np.mean(pos_errors, axis=0)
avg_pos_std = np.std(pos_errors, axis=0)

confidence = 0.90
z_score = {0.70: 1.040, 0.75: 1.15, 0.80: 1.28, 0.85: 1.44, 
           0.90: 1.645, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}[confidence]
y_err = z_score * avg_pos_std / np.sqrt(N_rep)

x = np.arange(1,N+1)
y = avg_pos_err

plt.figure(dpi=200)
plt.plot(x, y)
plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)
plt.ylabel('Position Error [m]')
plt.xlabel('Number of Samples in Time')
# plt.xlabel('Number of Beams Reported')
plt.title(f"Position Error vs Time Samples (NT) \n"
            f"DB resolution = {p['cell_size']} m | N_rep = {N_rep}")
# plt.title(f"Position Error vs Number of Beams (NK) \n"
#           f"DB resolution = {p['cell_size']} m | N_rep = {N_rep} | NT = {params_combo[-1]['T'][-1]}")
plt.grid()

#%% Minimum theoretical error
closest_cell = rssi_db.rx_pos[rssi_db.get_closest_pos_idx(rssi_db.rx_pos[p_idx])]

min_theoretic_err = np.linalg.norm(rssi_db.rx_pos[p_idx] - closest_cell)

print(f'closest_cell = {closest_cell}')
print(f'min_theoretic_err = {min_theoretic_err:.3f}')

#%% Position accuracy for all positions in 2D grid

N_rep = 10 # repetitions of each meas
params_combo = [
    {'NK': 1,
     'B': [0],
     'T': [1], #i+1 for i in range(5)],
     'pos': rssi_db.rx_pos[p_idx],
    
    } for p_idx in range(rssi_db.rx_pos.shape[0])]

n_param_combos = len(params_combo)

base_results_shape = [N_rep, n_param_combos]
pos_errors = np.zeros(base_results_shape)

t = time.time()
for combo_idx in tqdm(range(n_param_combos), desc='Repeating for different Positions'):
    for rep_idx in range(N_rep):
        pos_errors[rep_idx, combo_idx] = gu.make_experiment(params_combo[combo_idx], rssi_db)

print(f'\nTotal time consumed = {time.time() - t:.2f}s')

avg_pos_err = np.mean(pos_errors, axis=0)
avg_pos_std = np.std(pos_errors, axis=0)

#%% Plot the 2D average accuracy

# invert the colormap by doing - pos
title = f"Position Error [DB res = {p['cell_size']} m | N_rep = {N_rep}]"
rssi_db.plot_coverage_map(matrix=avg_pos_err, #title=title, 
                          cm_label='Absolute Position Error [m]', 
                          scatter=True, convert_to_2D=False, lims=False, scat_sz=6.8,
                          rx_pos=[[0,0,0], [23,-25,0]], rx_labels=['Pos1 (LoS)', 'Pos2 (NLoS)'],
                          legend=True, dpi=600)

#%% Plot Received Power in Best beam (for band 0)

title = f"Received Power in Best Beam [DB res = {p['cell_size']} m]"
rssi_db.plot_coverage_map(matrix=np.max(rssi_db.dbm[:,0,:], axis=0), title=title, lims=False, 
                          scatter=False, convert_to_2D=True,
                          cm_label='Received Power in Best Beam [dBm]')

#%% Plot Pos Error vs received power (which is a function of LoS/NLoS and distance)

# for each position, get the received power (db) in best beam in subband 0
plt.figure(dpi=300)
x = np.max(rssi_db.dbm[:,0,:], axis=0)
y = avg_pos_err
plt.scatter(x, y, s=5, lw=.3, edgecolors='k', zorder=3)
plt.xlabel('Received Power [dBm]')
plt.ylabel('Localization Error [m]')
plt.title('Relation between received power and location error')
plt.grid()

print(f'Correlation coeff = {np.corrcoef(x,y)[0,1]:.2f}')

#%% Make Multi-parameter combination experiment

N_rep = 50 # repetitions of each meas
near_pos = [0,0,1.5]
near_pos = [23,-25,1.5]

np.random.seed(1)

params_base = {'NK': 1,
               'B': [0],
               'T': [1],
               'pos': near_pos,
               }

csvs_folder = 'csvs'
os.makedirs(csvs_folder, exist_ok=True)
csv_path = f"{csvs_folder}/res_N_rep={N_rep}_pos={params_base['pos']}_t={time.time():.0f}.csv" 

N_VARS = 6 # number of values of each parameter to test below (easier to hardcode)
# NT variation
params_variations = {'NK': [i for i in [2,3,4,5,8,12]],
                     'B':  [[i  for i in range(nb)] for nb in [2,4,8,12,16,20]],
                     'T':  [[i  for i in range(nt)] for nt in [4,8,12,16,20,24]],
                     }

params_combos = gu.build_params_combinations(params_base, params_variations, n_vars=N_VARS)
n_param_combos = len(params_combos)

base_results_shape = [N_rep, n_param_combos]
pos_errors = np.zeros(base_results_shape)

t = time.time()
for rep_idx in tqdm(range(N_rep), desc='Repeating experiment'):
    for combo_idx in range(n_param_combos):
        pos_errors[rep_idx, combo_idx] = gu.make_experiment(params_combos[combo_idx], rssi_db)

print(f'\nTotal time consumed = {time.time() - t:.2f}s')

# Write results to CSV
df = pd.DataFrame()

df['combo_idx'] = np.arange(n_param_combos)

var_change_list = [[key]*len(params_variations[key]) for key in params_variations.keys()]
var_change_list_flat = [''] + [item for sublist in var_change_list for item in sublist] + ['all'] * N_VARS
df['variable'] = var_change_list_flat

df['avg'] = np.mean(pos_errors, axis=0)
df['std'] = np.std(pos_errors, axis=0)

# Confidence intervals
z_score_dict = {0.70: 1.040, 0.75: 1.15, 0.80: 1.28, 0.85: 1.44, 
                0.90: 1.645, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}
for val in [0.8, 0.9, 0.95, 0.99]:
    df[f'conf{int(val*100)}'] = z_score_dict[val] * df['avg']  / np.sqrt(N_rep)

# Cumulative distribution
sorted_errors = np.sort(pos_errors, axis=0)
for val in [0.8, 0.9, 0.95, 0.99]:
    df[f'cum{int(val*100)}'] = sorted_errors[int(len(sorted_errors) * val)]

df.to_csv(csv_path, index=False)

#%% Plot Multi-parameter combos

csv_path = '/home/joao/Documents/GitHub/SionnaProjects/Loc-DT-paper-repo/csvs/res_N_rep=500_pos=[0, 0, 1.5].csv'
df = pd.read_csv(csv_path)

plt.figure(dpi=200, figsize=[6, 4])
# var_names = ['B', 'T', 'NK', 'all'] # choose the order of plotting
var_names = ['NK', 'B', 'T', 'all']

var_plot_params = {'B': {'marker': 'D', 'markersize': 5, 'label': '#$\mathcal{B}$', 'color': 'tab:blue'},
                   'T': {'marker': '^', 'markersize': 7, 'label': '#$\mathcal{T}$', 'color': 'tab:orange'},
                   'NK': {'marker': 's', 'markersize': 5, 'label': '#$\mathcal{K}$', 'color': 'tab:green'},
                   'all': {'marker': '*', 'markersize': 9, 'label': 'All', 'color': 'tab:red'},
                   }

def_plot_params = {'markerfacecolor': 'w'}

# text parameters
x_off = 0.05
y_off = 0.1 if near_pos[0] == 0 else 0.5
txt_labels = {'B':'B', 'T':'T', 'NK': 'K'}
txt_flip_y = {'B': [0,0,0,0,0,0],
              'T': [0,0,0,0,0,0],
              'NK': [1,1,0,0,0,0],
              'all': [0,0,0,0,0,0,0] if near_pos[0] == 0 else [0,1,0,0,0,0,0]}
skip_joint_text = False

for var_name in var_names:
    data = np.array([df['avg'][0]] + list(df['avg'][df['variable'] == var_name]))
    # print(f'var = {var_name}; data = {data}')
    plt.plot(data, **def_plot_params, **var_plot_params[var_name])
    
    conf = 90
    conf_int = np.array([0] + df[f'conf{conf}'][df['variable'] == var_name].tolist())
    plt.fill_between(np.arange(N_VARS+1), data - conf_int, data + conf_int, alpha=0.2,
                     facecolor=var_plot_params[var_name]['color'])
    
    if var_name != var_names[-1]:
        for i, y_val in enumerate(data[1:]):
            if type(params_variations[var_name][i]) in [list]:
                text = len(params_variations[var_name][i])
            else:
                text = params_variations[var_name][i]
            text = f'{txt_labels[var_name]}={text}'
            va = 'bottom' if not txt_flip_y[var_name][i] else 'top'
            y = y_val + y_off if not txt_flip_y[var_name][i] else y_val - 2*y_off
            plt.text(i+1+x_off, y, text, fontsize=8, verticalalignment=va)
    
    if var_name == var_names[-1] and not skip_joint_text:
        for i in range(7 if near_pos[0] == 0 else 5):
            txt_ele = []
            for var_n in var_names[:-1]:
                val = params_base[var_n] if i == 0 else params_variations[var_n][i-1]
                text = len(val) if type(val) in [list] else val
                txt_ele.append(str(text))
                
            text = f"({','.join(txt_ele)})"
            
            if i == 0:
                text = '(K,B,T)\n' + text
            
            va = 'bottom' if not txt_flip_y[var_name][i] else 'top'
            y = data[i] + 2*y_off if not txt_flip_y[var_name][i] else data[i] - 2*y_off
            
            plt.text(i, y, text, fontsize=8, 
                     horizontalalignment='center', verticalalignment=va)
                
    plt.xlim([-0.3, N_VARS + 0.56])
    plt.ylim([-0., df['avg'][0]*1.05 if skip_joint_text else df['avg'][0]*1.12])
    
plt.ylabel('Position error for user in LoS (m)'
           if near_pos[0] == 0 else 'Position error for user in NLoS (m)')
plt.xlabel('Reporting parameters combinations')
plt.legend(loc='upper right', ncols=10, columnspacing=0.8)
plt.grid()



