#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:01:09 2023

@author: joao

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
     'freq': 3.5e9,
     'tx_pos': np.array([-42., 27., 32.]),
     'tx_ori': [0,0,-45],
     
      # Parameters for database generation (from the ray tracing simulation)
     'bandwidth': 100e6,    # [Hz]
     'subband_size': 5e6,   # [Hz]
     'n_subbands': 20,      # = int(bandwith / subband_size)
     
     'fov': 120,            # [º] 
     'beam_res': 5,         # [º] angle difference between consecutive beams
     'n_beams': 25,         # = len(np.arange(fov/2, fov/2+.01, beam_res)
     
     'Nt_h': 64,            # TX antenna horizontal elements 
     'Nt_v': 1,             # TX antenna vertical elements 
     
     'db_n_decimals': 1,    # Number of decimal places for the DB (values in dBm)
     
     'cell_size': 2, # [m]
     
     # Specific: DeepMIMO
      'scenario': 'new_scen',
     'scenarios_folder': 'deepmimo_scenarios',
     'db_save_folder': 'databases',
     
     }

LOS_POS = [-0.1, 19.9, 2]
NLOS_POS = [23.9, -10.1, 2]

#%% Load the default parameters
parameters = DeepMIMO.default_params()

parameters['scenario'] = p['scenario']
parameters['user_row_last'] = 141
parameters['bs_antenna']['shape'] = np.array([1, p['Nt_h'], 1])
parameters['bs_antenna']['rotation'] =  np.array(p['tx_ori'])
parameters['ue_antenna']['shape'] = np.array([1, 1, 1])
parameters['enable_BS2BS'] = False
parameters['activate_OFDM'] = 1
parameters['OFDM']['subcarriers'] = int(p['bandwidth']/1e5) # rule of thumb: 1 subcarrier/100 khz
parameters['OFDM']['subcarriers_limit'] = parameters['OFDM']['subcarriers']
parameters['bandwidth'] = p['bandwidth']/1e9 # [GHz]

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = p['scenarios_folder']

ant_shape = np.array([1, p['Nt_h'], 1])
oversampling_rate = np.array([1, 1, 1])

beam_angles = gu.get_beam_angles(p['fov'], beam_res=p['beam_res'])

F1 = np.array([gu.get_steering_vec_ULA(ang, n_ele=p['Nt_h']) for ang in beam_angles])

#%% Generate dataset

dataset_i = DeepMIMO.generate_data(parameters)

dataset_i[0]['location'] = p['tx_pos']

# Subsample dataset
uniform_subsampling = True

# 2 = half the samples, 3 = a third, etc.. along [x,y]
sampling_div = [p['cell_size'],p['cell_size']] 
n_rows = parameters['user_row_last'] - parameters['user_row_first'] + 1
n_usr_row = 181 # n_cols = 595 for Boston, 411 for asu campus, = 181 for new_scen

if uniform_subsampling:
    
    cols = np.arange(n_usr_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i*n_usr_row for i in rows for j in cols])
    dataset_u = []
    n_bs = len(dataset_i)
    
    for bs_idx in range(n_bs):
        dataset_u.append({})
        for key in dataset_i[bs_idx].keys():
            dataset_u[bs_idx]['location'] = dataset_i[bs_idx]['location']
            dataset_u[bs_idx]['user'] = {}
            for key in dataset_i[bs_idx]['user']:
                dataset_u[bs_idx]['user'][key] = dataset_i[bs_idx]['user'][key][uniform_idxs]
    
dataset = dataset_u if uniform_subsampling else dataset_i


# filter some users with errors (problems in the ray tracing)
def get_idxs_in_xy_box(data_pos, x_min, x_max, y_min, y_max, only_non_nan=False):

    idxs_x = np.where((x_min < data_pos[:, 0]) & (data_pos[:, 0] < x_max))[0]
    idxs_y = np.where((y_min < data_pos[:, 1]) & (data_pos[:, 1] < y_max))[0]
    
    return np.array(list(set(idxs_x).intersection(idxs_y)))

not_idxs = get_idxs_in_xy_box(dataset[0]['user']['location'], x_min=30, x_max=50, y_min=-50, y_max=-30)

grid_idxs_enabled = np.where(dataset[0]['user']['LoS'] != -1)[0]  # enabled UEs
grid_idxs_enabled = np.array([i for i in grid_idxs_enabled if i not in not_idxs])

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

# rssi_db = RSSI_Database(n_beams=p['n_beams'], n_subbands=p['n_subbands'],
#                         bandwidth=p['bandwidth'], n_ant=p['Nt_h'],
#                         save_folder=p['db_save_folder'])
# rssi_db.load()

#%% Plot Coverage maps for the whole database

for beam_idx in range(rssi_db.n_beams):
    for subband_idx in range(rssi_db.n_subbands):
        beam_dir = rssi_db.get_ang_of_beam(beam_idx)
        subband_freq = rssi_db.get_freq_of_subband(subband_idx) / 1e9 # [GHz]
        title = f'Beam = {beam_idx} ({beam_dir:.1f}º) | Subband = {subband_idx} ({subband_freq:.3f} GHz)'
        rssi_db.plot_coverage_map(title=title, beam_idx=beam_idx, subband_idx=subband_idx)
        # break
    break

#%% Plot Best beam in complete database

rssi_db.plot_best_beam(subband_idx=0)

#%% Plot Received Power in Best beam (for band 0)

rssi_db.plot_coverage_map(matrix=np.max(rssi_db.dbm[:,0,:], axis=0), lims=False,
                          title=f"Received Power in Best Beam [DB res = {p['cell_size']} m]",
                          cm_label='Received Power in Best Beam [dBm]')

#%% (testing) Generate Measurements and plot 1000 of them in time

n_samp = 10000
m_gen = RSSI_Measurement_Generator(db=rssi_db, std_def=2)

K = [3]
B = [3]
T = [i for i in range(n_samp)]

meas_list = m_gen.gen_measurement(K, B, T, pos=LOS_POS)

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

#%% (testing) [Predetermined Measurements] Plot and Intersect probability grids for different beams/bands/times

K = [12]
B = [0] #[0, 10, 19]
T = [i for i in range(5)]

m_gen2 = RSSI_Measurement_Generator(db=rssi_db, std_def=1)

m_gen2.gen_measurement(K, B, T, pos=LOS_POS)
prob_grid = m_gen2.comp_loc_prob(plot=True)
pos_estimate = m_gen2.estimate_loc(plot=True)
pos_error = m_gen2.comp_pos_error()
m_gen2.plot_final_result()

#%% (testing) [Real Measurements] 

NK = 3 # these are ignored in gen_real_beam_meas()
B = [0] #[0, 10, 19]
T = [0]# for i in range(12)]

m3 = RSSI_Measurement_Generator(db=rssi_db, std_def=2)
meas3 = m3.gen_real_beam_meas(NK, B, T, pos=LOS_POS)
prob_grid3 = m3.comp_loc_prob(plot=True)
pos_estimate = m3.estimate_loc(plot=True)
pos_error = m3.comp_pos_error(verbose=True)

m3.plot_final_result()

#%% Position accuracy for all positions in 2D grid

N_rep = 10 # repetitions of each meas
params_combo = [
    {'NK': 1,
     'B': [0],
     'T': [1], #i+1 for i in range(5)],
     'pos': pos,
    
    } for pos in rssi_db.rx_pos]

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
                          rx_pos=[LOS_POS, NLOS_POS], rx_labels=['Pos1 (LoS)', 'Pos2 (NLoS)'],
                          legend=True, dpi=300)

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

N_rep = 100 # repetitions of each meas
los = 1
np.random.seed(1)

params_base = {
    'NK': 1 if los else 2,
    'B': [0],
    'T': [1],
    'pos': LOS_POS if los else NLOS_POS,
    }

csvs_folder = 'csvs'
os.makedirs(csvs_folder, exist_ok=True)
csv_path = f"{csvs_folder}/res_N_rep={N_rep}_pos={params_base['pos']}_t={time.time():.0f}.csv" 

N_VARS = 6 # number of values of each parameter to test below (easier to hardcode)
params_variations = {'NK': [i for i in ([2, 4, 6, 8, 10, 12] if los else [3,4,6,8,10,12])],
                      'B': ([[0,19], [0,6,12,19], [0,4,8,12,16,19], 
                            [0,3,6,9,12,15,17,19], [0,2,4,6,8,10,12,14,16,19],
                            [0,2,3,4,6,8,9,10,12,14,16,19]] if los else
                            [[i for i in range(nb)] for nb in [2,4,6,8,10,12]]),
                      'T':  [[i  for i in range(nt)] for nt in [2,4,6,8,10,12]],
                      }

params_combos = gu.build_params_combinations(params_base, params_variations, n_vars=N_VARS)
n_param_combos = len(params_combos)

base_results_shape = [N_rep, n_param_combos]
pos_errors = np.zeros(base_results_shape)

t = time.time()
for rep_idx in tqdm(range(N_rep), desc='Repeating experiment'):
    for combo_idx in range(n_param_combos):
        pos_errors[rep_idx, combo_idx] = gu.make_experiment(params_combos[combo_idx], rssi_db,
                                                            std_def=2)

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
    df[f'conf{int(val*100)}'] = z_score_dict[val] * df['std']  / np.sqrt(N_rep)

# Cumulative distribution
sorted_errors = np.sort(pos_errors, axis=0)
for val in [0.8, 0.9, 0.95, 0.99]:
    df[f'cum{int(val*100)}'] = sorted_errors[int(len(sorted_errors) * val)]

df.to_csv(csv_path, index=False)

#%% Plot Multi-parameter combos

csv_path = 'csvs/res_N_rep=500_pos=[-0.1 19.9  2. ]_t=1710119053.csv'
# csv_path = 'csvs/res_N_rep=500_pos=[ 23.9 -10.1   2. ]_t=1710119181.csv'
df = pd.read_csv(csv_path)

plt.figure(dpi=200, figsize=[6, 4])
var_names = ['NK', 'B', 'T', 'all']

var_plot_params = {'B': {'marker': 'D', 'markersize': 5, 'label': r'Number of Subbands $|\mathcal{B}|$', 'color': 'tab:blue'},
                   'T': {'marker': '^', 'markersize': 7, 'label': r'Number of Times $|\mathcal{T}|$', 'color': 'tab:orange'},
                   'NK': {'marker': 's', 'markersize': 5, 'label': r'Number of Beams $|\mathcal{K}|$', 'color': 'tab:green'},
                   'all': {'marker': '*', 'markersize': 9, 'label': 'All', 'color': 'tab:red'},
                   }

def_plot_params = {'markerfacecolor': 'w'}

# text parameters
x_off = -0.05
y_off = 0.2 if los else 0.7
txt_labels = {'B':'B', 'T':'T', 'NK': 'K'}
txt_flip_y = {'B': [0,1,0,0,0,0] if los else [0,0,0,1,1,1,1],
              'T': [0,0,0,0,0,0] if los else [1,1,1,0,0,0,0],
              'NK': [0,0,0,0,0,0] if los else [0,0,0,0,0,0,0],
              'all': [0,0,0,0,0,0,0] if los else [0,0,0,0,0,0,0]}
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
                text_i = len(params_variations[var_name][i])
            else:
                text_i = params_variations[var_name][i]
            
            # text = f'{txt_labels[var_name]}={text_i}'
            if txt_labels[var_name] == 'K': 
                lab_text = r'|$\mathcal{K}|$='
            elif txt_labels[var_name] == 'B':
                 lab_text = r'|$\mathcal{B}|$='
            elif txt_labels[var_name] == 'T':
                lab_text = r'|$\mathcal{T}|$='
            else:
                print('smth wrong')#lab_text = f'{txt_labels[var_name]}={text}'
            text = lab_text + f'{text_i}'
            
            va = 'bottom' if not txt_flip_y[var_name][i] else 'top'
            y = y_val + y_off if not txt_flip_y[var_name][i] else y_val - 2*y_off
            plt.text(i+1+x_off, y, text, fontsize=8, verticalalignment=va)
    
    if var_name == var_names[-1] and not skip_joint_text:
        for i in range(7 if los else 4):
            txt_ele = []
            for var_n in var_names[:-1]:
                val = params_base[var_n] if i == 0 else params_variations[var_n][i-1]
                text = len(val) if type(val) in [list] else val
                txt_ele.append(str(text))
                
            text = f"({','.join(txt_ele)})"
                
            if i == 0:
                # text = '(K,B,T)\n' + text
                text = r'$(|\mathcal{K}|,|\mathcal{B}|,|\mathcal{T}|)$' + '\n' + text
                
            
            va = 'bottom' if not txt_flip_y[var_name][i] else 'top'
            y = data[i] + 2*y_off if not txt_flip_y[var_name][i] else data[i] - 2*y_off
            
            plt.text(i, y, text, fontsize=8, 
                     horizontalalignment='center', verticalalignment=va)
                
    plt.xlim([-0.5, N_VARS + 0.56])
    plt.ylim([-0., df['avg'][0]*1.08 if skip_joint_text else df['avg'][0]*1.15])
    
plt.ylabel(f"Mean position error for a {'' if los else 'N'}LoS user (m)")
plt.xlabel('Overhead level')
plt.legend(ncols=1, columnspacing=0.8)
plt.grid()
# plt.savefig("4.svg")