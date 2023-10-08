#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:18:26 2023

@author: joao
"""
import copy
import numpy as np

from RSSI_db import RSSI_Measurement_Generator

############################# Beamforming Functions ###########################

def get_beam_angles(fov, n_beams=None, beam_res=None):
    """
    3 ways of computing beam angles:
        1- given the codebook size (n_beams) and fov --- compute resolution
        2- given codebook size and resolution        --- computes range (not done)
        3- given range and resolution                --- computes codebook size
    """
    
    if n_beams:
        angs = np.linspace(-fov/2, fov/2, n_beams)
    elif beam_res:
        angs = np.arange(-fov/2, fov/2+.001, beam_res)
    else:
        raise Exception('Not enough information to compute beam angles.')
    
    return angs

def get_steering_vec_ULA(ang, n_ele):
    """ 2D steering vector, uses only azimuth"""
    steering_angle = ang*np.pi/180. # radian
    ele_range = np.arange(n_ele)
    steering_vec = np.exp(1j*2.*np.pi*ele_range*0.5*np.sin(steering_angle))
    steering_vec_normalized = steering_vec / np.linalg.norm(steering_vec)
    
    return steering_vec_normalized


####################### Database experiments functinons #######################

def make_experiment(params_combo, db, std_def=2):
    NK = params_combo['NK']
    B = params_combo['B']
    T = params_combo['T']
    pos = params_combo['pos']
    
    m = RSSI_Measurement_Generator(db, std_def=std_def)
    m.gen_real_beam_meas(NK, B, T, pos, verbose=False)
    m.comp_loc_prob(plot=False, verbose=False)
    m.estimate_loc(plot=False)
    pos_err = m.comp_pos_error(verbose=False)
    
    return pos_err


def build_params_combinations(params_base, params_variations, include_base=True, 
                              include_all=True, n_vars=6):
    params_combos = []
    if include_base:
        params_combos.append(params_base)
    
    for par_key in params_variations.keys():
        # print(par_key)
        for variation in params_variations[par_key]:
            new_combo = copy.copy(params_base)
            new_combo[par_key] = variation
            # print(new_combo)
            params_combos.append(new_combo)
    
    if include_all:
        # Note: they must have the same number of variations
        for var_idx in range(n_vars):
            new_combo = copy.copy(params_base)
            for key in params_variations.keys():
                new_combo[key] = params_variations[key][var_idx]
            # print(new_combo)
            params_combos.append(new_combo)
    
    return params_combos