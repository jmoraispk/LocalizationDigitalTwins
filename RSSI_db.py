#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:33:31 2023

@author: joao
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm # for the cdf of a normal distribution


class RSSI_Database():
    def __init__(self, **kwargs):
        self.init_vars()
        self.add_info(**kwargs)

    def init_vars(self):

        # RSSI matrix of the database
        self.dbm = None         # [n_beams, n_subbands, n_rx] in dBm

        self.n_beams = None
        self.n_ant = None
        self.beam_angles = None
        self.beamforming_vectors = None

        self.n_subbands = None
        self.center_freq = None
        self.bandwidth = None
        self.subband_size = None

        self.n_rx = None
        self.rx_pos = None     # [n_rx, 3]
        self.grid_dims = None  # list of 3 elements
        self.grid_idxs_enabled = None # list of where the rx_pos are in grid_dims
        self.real_grid_dims = None # TODO: is this necessary beyond scatter = False???

        self.scatter = False # MODE # TODO: revise the need for this!!!!
        # scatter = True: the db needs the full grid with nans
        # scatter = False: the db only needs the dbm matrix and the positions

        self.tx_pos = None  # [x,y,z] in meters
        self.tx_ori = None  # [around x, around y, around z] in radians
        
        self.save_folder = None
        
    def add_info(self, **kwargs):

        if 'dbm' in kwargs:
            self.dbm = kwargs['dbm']
            self.dbm_min = self.dbm.min()
            self.dbm_max = self.dbm.max()
        
        if 'n_beams' in kwargs:
            self.n_beams = kwargs['n_beams']

        if 'n_subbands' in kwargs:
            self.n_subbands = kwargs['n_subbands']

        if 'n_ant' in kwargs:
            self.n_ant = kwargs['n_ant']
            if self.beamforming_vectors is not None:
                if self.n_ant != len(self.beamforming_vectors[0]):
                    raise Exception(
                        'Incompatible information! n_ant vs beamforming_vectors')

        if 'beam_angles' in kwargs:
            self.beam_angles = kwargs['beam_angles']
            if self.n_beams != len(self.beam_angles):
                raise Exception(
                    'Incompatible information! n_beams vs beam_angles')

        if 'beam_vectors' in kwargs:
            self.beamforming_vectors = kwargs['beam_vectors']
            self.n_ant = len(self.beamforming_vectors[0])
            if self.n_beams != len(self.beamforming_vectors):
                raise Exception(
                    'Incompatible information! n_beams vs beamforming_vectors')

        if 'fc' in kwargs or 'center_freq' in kwargs:
            self.center_freq = kwargs['fc'] if 'fc' in kwargs else kwargs['center_freq']

        if 'b' in kwargs or 'bandwidth' in kwargs:
            self.bandwidth = kwargs['b'] if 'b' in kwargs else kwargs['bandwidth']

        if 'subband_size' in kwargs:
            self.subband_size = kwargs['subband_size']

        if self.bandwidth:
            if self.subband_size:
                if self.n_subbands != int(self.bandwidth / self.subband_size):
                    raise Exception('Incompatible information! n_subbands vs (subband_size or bandwidth)')
            else:
                self.subband_size = self.bandwidth / self.n_subbands

        if 'rx_pos' in kwargs:
            self.rx_pos = kwargs['rx_pos']
            if self.n_rx is None:
                self.n_rx = len(self.rx_pos)
            elif self.n_rx != len(self.rx_pos):
                raise Exception('Incompatible information! n_rx and rx_pos')

        if 'grid_dims' in kwargs:
            self.grid_dims = kwargs['grid_dims']
            self.compute_real_dims()

        if 'grid_idxs_enabled' in kwargs:
            self.grid_idxs_enabled = kwargs['grid_idxs_enabled']

            if self.dbm is not None:
                if self.dbm.shape[-1] != len(self.grid_idxs_enabled):
                    raise Exception('Incompatible information! dbm and enabled grid idxs')

        if 'scatter' in kwargs:
            self.scatter = kwargs['scatter']
            #self.scat_sz = 6 # TODO: derive from the resolution of the db. 6 for 2 m
            # TODO: derive resolution as well (we don't require as input because the db might be non-uniform)
        
        if 'tx_pos' in kwargs:
            self.tx_pos = kwargs['tx_pos']

        if 'tx_ori' in kwargs:
            self.tx_ori = kwargs['tx_ori']
        
        if 'save_folder' in kwargs:
            self.save_folder = kwargs['save_folder']
             
    def convert_to_2D(self, mat_1d):
        
        mat_2d_flat = np.zeros(self.grid_dims.prod()) * np.nan
        mat_2d_flat[self.grid_idxs_enabled] = mat_1d
        
        return mat_2d_flat.reshape(self.grid_dims, order='F')
    
    def plot_coverage_map(self, matrix=None, lims=True, tx_pos=True, tx_ori=True, title='',
                          beam_idx=0, subband_idx=0, rx_pos=None, rx_labels=None, legend=False, 
                          cm_label='', scat_sz=6, convert_to_2D=None, scatter=None, dpi=200,
                          tight=False):
        if self.dbm is None and matrix is None:
            raise Exception('No available grid to plot')
        
        if matrix is None:
            mat_to_plt = self.dbm[beam_idx, subband_idx]
        else:
            mat_to_plt = matrix
        
        if scatter is None:
            scatter = self.scatter
            
        if convert_to_2D is None and not scatter:
            convert_to_2D = True
        
        mat_to_plt = (mat_to_plt if not convert_to_2D else self.convert_to_2D(mat_to_plt).T)
        
        lims_dict = {} if not lims else {'vmin': self.dbm_min, 'vmax': self.dbm_max}
        
        fig, ax = plt.subplots(dpi=dpi)
        if scatter:
            im = plt.scatter(self.rx_pos[:,0], self.rx_pos[:,1], c=mat_to_plt, 
                             s=scat_sz, marker='s', **lims_dict)
        else:
            im = plt.imshow(mat_to_plt, origin='lower', **lims_dict)

        plt.colorbar(im, label='Received Power [dBm]' if not cm_label else cm_label)
        
        plt.xlabel('x (m)' if scatter else 'User index (X-axis)')
        plt.ylabel('y (m)' if scatter else 'User index (Y-axis)')
        
        # TX position
        tx_pos_plt = None
        if type(tx_pos) in [list, np.ndarray]:
            tx_pos_plt = tx_pos if scatter else self.transform_coords_to_cells(tx_pos)
        elif tx_ori:
            tx_pos_plt = self.tx_pos if scatter else self.transform_coords_to_cells(self.tx_pos)
        
        if tx_pos_plt is not None:
            ax.scatter(tx_pos_plt[0], tx_pos_plt[1], marker='P', c='r', label='TX')
        
        # TX orientation
        tx_lookat = None
        r = 20
        if type(tx_ori) in [list, np.ndarray]:
            tx_lookat = tx_pos[:2] + r * np.array([np.cos(tx_ori[2]), np.sin(tx_ori[2])])
        elif tx_pos_plt is not None and self.tx_ori is not None and tx_ori:
            tx_lookat = self.tx_pos[:2] + r * np.array([np.cos(self.tx_ori[2]), np.sin(self.tx_ori[2])])
            
        if tx_lookat is not None and tx_pos_plt is not None:
            tx_lookat_plt = tx_lookat if scatter else self.transform_coords_to_cells(tx_lookat)
            ax.plot([tx_pos_plt[0], tx_lookat_plt[0]],
                    [tx_pos_plt[1], tx_lookat_plt[1]], c='k', alpha=.5)
        
        # RX positions
        if rx_pos is not None:
            if type(rx_pos) == list and type(rx_pos[0]) in [np.ndarray, list]:
                rx_pos2 = rx_pos
            elif type(rx_pos) == np.ndarray and type(rx_pos[0]) == np.ndarray:
                rx_pos2 = rx_pos
            else:
                rx_pos2 = [rx_pos]
                
            colors = ['m', 'b', 'g', 'k']
            for rx_idx, rx_p in enumerate(rx_pos2):
                rx_pos_t = rx_p if scatter else self.transform_coords_to_cells(rx_p)
                ax.scatter(rx_pos_t[0], rx_pos_t[1], marker='o', c=colors[rx_idx],
                           s=9, label=f'RX {rx_idx}' if rx_labels is None else rx_labels[rx_idx])
        
        if title:
            plt.title(title)
        
        if legend:
            plt.legend(loc='upper center', ncols=10, framealpha=.5)
        
        if tight:
            s = 1
            plt.xlim(self.rx_pos[:, 0].min() - s, self.rx_pos[:, 0].max() + s)
            plt.ylim(self.rx_pos[:, 1].min() - s, self.rx_pos[:, 1].max() + s)
        
        return fig, ax

    def compute_real_dims(self):
        if self.rx_pos is None:
            print('Could not compute real dimensions because rx_pos is None.')
            
        x_dist = np.max(self.rx_pos[:, 0]) - np.min(self.rx_pos[:, 0])
        y_dist = np.max(self.rx_pos[:, 1]) - np.min(self.rx_pos[:, 1])
        self.real_grid_dims = [x_dist, y_dist]
        
        self.scale = self.real_grid_dims / self.grid_dims
    
    def transform_coords_to_cells(self, orig_pos):
        if self.real_grid_dims is None:
            self.compute_real_dims()
        
        return orig_pos[:2] / self.scale + self.grid_dims/2

    def transform_2d_cells_to_coords(self, cell_pos):
        if self.real_grid_dims is None:
            self.compute_real_dims()
        
        return (cell_pos[:2] - self.grid_dims/2) * self.scale
    
    def transform_1d_cells_to_coords(self, cell_idx):
        return self.rx_pos[cell_idx,:2]
    
    def compute_prob_grid(self, avg, std, slack=0.01, normalize=True,
                          beam_idx=0, subband_idx=0):
        """
        slack
            2 * slack is the width of the integration interval around the assumed
            probability density function.
            The default is 0.01.
        Returns
        -------
        TYPE
            Probability grid for each point in x.
        """
        p = (norm.cdf(self.dbm[beam_idx, subband_idx] + slack, loc=avg, scale=std) -
             norm.cdf(self.dbm[beam_idx, subband_idx]- slack, loc=avg, scale=std))

        # makes the sum of probabilities = 1
        norm_fact = np.nansum(p) if normalize else 1

        return p / norm_fact
        
    def get_rx_positions(self):
        return self.rx_pos
    
    def get_closest_pos_idx(self, pos):
        
        # Make np.array if it's a simple list
        if isinstance(pos, list):
            pos = np.array(pos)
        
        return np.argmin(np.linalg.norm(self.rx_pos - pos, axis=1))
    
    def get_n_closest_pos_idx(self, pos, n):
        
        # Make np.array if it's a simple list
        if isinstance(pos, list):
            pos = np.array(pos)
        
        # TODO: When this function comes in use, test more sorting from the post
        #       https://stackoverflow.com/a/38772601
        # dists = np.linalg.norm(self.rx_pos - pos, axis=1)
        # idxs = np.argpartition(dists, n-1)[:n]
        # min_elements = dists[idxs]
        # min_elements_order = np.argsort(min_elements)
        # ordered_indices = idxs[min_elements_order]
        
        return np.argsort(np.linalg.norm(self.rx_pos - pos, axis=1))[:n]
    
    def get_freq_of_subband(self, subband_idx):
    
        return self.center_freq - self.bandwidth/2 + self.subband_size * subband_idx
    
    def get_ang_of_beam(self, beam_idx):
        return self.beam_angles[beam_idx]

    def plot_best_beam(self, subband_idx=0):
        
        if type(subband_idx) == list:
            subband_idxs = subband_idx
        elif subband_idx == 'all':
            subband_idxs = [i for i in range(self.n_subbands)]
        else:
            subband_idxs = [subband_idx]
            
        for b_idx in subband_idxs:
            sorted_beams = np.argsort(self.dbm, axis=0)[-1,:] # last is highest
            
            title = f'Best Beam (Freq: {self.get_freq_of_subband(b_idx)/1e9:.3f} GHz)'
            self.plot_coverage_map(matrix=sorted_beams[b_idx, :], 
                                   cm_label='Best Beam index', lims=False,
                                   title=title)
            plt.show()
    
    def get_default_db_path(self):
        
        req_vars = [self.n_beams, self.n_subbands, self.bandwidth, self.n_ant]
        if None in req_vars:
            raise Exception('Some of the variables necessary to generate a path are *None*I')
        
        save_path = (f"rssi_db_nbeams={self.n_beams}_nsubbands={self.n_subbands}_"
                     f"B={self.bandwidth/1e6:.0f}MHz_Nt={self.n_ant}.p")
        
        return os.path.join(self.save_folder, save_path)
    
    def gen_vars_dict(self):
        vars_dict = {'dbm': self.dbm,
                     
                     'n_beams': self.n_beams,
                     'n_ant': self.n_ant,
                     'beam_angles': self.beam_angles,
                     'beamforming_vectors': self.beamforming_vectors,
                     
                     'n_subbands': self.n_subbands,
                     'center_freq': self.center_freq,
                     'bandwidth': self.bandwidth,
                     'subband_size': self.subband_size,
                     
                     'n_rx': self.n_rx,
                     'rx_pos': self.rx_pos,
                     'grid_dims': self.grid_dims,
                     'grid_idxs_enabled': self.grid_idxs_enabled,
                     'real_grid_dims': self.real_grid_dims,
                     
                     'scatter': self.scatter,
                     
                     'tx_pos': self.tx_pos,
                     'tx_ori': self.tx_ori,
                     
                     'save_folder': self.save_folder,
                     }
        
        return vars_dict
    
    def load_vars_dict(self, vars_dict):
        
        self.dbm = vars_dict['dbm']
        self.dbm_min = self.dbm.min()
        self.dbm_max = self.dbm.max()
        
        self.n_beams = vars_dict['n_beams']
        self.n_ant = vars_dict['n_ant']
        self.beam_angles = vars_dict['beam_angles']
        self.beamforming_vectors = vars_dict['beamforming_vectors']

        self.n_subbands = vars_dict['n_subbands']
        self.center_freq = vars_dict['center_freq']
        self.bandwidth = vars_dict['bandwidth']
        self.subband_size = vars_dict['subband_size']

        self.n_rx = vars_dict['n_rx']
        self.rx_pos = vars_dict['rx_pos']
        self.grid_dims = vars_dict['grid_dims']
        self.grid_idxs_enabled = vars_dict['grid_idxs_enabled']
        self.real_grid_dims = vars_dict['real_grid_dims']
        self.scale = self.real_grid_dims / self.grid_dims
        
        self.scatter = vars_dict['scatter']

        self.tx_pos = vars_dict['tx_pos']
        self.tx_ori = vars_dict['tx_ori']
        
        self.save_folder = vars_dict['save_folder']
        
    def save(self, path=None, override=False):
        save_path = self.get_default_db_path() if path is None else path
        
        if not override and os.path.exists(save_path):
            raise Exception(f'Override=False & Save path already exists ({save_path})')
        
        vars_dict = self.gen_vars_dict()
        
        os.makedirs(self.save_folder, exist_ok=True)
        with open(save_path, 'wb') as fp:
            pickle.dump(vars_dict, fp)
    
    def load(self, path=None):
        load_path = self.get_default_db_path() if path is None else path
        
        with open(load_path, 'rb') as fp:
            vars_dict = pickle.load(fp)
        
        self.load_vars_dict(vars_dict)
    

MTYPES = ['beam-rssi-subband']

class Measurement():
    """ Measurement Class"""
    def __init__(self, mtype, db_indexing, value, real_pos, time, weight=1):
        
        if mtype in MTYPES:
            self.mtype = mtype
        else:
            raise Exception(f'mtype {mtype} not in available MTYPES {MTYPES}')
        
        # Database indexing
        self.db_indexing = db_indexing
        if mtype == 'beam-rssi-subband':
            self.k = db_indexing[0]
            self.b = db_indexing[1]
            self.db_indexing_str = '(beam, subband)'
        
        # Measurement value
        if mtype == 'beam-rssi-subband':
            self.rssi = value
        
        self.t = time
        self.w = weight
        
        self.real_pos = None
        
    def __repr__(self):
        # return (f'TYPE = {self.mtype} | {self.db_indexing_str} = {self.db_indexing} | ' 
                # f'TIME = {self.t} | weight = {self.w}')
        return (f'(k,b) = {self.db_indexing} | t = {self.t} | rssi = {self.rssi:.1f} dBm')
        

class RSSI_Measurement_Generator():
    """
    Returns a list of Measurements
    """
    def __init__(self, db, std_def):
        # DB: a) where the measurement is going to be derived from (in part)
        #     b) where the measurement is going to be compared with to give a loc accuracy
        self.db = db
        
        self.n_beams = db.n_beams
        self.n_subbands = db.n_subbands
        
        # Standard deviation of the real RSSI measurements
        self.std_default = std_def
        
        self.type = 'beam-rssi-subband'
        
        # Information about the last generated measurements
        self.beams_rep = set()    # beams reported
        self.subbands_rep = set() # subbands reported
        self.times_rep = set()    # times reported
        self.real_pos = None  
        self.meas_list = None
        self.meas_dict = {}
        self.prob_grids = None
        self.merged_grid = None
        self.pos_est = None
        self.pos_error = None
    
    def add_m(self, k, b, t, m, m_list, m_dict):
        m_list.append(m)
        
        idxs = [(k,b,-1), (-1,b,t)] # (k,-1,-1), (-1,b,-1), (-1,-1,t), (k,-1,t)
        for idx in idxs:
            if idx in m_dict:
                m_dict[idx] += [m]
            else:
                m_dict[idx] = [m]
        
        m_dict[(k,b,t)] = m
    
    def gen_measurement(self, beams, subbands, times, pos=None, pos_idx=None, 
                        avg_delta=0, std_delta=0, verbose=False, weighted_mean_pos=False):
        """
        Samples the database for a measurement in certain beams, subbands, and
        a certain number of times. 
        
        The DB gives the power obtained via ray tracing. 
        But the simulated power is supposed to be different from reality. 
        To acocunt for this, we sample the real powers from a normal distribution with 
        mean = RSSI from DB + avg_delta, and standard deviation = default_DB_std + std_delta. 
        
        Note that the DB HAS knowledg of its mean and default std (which is assumed). 
        If we decide to change these for a given measurement using the available deltas,
        the localization performance is expected to decrease.
        
        Regarding the position, we currently need an index to sample from the DB,
        so we either compute the closest index or we use the one in the arguments.
        
        weighted_mean_pos will compute a mean for the real RSS distribution
        that is a weighted mean from the N closest cells to that position. N=4 currently.
        """
        if pos is not None and pos_idx is not None:
            raise Exception('Not sure which one to take as the input position')
        
        if pos_idx is None and not weighted_mean_pos:
            pos_idx = self.db.get_closest_pos_idx(pos)
        
        if pos is None:
            pos = self.db.rx_pos[pos_idx]
        
        self.real_pos = pos
        
        # TODO: this is properly implemented, I think, but the results get much worse. 
        # TODO: Maybe move this logic to the DB as in self.db.sample_pos(pos)
        # NOTEEEE: maybe it's better to record the shift vector and apply the same shift at the end
        # Compute the weights for the N closest neighbor cells 
        if weighted_mean_pos:
            N_CELLS = 4 # cells to consider for averaging their RSS values
            pos_idxs = self.db.get_n_closest_pos_idx(self.real_pos, n=N_CELLS)
            min_dist = np.linalg.norm(self.db.rx_pos[pos_idxs[0]] - self.real_pos)
            if min_dist == 0:
                normed_weights = np.array([1] + [0 for i in range(N_CELLS-1)])
            else:
                dists = np.linalg.norm(self.db.rx_pos[pos_idxs] - self.real_pos,axis=1)
                weights = 1 / (dists-1)**2
                normed_weights = weights / np.sum(weights)
            print(normed_weights)
                
        self.beams_rep = set(beams)
        self.subbands_rep = set(subbands)
        self.times_rep = set(times)
        
        meas_list = []
        meas_dict = {}
        for k in beams:
            for b in subbands:
                mean = self.db.dbm[k,b, pos_idx]
                # mean = np.sum(self.db.dbm[k,b, pos_idxs] * normed_weights)
                vals = np.random.normal(loc=mean + avg_delta,
                                        scale=self.std_default + std_delta,
                                        size=len(times))
                
                for t_idx, t in enumerate(times):
                    m = Measurement(mtype=self.type,
                                    db_indexing=(k,b),
                                    value=vals[t_idx],
                                    real_pos=self.real_pos,
                                    time=t,
                                    weight=1)
                
                    self.add_m(k, b, t, m, meas_list, meas_dict)
                
                if verbose:
                    print(m)
        
        self.meas_list = meas_list
        self.meas_dict = meas_dict 
        self.real_meas = False
        return meas_list
    
    
    def gen_real_beam_meas(self, NK, subbands, times, pos=None, pos_idx=None, verbose=True):
        """
        Generates measurements closer to reality in the sense that the K beams
        with the most power are the ones reported.
        """
        
        # Sample ALL beams
        self.gen_measurement(beams=np.arange(self.n_beams), 
                             subbands=subbands, times=times,
                             pos=pos, pos_idx=pos_idx, verbose=verbose)
        
        self.beams_rep = set()
        self.subbands_rep = set(subbands)
        self.times_rep = set(times)
        
        # For each BAND, and for each TIME, select the top K best beams
        topk_meas_list = []
        new_dict = {}
        for b in subbands:
            for t in times:
                sorted_idxs = np.flip(np.argsort([m.rssi for m in self.meas_dict[(-1,b,t)]]))
                
                top_k_beams = sorted_idxs[:NK]
                
                for k in top_k_beams:
                    self.add_m(k, b, t, self.meas_dict[(k,b,t)], 
                               topk_meas_list, new_dict)
                    self.beams_rep.add(k)
        
        self.meas_list = topk_meas_list
        self.meas_dict = new_dict
        self.real_meas = True
        return topk_meas_list
    
    def average_beams_on_time(self, verbose=True):
        avg_meas_list = []
        avg_meas_dict = {}
        
        for k in self.beams_rep:
            for b in self.subbands_rep:
                if not (k,b,-1) in self.meas_dict:
                    continue
        
                ms = self.meas_dict[(k,b,-1)]
                # get times of all measurements
                t_avg = np.mean([m.t for m in ms])
                
                # get values of all measurements
                val = np.mean([m.rssi for m in ms])
                
                m = Measurement(mtype=self.type,
                                db_indexing=(k,b),
                                value=val,
                                real_pos=self.real_pos,
                                time=t_avg,
                                weight=1) # TODO: gain discounting (more directive 
                                          # beams should give better acc)
                
                self.add_m(k,b,t_avg, m, avg_meas_list, avg_meas_dict)
                
                if verbose:
                    print('---------------------------------------------------------')
                    print(f'Compacting {len(ms)} measurements in (beam,subband) = ({k},{b})')
                    print(f'>>>Input: \n{ms} \n>>>Result: \n{m}\n')
        
        self.meas_list = avg_meas_list
        self.meas_dict = avg_meas_dict
    
    def comp_loc_prob(self, plot=False, verbose=True):
        """ 
        Location probability from the last measurement 
        """
        if self.meas_list is None:
            raise Exception('No last measurement found. '
                            'Call gen_measurements() or gen_real_measurements() first.')
        
        self.prob_grid_shape = self.db.grid_dims.tolist()
        prob_grids = []
        
        if self.real_meas:
            self.average_beams_on_time(verbose=verbose)
            
        merged_grid = np.zeros(self.db.n_rx)
            
        for m in self.meas_list:
            prob_grid = self.db.compute_prob_grid(m.rssi,
                                                  self.std_default,
                                                  beam_idx=m.k, 
                                                  subband_idx=m.b)
            prob_grids.append(prob_grid)
            if plot:
                title = (f'(k,b,t) = ({m.k},{m.b},{m.t}) | GT pos = {self.real_pos} | '
                         f'Value = {m.rssi:.1f} dBm')
                
                self.db.plot_coverage_map(matrix=prob_grid*100,
                                          legend=True, cm_label='Probability [0..100]',
                                          title=title, lims=False)
        
            # TODO: try discout merging of the grids!! 
            # (the changes for this might be in another place)
            merged_grid += prob_grid * m.w # weight
        
        # Normalize
        merged_grid /= np.nansum(merged_grid)
        if plot:
            title = f'Merged Probability Grid for GT Loc = {self.real_pos} \n'
            self.db.plot_coverage_map(matrix=merged_grid*100,
                                      legend=True, cm_label='Probability [0..100]',
                                      title=title, lims=False)
        
        self.prob_grids = prob_grids
        self.merged_grid = merged_grid
        
        return merged_grid
    
    def get_n_highest_prob_cells_2d(self, n):
        merged_grid_2d = self.db.convert_to_2D(self.merged_grid)
        n_nans = np.sum(np.isnan(merged_grid_2d))
        flat_sorted_idxs = np.flip(np.argsort(merged_grid_2d.ravel()))
        
        arg_sort_2d = np.dstack(np.unravel_index(flat_sorted_idxs, self.db.grid_dims))[0]
            
        top_cells = arg_sort_2d[n_nans:n_nans+n]
        
        return top_cells
      
    def get_n_highest_prob_cells(self, n, verbose=False):
        n_nans = np.sum(np.isnan(self.merged_grid)) 
        
        return np.flip(np.argsort(self.merged_grid))[n_nans:n_nans+n] 
        
    def dist_coord_clustering(self, top_cells, cluster_max_dist=6):
        """
        Minimum distance to Mean clustering: adds the point to the cluster with 
        the average between its elements that is closest to the point.
        """
        # TODO: optimize clustering distance
        
        # Definition of a cluster: group of positions with distances between elements < max dist
        clusters_cells = []
        clusters_coords = []
        cluster_probs = []
        for idx, cell in enumerate(top_cells):
            cell_coords = self.db.rx_pos[cell,:2]
            cell_prob = self.merged_grid[cell]
            belongs_to_cluster = False
            min_cluster_dist = 1e9
            for cluster_idx, cluster_coords in enumerate(clusters_coords):
                cluster_mean = (cluster_coords[0] if len(cluster_coords) == 1 
                                else np.mean(np.stack(cluster_coords),axis=0))
                dist_to_cluster = np.linalg.norm(cluster_mean - cell_coords)
                if dist_to_cluster < cluster_max_dist:
                    belongs_to_cluster = True
                    if dist_to_cluster < min_cluster_dist:
                        min_cluster_dist = dist_to_cluster
                        closest_cluster_idx = cluster_idx
            
            if belongs_to_cluster:
                clusters_cells[closest_cluster_idx].append(cell)
                clusters_coords[closest_cluster_idx].append(cell_coords)
                cluster_probs[closest_cluster_idx] += cell_prob
            else:
                # create new cluster if it doesn't belong to any of the existing
                clusters_cells.append([cell])
                clusters_coords.append([cell_coords])
                cluster_probs.append(cell_prob)
                
        return clusters_cells, clusters_coords, cluster_probs
    
    def comp_cluster_probabilities(self, clusters):
        probs = []
        cluster_probs = []
        for cluster_idx, cluster in clusters:
            # Find the idxs of the coordinates in each cluster
            idxs_of_cluster_coords = 0
            # Get the probability of each coordinate and sum them for each cluster
            cluster_probs[cluster_idx] = np.sum(probs[i] for i in idxs_of_cluster_coords)
        
        return cluster_probs
    
    def estimate_loc(self, topx=3, plot=False):
        if self.merged_grid is None:
            raise Exception('No last merged probability grid. Call merge_prob_grids() first.')
        
        # Objective: get top-x higher probabilities
        # TODO: define X number of cells to consider based on the accumulated probability
        # TODO: check why topx != 3 creates problems...
        
        # Choose top NK beams
        top_cells = self.get_n_highest_prob_cells(topx)
        
        if plot:
            top_cells_2d = self.get_n_highest_prob_cells_2d(topx)
            print(f'2D cells found: {top_cells_2d}')
        
        # if the probabilities belong to different clusters (clusters can be defined by distances)
        # then simply use the location with max probability
        
        # Compute distances between coordinates (3 coords have 3! distance)
        clusters_cells, clusters_coords, cluster_probs = self.dist_coord_clustering(top_cells)
        
        # Average the positiosn in the cluster with highest probability
        best_cluster_idx = np.argmax(cluster_probs)
        positions_in_cluster = np.stack(clusters_coords[best_cluster_idx])
        probs_in_best_cluster = self.merged_grid[clusters_cells[best_cluster_idx]]
        pos_estimate = np.average(positions_in_cluster, axis=0, weights=probs_in_best_cluster)
        
        if plot:
            top_coords = np.stack([self.db.rx_pos[cell,:2] for cell in top_cells])
                 
            self.db.plot_coverage_map(title='GT locations vs 3 most likely cells',
                                      beam_idx=12, subband_idx=0, rx_pos=top_coords,
                                      tx_pos=pos_estimate, tx_ori=None, legend=True)
        
        self.pos_est = pos_estimate
        
        return pos_estimate
        
    def comp_pos_error(self, verbose=True):
        if self.pos_est is None:
            raise Exception('No last pos estimate. Call estimate_loc() first.')
        
        if verbose:
            print(f'Pos: {self.real_pos[:2]} | Pos Estimate: {self.pos_est}')
        self.pos_error = np.linalg.norm(self.real_pos[:2] - self.pos_est)
        if verbose:
            print(f'pos error = {self.pos_error:.2f}')
        
        # NOTE: if error < cell size, then we got the right cell!
        self.pos_error_in_cells = self.pos_error / self.db.scale[0] # assumes square cells
        if verbose:
            print(f'error in cells = {self.pos_error_in_cells:.2f}')
        
        return self.pos_error 
        
    def plot_final_result(self, tx_pos=None, tx_ori=None):
        
        title =  f'B,K,T = {self.beams_rep},{self.subbands_rep},{self.times_rep} | '
        title += f'GT Loc = {self.real_pos[:2]} | \n'
        title += f'Estimate Loc = [{self.pos_est[0]:.2f}, {self.pos_est[1]:.2f}] | '
        title += f'Pos error = {self.pos_error:.2f} m'
        
        # RX 0 will be the TRUE position, RX1 will be the esimated
        self.db.plot_coverage_map(matrix=self.merged_grid*100, 
                                  title=title, lims=False,
                                  rx_pos=[self.real_pos, self.pos_est],
                                  legend=True, cm_label='Probability [0..100]')
        