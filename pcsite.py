#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:56:06 2019
Module for the Site class.
@author: parrenif
"""

import os
import sys
import warnings
import math as m
import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import cholesky
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
import pickle
import yaml
from pcmath import interp_lin_aver, interp_stair_aver, grid, truncation, stretch
import pccfg

##dummy use of the interp1d function
Fooooooo = interp1d

class Site(object):
    """This is the class for a site."""

    def __init__(self, dlab):
        self.label = dlab

        #Default parameters
        self.archive = 'icecore'
        self.accu_prior_rep = 'staircase'
        self.age_top = None
        self.depth = np.empty(0)
        self.corr_a_age = None
        self.calc_a = False
        self.calc_a_method = None
        self.gamma_source = None
        self.beta_source = None
        self.calc_tau = False
        self.thickness = None
        self.calc_lid = False
        self.lid_value = None
        self.start = 'prior'
        self.corr_lid_age = None
        self.corr_tau_depth = None
        self.accu0 = None
        self.beta = None
        self.pprime = None
        self.muprime = None
        self.sliding = None
        self.dens_firn = None

##Setting of the parameters from the parameter files
        
        yamls = ''
        filename = pccfg.datadir+'/parameters_all_sites.yml'
        if os.path.isfile(filename):
            yamls = open(filename).read()
        filename = pccfg.datadir+self.label+'/parameters.yml'
        if os.path.isfile(filename):
            yamls += '\n'+open(filename).read()
        if yamls != '':
            para = yaml.load(yamls, Loader=yaml.Loader)
            self.__dict__.update(para)
            self.depth = grid(self.depth_grid)
            self.corr_deporate_age = grid(self.corr_deporate_grid)
            try:
                inf = self.age_truncation['inf']
                sup = self.age_truncation['sup']
                self.corr_deporate_age = truncation(self.corr_deporate_age, inf, sup)
            except AttributeError:
                pass
            if self.archive == 'icecore':
                self.corr_lid_age =grid(self.corr_lid_grid)
                try:
                    inf = self.age_truncation['inf']
                    sup = self.age_truncation['sup']
                    self.corr_lid_age = truncation(self.corr_lid_age, inf, sup)
                except AttributeError:
                    pass
                self.corr_thinning_depth = grid(self.corr_thinning_grid)
                self.corr_thinning_depth = stretch(self.corr_thinning_depth, self.depth[0], 
                                                   self.depth[-1])
        else:
            filename = pccfg.datadir+'/parameters_all_sites.py'
            if os.path.isfile(filename):
                exec(open(filename).read())
                print('Python format for parameters_all_sites file is deprecated.'
                      'Use YAML format instead.')
            else:
                filename = pccfg.datadir+'/parameters-AllDrillings.py'
                if os.path.isfile(filename):
                    exec(open(filename).read())
                    print('parameters-AllDrillings.py file is deprecated.'
                          'Use YAML parameters_all_sites.yml file instead.')

            print('WARNING: python parameters file for sites are deprecated.'
                  'Use YAML format instead.')
            exec(open(pccfg.datadir+self.label+'/parameters.py').read())

##Translation of deprecated names

        try:
            self.calc_lid = self.calc_LID
            print('WARNING: calc_LID is deprecated. Use calc_lid instead.')
        except AttributeError:
            pass
        try:
            self.corr_lid_age = self.corr_LID_age
            print('WARNING: corr_LID_age is deprecated. Use corr_lid_age instead.')
        except AttributeError:
            pass
        try:
            self.dens_firn = self.Dfirn
            print('WARNING: Dfirn is deprecated. Use dens_firn instead.')
        except AttributeError:
            pass
        try:
            self.sliding = self.s
            print('WARNING: s is deprecated. Use sliding instead.')
        except AttributeError:
            pass
        try:
            self.accu0 = self.A0
            print('WARNING: A0 is deprecated. Use deporate0 instead.')
        except AttributeError:
            pass
        try:
            self.accu0 = self.deporate0
        except AttributeError:
            pass
        try:
            self.calc_a = self.calc_deporate
        except AttributeError:
            print('WARNING: calc_a is deprecated. Use calc_deporate instead.')
        try:
            self.corr_a_age = self.corr_deporate_age
        except AttributeError:
            print('WARNING: corr_a_age is deprecated. Use corr_deporate_age instead.')
        try:
            self.lambda_lid = self.lambda_LID
            print('WARNING: lambda_LID is deprecated. Use lambda_lid instead.')
        except AttributeError:
            pass
        try:
            self.lambda_a = self.lambda_deporate
        except AttributeError:
            print('WARNING: lambda_a is deprecated. Use lambda_deporate instead.')
        try:
            self.lambda_tau = self.lambda_thinning
        except AttributeError:
            print('WARNING: lambda_tau is deprecated. Use lambda_thinning instead.')
        try:
            self.calc_tau = self.calc_thinning
        except AttributeError:
            print('WARNING: calc_tau is deprecated. Use calc_thinning instead.')
        if self.archive == 'icecore':
            try:
                self.corr_tau_depth = self.corr_thinning_depth
            except AttributeError:
                print('WARNING: corr_tau_depth is deprecated. Use corr_thinning_depth instead.')
        try:
            self.accu_prior_rep = self.deporate_prior_rep
        except AttributeError:
            print('WARNING: accu_prior_rep is deprecated. Use deporate_prior_rep instead.')
        try:
            self.sigmap_corr_a
            print('WARNING: sigmap_corr_a is deprecated. Use deporate_prior_sigma instead.')
        except AttributeError:
            try:
                self.sigmap_corr_a = self.deporate_prior_sigma
            except AttributeError:
                pass
        try:
            self.sigmap_corr_tau
            print('WARNING: sigmap_corr_tau is deprecated. Use thinning_prior_sigma instead.')
        except AttributeError:
            try:
                self.sigmap_corr_tau = self.thinning_prior_sigma
            except AttributeError:
                pass
        try:
            self.sigmap_corr_lid
            print('WARNING: sigmap_corr_lid is deprecated. Use lid_prior_sigma instead.')
        except AttributeError:
            try:
                self.sigmap_corr_lid = self.lid_prior_sigma
            except AttributeError:
                pass
        try:
            self.age_top_prior
        except AttributeError:
            self.age_top_prior = self.age_top
            print('WARNING: Now, age_top is a variable to be optimized.'
                  'Therefore you need to define age_top_prior.'
                  'Using age_top for now.')
        try:
            self.age_top_sigma
        except AttributeError:
            self.age_top_sigma = 10.
            print('WARNING: Now, age_top is a variable to be optimized.'
                  'Therefore you need to define age_top_sigma.'
                  'Setting age_top_sigma to', self.age_top_sigma, 'for now.')
        
        ##Initialisation of variables
        self.depth_mid = (self.depth[1:]+self.depth[:-1])/2
        self.depth_inter = (self.depth[1:]-self.depth[:-1])
        self.lid = np.empty_like(self.depth)
        self.sigma_delta_depth = np.empty_like(self.depth)
        self.sigma_airlayerthick = np.empty_like(self.depth_mid)
        self.airlayerthick_init = np.empty_like(self.depth_mid)
        self.age_init = np.empty_like(self.depth)
        self.sigma_accu = np.empty_like(self.depth_mid)
        self.sigma_accu_model = np.empty_like(self.depth_mid)
        self.tau_init = np.empty_like(self.depth_mid)
        self.a_init = np.empty_like(self.depth_mid)
        self.airage_init = np.empty_like(self.depth_mid)
        self.sigma_icelayerthick = np.empty_like(self.depth_mid)
        self.airlayerthick = np.empty_like(self.depth_mid)
        self.ice_equiv_depth = np.empty_like(self.depth)
        self.sigma_tau = np.empty_like(self.depth_mid)
        self.icelayerthick = np.empty_like(self.depth_mid)
        self.icelayerthick_init = np.empty_like(self.depth_mid)
        self.sigma_tau_model = np.empty_like(self.depth_mid)
        self.delta_depth_init = np.empty_like(self.depth)
        self.sigma_lid_model = np.empty_like(self.depth)
        self.lid_init = np.empty_like(self.depth)
        self.sigma_age = np.empty_like(self.depth)
        self.sigma_airage = np.empty_like(self.depth)
        self.sigma_lid = np.empty_like(self.depth)
        self.ulidie = np.empty_like(self.depth)
        self.cov = np.array([])

## We set up the raw model
        if self.calc_a:
            if self.archive == 'icecore':
                readarray = np.loadtxt(pccfg.datadir+self.label+'/isotopes.txt')
                if np.size(readarray) == np.shape(readarray)[0]:
                    readarray.resize(1, np.size(readarray))
                self.iso_depth = readarray[:, 0]
                if self.calc_a_method == 'fullcorr':
                    self.iso_d18o_ice = readarray[:, 1]
                    self.d18o_ice = interp_stair_aver(self.depth, self.iso_depth, self.iso_d18o_ice)
                    self.iso_deutice = readarray[:, 2]
                    self.deutice = interp_stair_aver(self.depth, self.iso_depth, self.iso_deutice)
                    self.iso_d18o_sw = readarray[:, 3]
                    self.d18o_sw = interp_stair_aver(self.depth, self.iso_depth, self.iso_d18o_sw)
                    self.excess = self.deutice-8*self.d18o_ice   # dans Uemura : d=excess
                    self.accu = np.empty_like(self.deutice)
                    self.d18o_ice_corr = self.d18o_ice-self.d18o_sw*(1+self.d18o_ice/1000)/\
                        (1+self.d18o_sw/1000)	#Uemura (1)
                    self.deutice_corr = self.deutice-8*self.d18o_sw*(1+self.deutice/1000)/\
                        (1+8*self.d18o_sw/1000) #Uemura et al. (CP, 2012) (2)
                    self.excess_corr = self.deutice_corr-8*self.d18o_ice_corr
                    self.deutice_fullcorr = self.deutice_corr+self.gamma_source/self.beta_source*\
                        self.excess_corr
                elif self.calc_a_method == 'deut':
                    self.iso_deutice = readarray[:, 1]
                    self.deutice_fullcorr = interp_stair_aver(self.depth, self.iso_depth,
                                                              self.iso_deutice)
                elif self.calc_a_method == 'd18O':
                    self.d18o_ice = readarray[:, 1]
                    self.deutice_fullcorr = 8*interp_stair_aver(self.depth, self.iso_depth,
                                                                self.iso_d18o_ice)
                else:
                    print('Accumulation method not recognized')
                    sys.exit()
        else:
            if os.path.isfile(pccfg.datadir+self.label+'/deposition.txt'):
                readarray = np.loadtxt(pccfg.datadir+self.label+'/deposition.txt')
            else:
                readarray = np.loadtxt(pccfg.datadir+self.label+'/accu-prior.txt')
            if np.size(readarray) == np.shape(readarray)[0]:
                readarray.resize(1, np.size(readarray))
            self.a_depth = readarray[:, 0]
            self.a_a = readarray[:, 1]
            if readarray.shape[1] >= 3:
                self.a_sigma = readarray[:, 2]
            if self.accu_prior_rep == 'staircase':
                self.a_model = interp_stair_aver(self.depth, self.a_depth, self.a_a)
            elif self.accu_prior_rep == 'linear':
                self.a_model = interp_lin_aver(self.depth, self.a_depth, self.a_a)
            else:
                print('Representation of prior accu scenario not recognized')
            self.accu = self.a_model

        self.age = np.empty_like(self.depth)
        self.airage = np.empty_like(self.depth)

        if self.archive == 'icecore':

            if os.path.isfile(pccfg.datadir+self.label+'/density.txt'):
                readarray = np.loadtxt(pccfg.datadir+self.label+'/density.txt')
            else:
                readarray = np.loadtxt(pccfg.datadir+self.label+'/density-prior.txt')
            #        self.density_depth=readarray[:,0]
            if np.size(readarray) == np.shape(readarray)[0]:
                readarray.resize(1, np.size(readarray))
            self.dens_depth = readarray[:, 0]
            self.dens_dens = readarray[:, 1]
            self.dens = np.interp(self.depth_mid, self.dens_depth, self.dens_dens)
            self.iedepth = np.cumsum(np.concatenate((np.array([0]), self.dens*self.depth_inter)))
            self.iedepth_mid = (self.iedepth[1:]+self.iedepth[:-1])/2

            if self.calc_tau:
                self.thickness_ie = self.thickness-self.depth[-1]+self.iedepth[-1]

            if self.calc_lid:
                if self.depth[0] < self.lid_value:
                    self.lid_depth = np.array([self.depth[0], self.lid_value, self.depth[-1]])
                    self.lid_lid = np.array([self.depth[0], self.lid_value, self.lid_value])
                else:
                    self.lid_depth = np.array([self.depth[0], self.depth[-1]])
                    self.lid_lid = np.array([self.lid_value, self.lid_value])
            else:
    #            self.lid_model=np.loadtxt(pccfg.datadir+self.label+'/LID-prior.txt')
                if os.path.isfile(pccfg.datadir+self.label+'/lock_in_depth.txt'):
                    readarray = np.loadtxt(pccfg.datadir+self.label+'/lock_in_depth.txt')
                else:
                    readarray = np.loadtxt(pccfg.datadir+self.label+'/LID-prior.txt')
                if np.size(readarray) == np.shape(readarray)[0]:
                    readarray.resize(1, np.size(readarray))
                self.lid_depth = readarray[:, 0]
                self.lid_lid = readarray[:, 1]
                if readarray.shape[1] >= 3:
                    self.lid_sigma = readarray[:, 2]
            self.lid_model = np.interp(self.depth, self.lid_depth, self.lid_lid)

            self.delta_depth = np.empty_like(self.depth)
            self.udepth = np.empty_like(self.depth)

#        print 'depth_mid ', np.size(self.depth_mid)
#        print 'zeta ', np.size(self.zeta)
            if self.calc_tau:
                self.thicknessie = self.thickness-self.depth[-1]+self.iedepth[-1]
                #FIXME: maybe we should use iedepth and thickness_ie here?
                self.zeta = (self.thicknessie-self.iedepth_mid)/self.thicknessie
                self.tau = np.empty_like(self.depth_mid)
            else:
                if os.path.isfile(pccfg.datadir+self.label+'/thinning.txt'):
                    readarray = np.loadtxt(pccfg.datadir+self.label+'/thinning.txt')
                else:
                    readarray = np.loadtxt(pccfg.datadir+self.label+'/thinning-prior.txt')
                if np.size(readarray) == np.shape(readarray)[0]:
                    readarray.resize(1, np.size(readarray))
                self.tau_depth = readarray[:, 0]
                self.tau_tau = readarray[:, 1]
                if readarray.shape[1] >= 3:
                    self.tau_sigma = readarray[:, 2]
                self.tau_model = np.interp(self.depth_mid, self.tau_depth, self.tau_tau)
                self.tau = self.tau_model

        self.raw_model()


## Now we set up the correction functions

        if self.start == 'restart':
            with open(pccfg.datadir+self.label+'/restart.bin', 'rb') as f:
                if self.archive == 'icecore':
                    resi_age_top, corr_a_age, corr_a, corr_lid_age, corr_lid, corr_tau_depth,\
                        corr_tau = pickle.load(f)
                    self.corr_lid = np.interp(self.corr_lid_age, corr_lid_age, corr_lid)
                    self.corr_tau = np.interp(self.corr_tau_depth, corr_tau_depth, corr_tau)
                    
                else:
                    resi_age_top, corr_a_age, corr_a = pickle.load(f)
            self.resi_age_top = resi_age_top
            self.corr_a = np.interp(self.corr_a_age, corr_a_age, corr_a)

        elif self.start == 'default' or self.start == 'prior':
            self.resi_age_top = np.array([0.])
            self.corr_a = np.zeros(np.size(self.corr_a_age))
            if self.archive == 'icecore':
                self.corr_lid = np.zeros(np.size(self.corr_lid_age))
                self.corr_tau = np.zeros(np.size(self.corr_tau_depth))
        elif self.start == 'random':
            self.resi_age_top = np.random.normal(loc=0., scale=1., size=1.)
            self.corr_a = np.random.normal(loc=0., scale=1., size=np.size(self.corr_a_age))
            if self.archive == 'icecore':
                self.corr_lid = np.random.normal(loc=0., scale=1., size=np.size(self.corr_lid_age))
                self.corr_tau = np.random.normal(loc=0., scale=1.,
                                                 size=np.size(self.corr_tau_depth))
        else:
            print('Start option not recognized.')



## Definition of the covariance matrix of the background

        try:
          x_out = (self.corr_a_age[:-1]+self.corr_a_age[1:])/2
          x_out = np.concatenate((np.array([self.corr_a_age[0]]), x_out,
                                 np.array([self.corr_a_age[-1]])))
          self.sigmap_corr_a = interp_stair_aver(x_out, self.fct_age_model(self.a_depth),
                                                        self.a_sigma)
        except AttributeError:
            print('Sigma on prior accu scenario not defined in the accu-prior.txt file')
            self.sigmap_corr_a=self.sigmap_corr_a*np.ones(np.size(self.corr_a_age))

        if self.archive == 'icecore':
            try:
                x_out = (self.corr_lid_age[:-1]+self.corr_lid_age[1:])/2
                x_out = np.concatenate((np.array([self.corr_lid_age[0]]), x_out,
                                        np.array([self.corr_lid_age[-1]])))

                lid_age = self.fct_airage_model(self.lid_depth)
                self.sigmap_corr_lid = interp_lin_aver(x_out,
                                                 lid_age[~np.isnan(lid_age)],
                                                 self.lid_sigma[~np.isnan(lid_age)])
            except AttributeError:
                print('Sigma on prior LID scenario not defined in the LID-prior.txt file')
                self.sigmap_corr_lid=self.sigmap_corr_lid*np.ones(np.size(self.corr_lid_age))

            try:
                x_out = (self.corr_tau_depth[:-1]+self.corr_tau_depth[1:])/2
                x_out = np.concatenate((np.array([self.corr_tau_depth[0]]), x_out,
                                 np.array([self.corr_tau_depth[-1]])))
                self.sigmap_corr_tau = interp_lin_aver(x_out, self.tau_depth,
                                                 self.tau_sigma)
            except AttributeError:
                print('Sigma on prior thinning scenario not defined in the thinning-prior.txt file')
                self.sigmap_corr_tau=self.k/self.thickness_ie*np.interp(self.corr_tau_depth,
                                                                        self.depth, self.iedepth)

        #Accu correlation matrix
        self.correlation_corr_a = np.interp(np.abs(np.ones((np.size(self.corr_a_age),\
            np.size(self.corr_a_age)))*self.corr_a_age-\
            np.transpose(np.ones((np.size(self.corr_a_age),\
            np.size(self.corr_a_age)))*self.corr_a_age)),\
            np.array([0,self.lambda_a]),np.array([1, 0]))
        
        
        if self.archive == 'icecore':
            #LID correlation matrix
            self.correlation_corr_lid = np.interp(np.abs(np.ones((np.size(self.corr_lid_age),\
                np.size(self.corr_lid_age)))*self.corr_lid_age-np.transpose(np.ones((np.size(\
                self.corr_lid_age),np.size(self.corr_lid_age)))*self.corr_lid_age)),\
                np.array([0,self.lambda_lid]),np.array([1, 0]))
            
            #Thinning correlation matrix
            self.correlation_corr_tau = np.interp(np.abs(np.ones((np.size(self.corr_tau_depth),\
                np.size(self.corr_tau_depth)))*self.corr_tau_depth-np.transpose(np.ones((np.size(\
                self.corr_tau_depth),np.size(self.corr_tau_depth)))*self.corr_tau_depth)),\
                np.array([0,self.lambda_tau]),np.array([1, 0]) )
    

        self.chol_a = cholesky(self.correlation_corr_a)
        if self.archive == 'icecore':
            self.chol_lid = cholesky(self.correlation_corr_lid)
            self.chol_tau = cholesky(self.correlation_corr_tau)

#Definition of the variable vector
            
        if self.archive == 'icecore':
            self.variables = np.concatenate((self.resi_age_top,
                                             self.corr_a, self.corr_tau, self.corr_lid))
        else:
            self.variables = np.concatenate((self.resi_age_top, self.corr_a))

#Reading of observations

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/ice_age_horizons.txt'
            if not os.path.isfile(filename):
                filename = pccfg.datadir+self.label+'/ice_age.txt'
        else:
            filename = pccfg.datadir+self.label+'/age_horizons.txt'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if os.path.isfile(filename) and open(filename).read() and\
                np.size(np.loadtxt(filename)) > 0:
                readarray = np.loadtxt(filename)
                if np.size(readarray) == np.shape(readarray)[0]:
                    readarray.resize(1, np.size(readarray))
                self.icehorizons_depth = readarray[:, 0]
                self.icehorizons_age = readarray[:, 1]
                self.icehorizons_sigma = readarray[:, 2]
            else:
                self.icehorizons_depth = np.array([])
                self.icehorizons_age = np.array([])
                self.icehorizons_sigma = np.array([])

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/ice_age_intervals.txt'
        else:
            filename = pccfg.datadir+self.label+'/age_intervals.txt'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if os.path.isfile(filename) and open(filename).read() and\
                np.size(np.loadtxt(filename)) > 0:
                readarray = np.loadtxt(filename)
                if np.size(readarray) == np.shape(readarray)[0]:
                    readarray.resize(1, np.size(readarray))
                self.iceintervals_depthtop = readarray[:, 0]
                self.iceintervals_depthbot = readarray[:, 1]
                self.iceintervals_duration = readarray[:, 2]
                self.iceintervals_sigma = readarray[:, 3]
            else:
                self.iceintervals_depthtop = np.array([])
                self.iceintervals_depthbot = np.array([])
                self.iceintervals_duration = np.array([])
                self.iceintervals_sigma = np.array([])

        if self.archive == 'icecore':
            filename = pccfg.datadir+self.label+'/air_age_horizons.txt'
            if not os.path.isfile(filename):
                filename = pccfg.datadir+self.label+'/air_age.txt'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if os.path.isfile(filename) and open(filename).read() and\
                    np.size(np.loadtxt(filename)) > 0:
                    readarray = np.loadtxt(filename)
                    if np.size(readarray) == np.shape(readarray)[0]:
                        readarray.resize(1, np.size(readarray))
                    self.airhorizons_depth = readarray[:, 0]
                    self.airhorizons_age = readarray[:, 1]
                    self.airhorizons_sigma = readarray[:, 2]
                else:
                    self.airhorizons_depth = np.array([])
                    self.airhorizons_age = np.array([])
                    self.airhorizons_sigma = np.array([])

            filename = pccfg.datadir+self.label+'/air_age_intervals.txt'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if os.path.isfile(filename) and open(filename).read() and\
                    np.size(np.loadtxt(filename)) > 0:
                    readarray = np.loadtxt(filename)
                    if np.size(readarray) == np.shape(readarray)[0]:
                        readarray.resize(1, np.size(readarray))
                    self.airintervals_depthtop = readarray[:, 0]
                    self.airintervals_depthbot = readarray[:, 1]
                    self.airintervals_duration = readarray[:, 2]
                    self.airintervals_sigma = readarray[:, 3]
                else:
                    self.airintervals_depthtop = np.array([])
                    self.airintervals_depthbot = np.array([])
                    self.airintervals_duration = np.array([])
                    self.airintervals_sigma = np.array([])

            filename = pccfg.datadir+self.label+'/delta_depths.txt'
            if not os.path.isfile(filename):
                filename = pccfg.datadir+self.label+'/Ddepth.txt'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if os.path.isfile(filename) and open(filename).read() and\
                    np.size(np.loadtxt(filename)) > 0:
                    readarray = np.loadtxt(filename)
                    if np.size(readarray) == np.shape(readarray)[0]:
                        readarray.resize(1, np.size(readarray))
                    self.delta_depth_depth = readarray[:, 0]
                    self.delta_depth_delta_depth = readarray[:, 1]
                    self.delta_depth_sigma = readarray[:, 2]
                else:
                    self.delta_depth_depth = np.array([])
                    self.delta_depth_delta_depth = np.array([])
                    self.delta_depth_sigma = np.array([])


        self.icehorizons_correlation = np.diag(np.ones(np.size(self.icehorizons_depth)))
        self.iceintervals_correlation = np.diag(np.ones(np.size(self.iceintervals_depthtop)))
        if self.archive == 'icecore':
            self.airhorizons_correlation = np.diag(np.ones(np.size(self.airhorizons_depth)))
            self.airintervals_correlation = np.diag(np.ones(np.size(self.airintervals_depthtop)))
            self.delta_depth_correlation = np.diag(np.ones(np.size(self.delta_depth_depth)))
#        print self.icehorizons_correlation

        filename = pccfg.datadir+'/parameters_covariance_observations_all_sites.py'
        filename2 = pccfg.datadir+'/parameters-CovarianceObservations-AllDrillings.py'
        if os.path.isfile(filename):
            exec(open(filename).read())
        elif os.path.isfile(filename2):
            exec(open(filename2).read())

        filename = pccfg.datadir+self.label+'/parameters_covariance_observations.py'
        filename2 = pccfg.datadir+self.label+'/parameters-CovarianceObservations.py'
        if os.path.isfile(filename):
            exec(open(filename).read())
        elif os.path.isfile(filename2):
            exec(open(filename2).read())
        if np.size(self.icehorizons_depth) > 0:
            self.icehorizons_chol = cholesky(self.icehorizons_correlation)
            #FIXME: we LU factor a triangular matrix. This is suboptimal.
            #We should set lu_piv directly instead.
            self.icehorizons_lu_piv = lu_factor(np.transpose(self.icehorizons_chol))
        if np.size(self.iceintervals_depthtop) > 0:
            self.iceintervals_chol = cholesky(self.iceintervals_correlation)
            self.iceintervals_lu_piv = lu_factor(np.transpose(self.iceintervals_chol))
        if self.archive == 'icecore':
            if np.size(self.airhorizons_depth) > 0:
                self.airhorizons_chol = cholesky(self.airhorizons_correlation)
                self.airhorizons_lu_piv = lu_factor(np.transpose(self.airhorizons_chol))
            if np.size(self.airintervals_depthtop) > 0:
                self.airintervals_chol = cholesky(self.airintervals_correlation)
                self.airintervals_lu_piv = lu_factor(np.transpose(self.airintervals_chol))
            if np.size(self.delta_depth_depth) > 0:
                self.delta_depth_chol = cholesky(self.delta_depth_correlation)
                self.delta_depth_lu_piv = lu_factor(np.transpose(self.delta_depth_chol))

    def raw_model(self):
        """Calculate the raw model, that is before applying correction functions."""

        self.age_top = self.age_top_prior
        
        if self.archive == 'icecore':
            #Accumulation
            if self.calc_a:
                self.a_model = self.accu0*np.exp(self.beta*(self.deutice_fullcorr-\
                    self.deutice_fullcorr[0])) #Parrenin et al. (CP, 2007a) 2.3 (6)

            #Thinning
            if self.calc_tau:
                self.p_def = -1+m.exp(self.pprime)
                self.mu_melt = m.exp(self.muprime)
    #            self.sliding=m.tanh(self.sprime)
                #Parrenin et al. (CP, 2007a) 2.2 (3)
                omega_def = 1-(self.p_def+2)/(self.p_def+1)*(1-self.zeta)+\
                          1/(self.p_def+1)*(1-self.zeta)**(self.p_def+2)
                #Parrenin et al. (CP, 2007a) 2.2 (2)
                omega = self.sliding*self.zeta+(1-self.sliding)*omega_def
                self.tau_model = (1-self.mu_melt)*omega+self.mu_melt

            #udepth
            self.udepth_model = self.depth[0]+np.cumsum(np.concatenate((np.array([0]),\
                                self.dens/self.tau_model*self.depth_inter)))

            self.ulidie_model = self.lid_model*self.dens_firn

        else:
            if self.calc_a:
                self.a_model = self.accu0*np.ones(np.size(self.depth_inter))

        #Ice age
        if self.archive == 'icecore':
            self.icelayerthick_model = self.tau_model*self.a_model/self.dens
        else:
            self.icelayerthick_model = self.a_model
        self.age_model = self.age_top+np.cumsum(np.concatenate((np.array([0]),\
                         self.depth_inter/self.icelayerthick_model)))

        #air age
        if self.archive == 'icecore':
            self.ice_equiv_depth_model = np.interp(self.udepth_model-self.ulidie_model,
                                                   self.udepth_model, self.depth, left=np.nan)
            self.delta_depth_model = self.depth-self.ice_equiv_depth_model
            self.airage_model = np.interp(self.ice_equiv_depth_model, self.depth, self.age_model,
                                          left=np.nan, right=np.nan)
            with np.errstate(divide='ignore'):
                self.airlayerthick_model = 1/np.diff(self.airage_model)

    def corrected_model(self):
        """Calculate the age model, taking into account the correction functions."""

        #Age top
        self.age_top = self.age_top_prior + self.resi_age_top[0] * self.age_top_sigma

        #Accu
        corr = np.dot(self.chol_a, self.corr_a)*self.sigmap_corr_a
        #FIXME: we should use mid-age and not age
        self.accu = self.a_model*np.exp(np.interp(self.age_model[:-1], self.corr_a_age, corr))

        #Thinning and LID
        if self.archive == 'icecore':
            self.tau = self.tau_model*np.exp(np.interp(self.depth_mid, self.corr_tau_depth,\
                       np.dot(self.chol_tau, self.corr_tau)*self.sigmap_corr_tau))
            self.udepth = self.depth[0]+np.cumsum(np.concatenate((np.array([0]),\
                          self.dens/self.tau*self.depth_inter)))
            corr = np.dot(self.chol_lid, self.corr_lid)*self.sigmap_corr_lid
            self.lid = self.lid_model*np.exp(np.interp(self.airage_model, self.corr_lid_age, corr))
            self.ulidie = self.lid*self.dens_firn
   
        #Ice age
        if self.archive == 'icecore':
            self.icelayerthick = self.tau*self.accu/self.dens
        else:
            self.icelayerthick = self.accu
        self.age = self.age_top+np.cumsum(np.concatenate((np.array([0]),\
                   self.depth_inter/self.icelayerthick)))

        #Air age
        if self.archive == 'icecore':
            self.ice_equiv_depth = np.interp(self.udepth-self.ulidie, self.udepth, self.depth,
                                             left=np.nan)
            self.delta_depth = self.depth-self.ice_equiv_depth
            self.airage = np.interp(self.ice_equiv_depth, self.depth, self.age, left=np.nan,
                                    right=np.nan)
            with np.errstate(divide='ignore'):
                self.airlayerthick = 1/np.diff(self.airage)

    def model(self, var):
        """Calculate the model from the vector var containing its variables."""

#        if self.calc_a==True:
#            self.accu0=var[index]
#            self.beta=var[index+1]
#            index=index+2
#        if self.calc_tau==True:
##            self.p_def=-1+m.exp(var[index])
##            self.sliding=var[index+1]
##            self.mu_melt=var[index+2]
##            index=index+3
#            self.pprime=var[index]
#            self.muprime=var[index+1]
#            index=index+2
        index = 0
#        self.resi_age_top = var[0:1]
        self.resi_age_top = var[0:1]
        index = 1
        self.corr_a = var[index:index+np.size(self.corr_a)]
        if self.archive == 'icecore':
            self.corr_tau = var[index+np.size(self.corr_a):\
                              index+np.size(self.corr_a)+np.size(self.corr_tau)]
            self.corr_lid = var[index+np.size(self.corr_tau)+np.size(self.corr_a):\
                        index+np.size(self.corr_tau)+np.size(self.corr_a)+np.size(self.corr_lid)]

        ##Raw model
#        self.raw_model()

        ##Corrected model
        self.corrected_model()

        if self.archive == 'icecore':
            return np.concatenate((self.age, self.accu, self.icelayerthick, self.airage,
                                   self.delta_depth, self.tau, self.lid, self.airlayerthick))
        else:
            return np.concatenate((self.age, self.accu, self.icelayerthick))


    def write_init(self):
        """Write the initial values of the variables in the corresponding *_init variables."""
        self.age_init = self.age
        self.a_init = self.accu
        self.icelayerthick_init = self.icelayerthick
        if self.archive == 'icecore':
            self.lid_init = self.lid
            self.tau_init = self.tau
            self.airlayerthick_init = self.airlayerthick
            self.airage_init = self.airage
            self.delta_depth_init = self.delta_depth

    def fct_age(self, depth):
        """Return the age at given depths."""
        return np.interp(depth, self.depth, self.age)

    def fct_age_init(self, depth):
        """Return the initial age at given depths."""
        return np.interp(depth, self.depth, self.age_init)

    def fct_age_model(self, depth):
        """Return the raw modelled age at given depths."""
        return np.interp(depth, self.depth, self.age_model)

    def fct_airage(self, depth):
        """Return the air age at given depths."""
        return np.interp(depth, self.depth, self.airage)

    def fct_airage_init(self, depth):
        """Return the initial air age at given depth."""
        return np.interp(depth, self.depth, self.airage_init)

    def fct_airage_model(self, depth):
        """Return the raw modelled air age at given depths."""
        return np.interp(depth, self.depth, self.airage_model)

    def fct_delta_depth(self, depth):
        """Return the delta_depth at given detphs."""
        return np.interp(depth, self.depth, self.delta_depth)

    def residuals(self):
        """Calculate the residuals from the vector of the variables"""
        resi_age = (self.fct_age(self.icehorizons_depth)-self.icehorizons_age)\
                   /self.icehorizons_sigma
        if np.size(self.icehorizons_depth) > 0:
            resi_age = lu_solve(self.icehorizons_lu_piv, resi_age)
        resi_iceint = (self.fct_age(self.iceintervals_depthbot)-\
                      self.fct_age(self.iceintervals_depthtop)-\
                      self.iceintervals_duration)/self.iceintervals_sigma
        if np.size(self.iceintervals_depthtop) > 0:
            resi_iceint = lu_solve(self.iceintervals_lu_piv, resi_iceint)

        if self.archive == 'icecore':
            resi_airage = (self.fct_airage(self.airhorizons_depth)-self.airhorizons_age)/\
                          self.airhorizons_sigma
            if np.size(self.airhorizons_depth) > 0:
                resi_airage = lu_solve(self.airhorizons_lu_piv, resi_airage)
            resi_airint = (self.fct_airage(self.airintervals_depthbot)-\
                           self.fct_airage(self.airintervals_depthtop)-\
                           self.airintervals_duration)/self.airintervals_sigma
            if np.size(self.airintervals_depthtop) > 0:
                resi_airint = lu_solve(self.airintervals_lu_piv, resi_airint)
            resi_delta_depth = (self.fct_delta_depth(self.delta_depth_depth)-\
                                self.delta_depth_delta_depth)/self.delta_depth_sigma
            if np.size(self.delta_depth_depth) > 0:
                resi_delta_depth = lu_solve(self.delta_depth_lu_piv, resi_delta_depth)
            return np.concatenate((resi_age, resi_airage,
                                   resi_iceint, resi_airint, resi_delta_depth))
        else:
            return np.concatenate((resi_age, resi_iceint))

    def cost_function(self):
        """Calculate the cost function."""
        cost = np.dot(self.residuals, np.transpose(self.residuals))
        return cost

    def jacobian(self):
        """Calculate the jacobian."""
        epsilon = np.sqrt(np.diag(self.cov))/100000000.
        model0 = self.model(self.variables)
        jacob = np.empty((np.size(model0), np.size(self.variables)))
        for i in np.arange(np.size(self.variables)):
            var = self.variables+0
            var[i] = var[i]+epsilon[i]
            model1 = self.model(var)
            with np.errstate(invalid='ignore'):
                jacob[:, i] = (model1-model0)/epsilon[i]
        model0 = self.model(self.variables)
        return jacob

    def optimisation(self):
        """Optimize a site."""
        self.variables, self.cov = leastsq(self.residuals, self.variables, full_output=1)
        print(self.variables)
        print(self.cov)
        return self.variables, self.cov

    def sigma(self):
        """Calculate the error of various variables."""
        jacob = self.jacobian()

        index = 0
        c_model = np.dot(jacob[index:index+np.size(self.age), :], np.dot(self.cov,\
                               np.transpose(jacob[index:index+np.size(self.age), :])))
        self.sigma_age = np.sqrt(np.diag(c_model))
        index = index+np.size(self.age)
        c_model = np.dot(jacob[index:index+np.size(self.accu), :], np.dot(self.cov,\
                               np.transpose(jacob[index:index+np.size(self.accu), :])))
        self.sigma_accu = np.sqrt(np.diag(c_model))
        index = index+np.size(self.accu)
        c_model = np.dot(jacob[index:index+np.size(self.icelayerthick), :], np.dot(self.cov,\
                               np.transpose(jacob[index:index+np.size(self.icelayerthick), :])))
        self.sigma_icelayerthick = np.sqrt(np.diag(c_model))
        index = index+np.size(self.icelayerthick)

        self.sigma_accu_model = np.interp((self.age_model[1:]+self.age_model[:-1])/2,
                                          self.corr_a_age, self.sigmap_corr_a)

        if self.archive == 'icecore':
            c_model = np.dot(jacob[index:index+np.size(self.airage), :], np.dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.airage), :])))
            self.sigma_airage = np.sqrt(np.diag(c_model))
            index = index+np.size(self.airage)
            c_model = np.dot(jacob[index:index+np.size(self.delta_depth), :], np.dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.delta_depth), :])))
            self.sigma_delta_depth = np.sqrt(np.diag(c_model))
            index = index+np.size(self.delta_depth)
            c_model = np.dot(jacob[index:index+np.size(self.tau), :], np.dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.tau), :])))
            self.sigma_tau = np.sqrt(np.diag(c_model))
            index = index+np.size(self.tau)
            c_model = np.dot(jacob[index:index+np.size(self.lid), :], np.dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.lid), :])))
            self.sigma_lid = np.sqrt(np.diag(c_model))
            index = index+np.size(self.lid)
            c_model = np.dot(jacob[index:index+np.size(self.airlayerthick), :], np.dot(self.cov,\
                                   np.transpose(jacob[index:index+np.size(self.airlayerthick), :])))
            self.sigma_airlayerthick = np.sqrt(np.diag(c_model))

            self.sigma_lid_model = np.interp(self.age_model, self.corr_lid_age,
                                             self.sigmap_corr_lid)
            self.sigma_tau_model = np.interp(self.depth_mid, self.corr_tau_depth,
                                             self.sigmap_corr_tau)


    def figures(self):
        """Build the figures of a site."""

        fig, ax1 = mpl.subplots()
        mpl.title(self.label+' Deposition rate')
        mpl.xlabel('Optimized age (yr)')
        mpl.ylabel('Deposition rate (m/yr)')
        if pccfg.show_initial:
            mpl.step(self.age, np.concatenate((self.a_init, np.array([self.a_init[-1]]))),
                     color=pccfg.color_init, where='post', label='Initial')
        mpl.step(self.age, np.concatenate((self.a_model, np.array([self.a_model[-1]]))),
                 color=pccfg.color_mod, where='post', label='Prior')
        mpl.step(self.age, np.concatenate((self.accu, np.array([self.accu[-1]]))),
                 color=pccfg.color_opt,
                 where='post', label='Posterior +/-$\sigma$')
        mpl.fill_between(self.age[:-1], self.accu-self.sigma_accu, self.accu+self.sigma_accu,
                         color=pccfg.color_ci)
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((self.age_top, x_up, y_low, y_up))
        ax2 = ax1.twinx()
        ax2.plot((self.corr_a_age[1:]+self.corr_a_age[:-1])/2, 
                 self.corr_a_age[1:]-self.corr_a_age[:-1], label='resolution',
                 color=pccfg.color_resolution)
        ax2.set_ylabel('resolution (yr)')
        ax2.spines['right'].set_color(pccfg.color_resolution)
        ax2.yaxis.label.set_color(pccfg.color_resolution)
        ax2.tick_params(axis='y', colors=pccfg.color_resolution)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        printed_page = PdfPages(pccfg.datadir+self.label+'/deposition.pdf')
        printed_page.savefig(fig)
        printed_page.close()
        if not pccfg.show_figures:
            mpl.close()

        fig, ax1 = mpl.subplots()
        if self.archive == 'icecore':
            mpl.title(self.label+' ice age')
        else:
            mpl.title(self.label+' age')
        mpl.xlabel('age (yr b1950)')
        mpl.ylabel('depth (m)')
        if pccfg.show_initial:
            mpl.plot(self.age_init, self.depth, color=pccfg.color_init, label='Initial')
        if np.size(self.icehorizons_depth) > 0:
            mpl.errorbar(self.icehorizons_age, self.icehorizons_depth, color=pccfg.color_obs,
                         xerr=self.icehorizons_sigma, linestyle='', marker='o', markersize=2,
                         label="dated horizons")
        for i in range(np.size(self.iceintervals_duration)):
            y_low = self.iceintervals_depthtop[i]
            y_up = self.iceintervals_depthbot[i]
            x_low = self.fct_age(y_low)
            x_up = x_low+self.iceintervals_duration[i]
            xseries = np.array([x_low, x_up, x_up, x_low, x_low])
            yseries = np.array([y_low, y_low, y_up, y_up, y_low])
            if i == 0:
                mpl.plot(xseries, yseries, color=pccfg.color_di, label="dated intervals")
                mpl.errorbar(x_up, y_up, color=pccfg.color_di, xerr=self.iceintervals_sigma[i],
                             capsize=1)
            else:
                mpl.plot(xseries, yseries, color=pccfg.color_di)
                mpl.errorbar(x_up, y_up, color=pccfg.color_di, xerr=self.iceintervals_sigma[i],
                             capsize=1)
        mpl.plot(self.age_model, self.depth, color=pccfg.color_mod, label='Prior')
        mpl.plot(self.age, self.depth, color=pccfg.color_opt, label='Posterior +/- 1$\sigma$')
        mpl.fill_betweenx(self.depth, self.age-self.sigma_age, self.age+self.sigma_age,
                          color=pccfg.color_ci)
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((self.age_top, x_up, self.depth[-1], self.depth[0]))
        ax2 = ax1.twiny()
        ax2.plot(self.sigma_age, self.depth, color=pccfg.color_sigma,
                 label='1$\sigma$')
        x_low, x_up, y_low, y_up = mpl.axis()
        mpl.axis((0., x_up, y_low, y_up))
        ax2.set_xlabel('1$\sigma$ uncertainty (yr)')
        ax2.spines['top'].set_color(pccfg.color_sigma)
        ax2.xaxis.label.set_color(pccfg.color_sigma)
        ax2.tick_params(axis='x', colors=pccfg.color_sigma)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        if self.archive == 'icecore':
            printed_page = PdfPages(pccfg.datadir+self.label+'/ice_age.pdf')
        else:
            printed_page = PdfPages(pccfg.datadir+self.label+'/age.pdf')
        printed_page.savefig(fig)
        printed_page.close()
        if not pccfg.show_figures:
            mpl.close()

        if self.archive == 'icecore':

            fig, ax = mpl.subplots()
            mpl.title(self.label+' ice layer thickness')
            mpl.xlabel('thickness of layers (m/yr)')
            mpl.ylabel('Depth (m)')
            if pccfg.show_initial:
                mpl.plot(self.icelayerthick_init, self.depth_mid, color=pccfg.color_init,
                         label='Initial')
            mpl.plot(self.icelayerthick_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.icelayerthick, self.depth_mid, color=pccfg.color_opt,
                     label='Posterior +/-$\sigma$')
            mpl.fill_betweenx(self.depth_mid, self.icelayerthick-self.sigma_icelayerthick,
                              self.icelayerthick+self.sigma_icelayerthick, color=pccfg.color_ci)
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((0, x_up, self.depth[-1], self.depth[0]))
            mpl.legend(loc="best")
            printed_page = PdfPages(pccfg.datadir+self.label+'/ice_layer_thickness.pdf')
            printed_page.savefig(fig)
            printed_page.close()
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' thinning')
            mpl.xlabel('Thinning')
            mpl.ylabel('Depth')
            if pccfg.show_initial:
                mpl.plot(self.tau_init, self.depth_mid, color=pccfg.color_init, label='Initial')
            mpl.plot(self.tau_model, self.depth_mid, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.tau, self.depth_mid, color=pccfg.color_opt, label='Posterior +/-$\sigma$')
            mpl.fill_betweenx(self.depth_mid, self.tau-self.sigma_tau, self.tau+self.sigma_tau,
                              color=pccfg.color_ci)
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
            ax2 = ax1.twiny()
            ax2.plot(self.corr_tau_depth[1:]-self.corr_tau_depth[:-1], 
                     (self.corr_tau_depth[1:]+self.corr_tau_depth[:-1])/2, label='resolution',
                     color=pccfg.color_resolution)
            ax2.set_xlabel('resolution (m)')
            ax2.spines['top'].set_color(pccfg.color_resolution)
            ax2.xaxis.label.set_color(pccfg.color_resolution)
            ax2.tick_params(axis='x', colors=pccfg.color_resolution)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            fig.tight_layout()
            printed_page = PdfPages(pccfg.datadir+self.label+'/thinning.pdf')
            printed_page.savefig(fig)
            printed_page.close()
            if not pccfg.show_figures:
                mpl.close()

            if pccfg.show_airlayerthick:
                fig, ax = mpl.subplots()
                mpl.title(self.label+' air layer thickness')
                mpl.xlabel('thickness of annual layers (m/yr)')
                mpl.ylabel('Depth')
                if pccfg.show_initial:
                    mpl.plot(self.airlayerthick_init, self.depth_mid, color=pccfg.color_init,
                             label='Initial')
                mpl.plot(self.airlayerthick_model, self.depth_mid, color=pccfg.color_mod,
                         label='Prior')
                mpl.plot(self.airlayerthick, self.depth_mid, color=pccfg.color_opt,
                         label='Posterior +/-$\sigma$')
                mpl.fill_betweenx(self.depth_mid, self.airlayerthick-self.sigma_airlayerthick,
                                  self.airlayerthick+self.sigma_airlayerthick, color=pccfg.color_ci)
                x_low, x_up, y_low, y_up = mpl.axis()
                mpl.axis((0, 2*max(self.icelayerthick), self.depth[-1], self.depth[0]))
                mpl.legend(loc="best")
                printed_page = PdfPages(pccfg.datadir+self.label+'/air_layer_thickness.pdf')
                printed_page.savefig(fig)
                printed_page.close()
                if not pccfg.show_figures:
                    mpl.close()

            fig, ax1 = mpl.subplots()
            mpl.title(self.label+' Lock-In Depth')
            mpl.xlabel('Optimized age (yr)')
            mpl.ylabel('LID')
            if pccfg.show_initial:
                mpl.plot(self.age, self.lid_init, color=pccfg.color_init, label='Initial')
            mpl.plot(self.age, self.lid_model, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.age, self.lid, color=pccfg.color_opt, label='Posterior +/-$\sigma$')
            mpl.fill_between(self.age, self.lid-self.sigma_lid, self.lid+self.sigma_lid,
                             color=pccfg.color_ci)
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((self.age_top, x_up, y_low, y_up))
            ax2 = ax1.twinx()
            ax2.plot((self.corr_lid_age[1:]+self.corr_lid_age[:-1])/2, 
                     self.corr_lid_age[1:]-self.corr_lid_age[:-1], label='resolution',
                     color=pccfg.color_resolution)
            ax2.set_ylabel('resolution (yr)')
            ax2.spines['right'].set_color(pccfg.color_resolution)
            ax2.yaxis.label.set_color(pccfg.color_resolution)
            ax2.tick_params(axis='y', colors=pccfg.color_resolution)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            fig.tight_layout()
            printed_page = PdfPages(pccfg.datadir+self.label+'/lock_in_depth.pdf')
            printed_page.savefig(fig)
            printed_page.close()
            if not pccfg.show_figures:
                mpl.close()

            fig, ax1 = mpl.subplots()
#            mpl.figure(self.label+' air age')
            mpl.title(self.label+' air age')
            mpl.xlabel('age (yr b1950)')
            mpl.ylabel('depth (m)')
            if pccfg.show_initial:
                mpl.plot(self.airage_init, self.depth, color=pccfg.color_init, label='Initial')
            if np.size(self.airhorizons_depth) > 0:
                mpl.errorbar(self.airhorizons_age, self.airhorizons_depth, color=pccfg.color_obs,
                             xerr=self.airhorizons_sigma, linestyle='', marker='o', markersize=2,
                             label="observations")
    #        mpl.ylim(mpl.ylim()[::-1])
            for i in range(np.size(self.airintervals_duration)):
                y_low = self.airintervals_depthtop[i]
                y_up = self.airintervals_depthbot[i]
                x_low = self.fct_airage(y_low)
                x_up = x_low+self.airintervals_duration[i]
                xseries = np.array([x_low, x_up, x_up, x_low, x_low])
                yseries = np.array([y_low, y_low, y_up, y_up, y_low])
                if i == 0:
                    mpl.plot(xseries, yseries, color=pccfg.color_di, label="dated intervals")
                    mpl.errorbar(x_up, y_up, color=pccfg.color_di, xerr=self.airintervals_sigma[i],
                                 capsize=1)
                else:
                    mpl.plot(xseries, yseries, color=pccfg.color_di)
                    mpl.errorbar(x_up, y_up, color=pccfg.color_di, xerr=self.airintervals_sigma[i],
                                 capsize=1)
            mpl.plot(self.airage_model, self.depth, color=pccfg.color_mod, label='Prior')
            mpl.fill_betweenx(self.depth, self.airage-self.sigma_airage,
                              self.airage+self.sigma_airage,
                              color=pccfg.color_ci)
            mpl.plot(self.airage, self.depth, color=pccfg.color_opt,
                     label='Posterior +/- 1$\sigma$')
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((self.age_top, x_up, self.depth[-1], self.depth[0]))
            ax2 = ax1.twiny()
            ax2.plot(self.sigma_airage, self.depth, color=pccfg.color_sigma,
                     label='1$\sigma$')
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((0., x_up, y_low, y_up))
            ax2.set_xlabel('1$\sigma$ uncertainty (yr)')
            ax2.spines['top'].set_color(pccfg.color_sigma)
            ax2.xaxis.label.set_color(pccfg.color_sigma)
            ax2.tick_params(axis='x', colors=pccfg.color_sigma)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
            fig.tight_layout()
#            mpl.plot(self.sigma_airage*pccfg.scale_ageci, self.depth, color=pccfg.color_sigma,
#                     label='1$\sigma$')            
            printed_page = PdfPages(pccfg.datadir+self.label+'/air_age.pdf')
            printed_page.savefig(fig)
            printed_page.close()
            if not pccfg.show_figures:
                mpl.close()

            fig, ax = mpl.subplots()
            mpl.title(self.label+' $\Delta$depth')
            mpl.xlabel('$\Delta$depth (m)')
            mpl.ylabel('Air depth (m)')
            if pccfg.show_initial:
                mpl.plot(self.delta_depth_init, self.depth, color=pccfg.color_init, label='Initial')
            if np.size(self.delta_depth_depth) > 0:
                mpl.errorbar(self.delta_depth_delta_depth, self.delta_depth_depth,
                             color=pccfg.color_obs,
                             xerr=self.delta_depth_sigma, linestyle='', marker='o', markersize=2,
                             label="observations")
            mpl.plot(self.delta_depth_model, self.depth, color=pccfg.color_mod, label='Prior')
            mpl.plot(self.delta_depth, self.depth, color=pccfg.color_opt,
                     label='Posterior +/-$\sigma$')
            mpl.fill_betweenx(self.depth, self.delta_depth-self.sigma_delta_depth,
                              self.delta_depth+self.sigma_delta_depth, color=pccfg.color_ci)
            x_low, x_up, y_low, y_up = mpl.axis()
            mpl.axis((x_low, x_up, self.depth[-1], self.depth[0]))
            mpl.legend(loc="best")
            printed_page = PdfPages(pccfg.datadir+self.label+'/delta_depth.pdf')
            printed_page.savefig(fig)
            printed_page.close()
            if not pccfg.show_figures:
                mpl.close()


    def save(self):
        """Save various variables for a site."""
        if self.archive == 'icecore':
            output = np.vstack((self.depth, self.age, self.sigma_age, self.airage,
                                self.sigma_airage,
                                np.append(self.accu, self.accu[-1]),
                                np.append(self.sigma_accu, self.sigma_accu[-1]),
                                np.append(self.tau, self.tau[-1]),
                                np.append(self.sigma_tau, self.sigma_tau[-1]),
                                self.lid, self.sigma_lid,
                                self.delta_depth, self.sigma_delta_depth,
                                np.append(self.a_model, self.a_model[-1]),
                                np.append(self.sigma_accu_model, self.sigma_accu_model[-1]),
                                np.append(self.tau_model, self.tau_model[-1]),
                                np.append(self.sigma_tau_model, self.sigma_tau_model[-1]),
                                self.lid_model, self.sigma_lid_model,
                                np.append(self.icelayerthick, self.icelayerthick[-1]),
                                np.append(self.sigma_icelayerthick, self.sigma_icelayerthick[-1]),
                                np.append(self.airlayerthick, self.airlayerthick[-1]),
                                np.append(self.sigma_airlayerthick, self.sigma_airlayerthick[-1])))
        else:
            output = np.vstack((self.depth, self.age, self.sigma_age,
                                np.append(self.accu, self.accu[-1]),
                                np.append(self.sigma_accu, self.sigma_accu[-1]),
                                np.append(self.a_model, self.a_model[-1]),
                                np.append(self.sigma_accu_model, self.sigma_accu_model[-1])))
        with open(pccfg.datadir+self.label+'/output.txt', 'w') as file_save:
            if self.archive == 'icecore':
                file_save.write('#depth\tage\tsigma_age\tair_age\tsigma_air_age\tdeporate'
                                '\tsigma_deporate\tthinning\tsigma_thinning\tLID\tsigma_LID'
                                 '\tdelta_depth\tsigma_delta_depth\tdeporate_model'
                                 '\tsigma_deporate_model'
                                 '\tthinning_model\tsigma_thinning_model\tLID_model'
                                 '\tsigma_LID_model\ticelayerthick\tsigma_icelayerthick'
                                 '\tairlayerthick\tsigma_airlayerthick\n')
            else:
                file_save.write('#depth\tage\tsigma_age\tdeporate\tsigma_deporate\tdeporate_model'
                                '\tsigma_deporate_model\n')
            np.savetxt(file_save, np.transpose(output), delimiter='\t')
            file_save.close()
        resi_age_top = self.resi_age_top
        corr_a_age = self.corr_a_age
        corr_a = self.corr_a
        if self.archive == 'icecore':
            corr_lid_age = self.corr_lid_age
            corr_lid = self.corr_lid
            corr_tau_depth = self.corr_tau_depth
            corr_tau = self.corr_tau
        with open(pccfg.datadir+self.label+'/restart.bin', 'wb') as f:
            if self.archive == 'icecore':
                pickle.dump([resi_age_top, corr_a_age, corr_a, corr_lid_age,
                             corr_lid, corr_tau_depth, corr_tau], f)
            else:
                pickle.dump([resi_age_top, corr_a_age, corr_a], f)
             
