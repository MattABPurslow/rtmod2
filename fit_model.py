import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import datetime
import numpy as np
import pandas as pd
import rioxarray as rx
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import pvlib
from pyproj import CRS, Proj, Transformer
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.cm import gist_earth as cmap

from multiprocessing import Pool
from functools import partial
import tqdm
from pySellersTwoStream.sellersTwoStream import twoStream

class RTMOD(object):
  def __init__(self, rootDir, ALSfile, lidar, threshold):
    """
    Initialise by loading data with time-invariate inputs
    """
    self.rootDir = rootDir
    self.ALSfile = ALSfile
    self.lidar = lidar
    self.alsThreshold = threshold
    self.nCPUs = 12
    self.getModel()
    self.getMODIS()
    self.modis = self.modis.loc[self.modis.MODIS_LAI<5]
    self.stratifyMODIS()
    idx = np.unique(self.modis.index)
    idx = np.sort(np.random.choice(idx, 10000, replace=False))
    with Pool(self.nCPUs) as pool:
      for _ in tqdm.tqdm(pool.imap_unordered(self.getFit, idx), total=len(idx)):
        pass

  def stratifyMODIS(self):
    stratificationVars = ['zen', 'omega', 'chm', 'cv', 'MODIS_LAI']
    for c in stratificationVars:
      if c in ['zen']:
        res = 1
      elif c in ['chm', 'cv']:
        res = 1
      elif c in ['MODIS_LAI']:
        res = 0.1
      else:
        res = 0.01
      self.modis[c+'Bin'] = (self.modis[c]/res).round()*res
    self.modis = self.modis.groupby([c+'Bin' for c in stratificationVars]).sample(n=1).reset_index()
    self.modis = self.modis.set_index([c+'Bin' for c in stratificationVars]).sort_index()

  def getFit(self, pixel):
    sellers_fit = pd.DataFrame()
    spartacus_fit = pd.DataFrame()
    modis = self.modis.loc[[pixel]]
    for alb in ['BSA', 'WSA']:
      if alb=='BSA':
        albtype = 'direct'
      else:
        albtype = 'diffuse'
      for wv in ['vis', 'nir', 'shortwave']:
        sellers_fit = pd.DataFrame()
        spartacus_fit = pd.DataFrame()
        columns = [f'leaf_r_{wv}',f'leaf_t_{wv}',f'alb_{albtype}_{wv}',
                   'veg_scale','veg_fsd',
                   'omega','chm','cv','zen','MODIS_LAI','MODIS_LAI_Sigma',
                   'MODIS_LAI_eff']
        value = modis[f'MODIS_{alb}_{wv}']
        for i in range(len(value)):
          sellers_mask = ((self.model[f'Sellers_{alb}_{wv}'] - value.iloc[i]).abs() <= 0.05)&\
                  ((self.model.zen - modis.zen.iloc[i]).abs() <= 0.5)
          spartacus_mask = ((self.model[f'SPARTACUS_{alb}_{wv}'] - value.iloc[i]).abs() <= 0.05)&\
                           ((self.model.zen - modis.zen.iloc[i]).abs() <= 0.5)
          sellers_error = self.model.loc[sellers_mask, columns+[f'Sellers_{alb}_{wv}']].copy().sort_index()
          spartacus_error = self.model.loc[spartacus_mask, columns+[f'SPARTACUS_{alb}_{wv}']].copy().sort_index()
          for c in columns+['lon', 'lat', 'date', f'rhog_{alb}_{wv}', f'err_rhog_{alb}_{wv}']:
            if c in modis.columns:
              val = modis[c].iloc[i]
              sellers_error['MODIS_'+c] = val
              spartacus_error['MODIS_'+c] = val
          sellers_error[f'MODIS_{alb}_{wv}'] = modis[f'MODIS_{alb}_{wv}'].iloc[i]
          spartacus_error[f'MODIS_{alb}_{wv}'] = modis[f'MODIS_{alb}_{wv}'].iloc[i]
          sellers_fit = pd.concat([sellers_fit, sellers_error])
          spartacus_fit = pd.concat([spartacus_fit, spartacus_error])
        label = f'{alb}_{wv}.zen{int(pixel[0]):02d}_omega{pixel[1]:.2f}_chm{int(pixel[2]):02d}_cv{int(pixel[3]):02d}_LAI{pixel[4]:.2f}'
        sellers_fit.reset_index().to_csv(os.path.join(self.rootDir,f'fitted/sellers.{label}.csv'))
        spartacus_fit.reset_index().to_csv(os.path.join(self.rootDir,f'fitted/spartacus.{label}.csv'))

  def getModel(self):
    print('reading model output sample @', datetime.datetime.now(), end='\r')
    inFile = os.path.join(self.rootDir, 'model.pinty.paramspace.csv')
    names = ['leaf_r_vis','leaf_r_nir','leaf_r_shortwave','leaf_t_vis','leaf_t_nir','leaf_t_shortwave','alb_diffuse_vis','alb_diffuse_nir','alb_diffuse_shortwave','alb_direct_vis','alb_direct_nir','alb_direct_shortwave','veg_sw_ssa','veg_fraction','veg_scale','veg_fsd','Sellers_BSA_vis','Sellers_BSA_nir','Sellers_BSA_shortwave','Sellers_WSA_vis','Sellers_WSA_nir','Sellers_WSA_shortwave','omega','chm','cv','zen','MODIS_LAI','MODIS_LAI_Sigma','MODIS_LAI_eff','MODIS_Snow_Albedo','MODIS_NDSI_Snow_Cover','SPARTACUS_BSA_nir','SPARTACUS_BSA_vis','SPARTACUS_BSA_shortwave','SPARTACUS_WSA_nir','SPARTACUS_WSA_vis','SPARTACUS_WSA_shortwave']
    df_test = pd.read_csv(inFile, nrows=5)
    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    self.model = pd.read_csv(inFile, engine='c', dtype=float32_cols).round(3).dropna()
    self.columns = ['omega', 'chm', 'cv', 'zen',
                    'MODIS_LAI', 'MODIS_LAI_Sigma', 'veg_scale',
                    'leaf_r_vis', 'leaf_r_nir', 'leaf_r_shortwave',
                    'leaf_t_vis', 'leaf_t_nir', 'leaf_t_shortwave',
                    'alb_diffuse_vis', 'alb_diffuse_nir',
                    'alb_diffuse_shortwave',
                    'alb_direct_vis', 'alb_direct_nir', 'alb_direct_shortwave']
    self.model = self.model.groupby(self.columns).mean().reset_index()

  def getMODIS(self):
    """
    Read MODIS data for given day
    """
    modisList = sorted(glob.glob(os.path.join(self.rootDir, 'modis/*')))
    self.modis = pd.read_pickle(modisList[0])
    for f in modisList[1:]:
      self.modis = pd.concat([self.modis, pd.read_pickle(f)])

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/als.tif'
  rtmod = RTMOD(rootDir, ALSfile, 'icesat2.atl08.finland', '2m')
