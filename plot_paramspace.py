import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
from matplotlib.cm import gist_earth as cmap

def getModel(inFile):
  import seaborn as sns
  inFile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/model.pinty.paramspace.csv'
  names = ['leaf_r_vis','leaf_r_nir','leaf_r_shortwave','leaf_t_vis','leaf_t_nir','leaf_t_shortwave','alb_diffuse_vis','alb_diffuse_nir','alb_diffuse_shortwave','alb_direct_vis','alb_direct_nir','alb_direct_shortwave','veg_sw_ssa','veg_fraction','veg_scale','veg_fsd','ground_sw_albedo','ground_sw_albedo_direct','Sellers_BSA_vis','Sellers_BSA_nir','Sellers_BSA_shortwave','Sellers_WSA_vis','Sellers_WSA_nir','Sellers_WSA_shortwave','omega','chm','cv','zen','MODIS_LAI','MODIS_LAI_Sigma','MODIS_LAI_eff','MODIS_Snow_Albedo','MODIS_NDSI_Snow_Cover','SPARTACUS_BSA_nir','SPARTACUS_BSA_vis','SPARTACUS_BSA_shortwave','SPARTACUS_WSA_nir','SPARTACUS_WSA_vis','SPARTACUS_WSA_shortwave']
  df_test = pd.read_csv(inFile, nrows=5)
  float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
  float32_cols = {c: np.float32 for c in float_cols}
  model = pd.read_csv(inFile, engine='c', dtype=float32_cols).round(3).dropna()
  columns = ['omega', 'chm', 'cv', 'zen',
                    'MODIS_LAI', 'MODIS_LAI_Sigma', 'veg_scale',
                    'leaf_r_vis', 'leaf_r_nir', 'leaf_r_shortwave',
                    'leaf_t_vis', 'leaf_t_nir', 'leaf_t_shortwave',
                    'alb_diffuse_vis', 'alb_diffuse_nir',
                    'alb_diffuse_shortwave',
                    'alb_direct_vis', 'alb_direct_nir', 'alb_direct_shortwave']
  return model

def getMODIS(inDir):
  modisList = sorted(glob.glob(os.path.join(inDir, '*')))
  modis = pd.read_pickle(modisList[0])
  for f in modisList[1:]:
    modis = pd.concat([modis, pd.read_pickle(f)])
  return modis

if __name__=='__main__':
  import seaborn as sns
  model = getModel('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/model.pinty.paramspace.csv')
  modis = getMODIS('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/modis')
  modis = modis.loc[modis.MODIS_LAI < 5]
  modis['chm'] = np.round(modis.chm, decimals=0)
  modis['cv'] = np.round(modis.cv, decimals=0)
  modis['omega'] = np.round(modis.omega, decimals=2)
  modis['MODIS_LAI'] = np.round(modis.MODIS_LAI, decimals=1)
  model['chm'] = np.round(model.chm, decimals=0)
  model['cv'] = np.round(model.cv, decimals=0)
  model['omega'] = np.round(model.omega, decimals=2)
  model['MODIS_LAI'] = np.round(model.MODIS_LAI, decimals=1)
  modis_count = modis.groupby(['omega', 'MODIS_LAI']).count()
  model = model.set_index(['omega', 'MODIS_LAI']).sort_index()
  sellers = []
  for idx in modis_count.index:
    N = np.min([[modis_count.loc[idx, f'MODIS_{alb}_{wv}']
                 for alb in ['BSA', 'WSA']]
                 for wv in ['vis', 'nir', 'shortwave']])
    if idx in model.index:
      for i in range(N):
        sellers.append(model.loc[idx])
  sellers = pd.concat(sellers).sample(n=int(1e6))
  modis_count = modis.groupby(['chm', 'cv', 'MODIS_LAI']).count()
  model = model.reset_index().set_index(['chm', 'cv', 'MODIS_LAI']).sort_index()
  spartacus = []
  for idx in modis_count.index:
    N = np.min([[modis_count.loc[idx, f'MODIS_{alb}_{wv}']
                 for alb in ['BSA', 'WSA']]
                 for wv in ['vis', 'nir', 'shortwave']])
    if idx in model.index:
      for i in range(N):
        spartacus.append(model.loc[idx])
  spartacus = pd.concat(spartacus).sample(n=int(1e6))
  cDict = {'Sellers': cmap(0.2),
           'SPARTACUS': cmap(0.5),
           'MODIS': cmap(0.8)}
  print('plotting')
  fig, ax = plt.subplots(2,2, figsize=(6,8), layout='compressed')
  alb = ['BSA', 'WSA']
  for i in range(len(alb)):
    print(f'Sellers {alb[i]}')
    sns.kdeplot(sellers, x=f'Sellers_{alb[i]}_vis', y=f'Sellers_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['Sellers'], linestyles=['-', ':'])
    sns.kdeplot(sellers, x=f'Sellers_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['Sellers'])
    print(f'SPARTACUS {alb[i]}')
    sns.kdeplot(spartacus, x=f'SPARTACUS_{alb[i]}_vis', y=f'SPARTACUS_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['SPARTACUS'], linestyles=['-', ':'])
    sns.kdeplot(spartacus, x=f'SPARTACUS_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['SPARTACUS'])
    print(f'MODIS {alb[i]}')
    sns.kdeplot(modis, x=f'MODIS_{alb[i]}_vis', y=f'MODIS_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['MODIS'], linestyles=['-',':'])
    sns.kdeplot(modis, x=f'MODIS_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['MODIS'])
    ax[0,i].set_xlabel(f'Visible {alb[i]}')
    ax[0,i].set_ylabel(f'NIR {alb[i]}')
    ax[0,i].set_xlim(0,1)
    ax[0,i].set_ylim(0,1)
    ax[0,i].set_aspect('equal')
    ax[1,i].set_xlabel(f'Shortwave {alb[i]}')
    ax[1,i].set_ylabel(f'Density')
    ax[1,i].set_xlim(0,1)
  for mod in ['Sellers', 'SPARTACUS', 'MODIS']:
    ax[0,-1].plot([-1,-1], [-1,-1], label=mod, c=cDict[mod])
  ax[0,-1].plot([-1,-1], [-1,-1], label='2σ', ls='-', color='k')
  ax[0,-1].plot([-1,-1], [-1,-1], label='σ', ls=':', color='k')
  ax[0,-1].legend(loc='upper right', edgecolor='none', facecolor='none', ncol=2)
  fig.savefig('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/albedo_distributions.pdf')
  print('plotting')
  modisi = modis.loc[modis.date.dt.month<=5]
  fig, ax = plt.subplots(2,2, figsize=(6,8), layout='compressed')
  alb = ['BSA', 'WSA']
  for i in range(len(alb)):
    print(f'Sellers {alb[i]}')
    sns.kdeplot(sellers, x=f'Sellers_{alb[i]}_vis', y=f'Sellers_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['Sellers'], linestyles=['-', ':'])
    sns.kdeplot(sellers, x=f'Sellers_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['Sellers'])
    print(f'SPARTACUS {alb[i]}')
    sns.kdeplot(spartacus, x=f'SPARTACUS_{alb[i]}_vis', y=f'SPARTACUS_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['SPARTACUS'], linestyles=['-', ':'])
    sns.kdeplot(spartacus, x=f'SPARTACUS_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['SPARTACUS'])
    print(f'MODIS {alb[i]}')
    sns.kdeplot(modisi, x=f'MODIS_{alb[i]}_vis', y=f'MODIS_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['MODIS'], linestyles=['-',':'])
    sns.kdeplot(modisi, x=f'MODIS_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['MODIS'])
    ax[0,i].set_xlabel(f'Visible {alb[i]}')
    ax[0,i].set_ylabel(f'NIR {alb[i]}')
    ax[0,i].set_xlim(0,1)
    ax[0,i].set_ylim(0,1)
    ax[0,i].set_aspect('equal')
    ax[1,i].set_xlabel(f'Shortwave {alb[i]}')
    ax[1,i].set_ylabel(f'Density')
    ax[1,i].set_xlim(0,1)
  for mod in ['Sellers', 'SPARTACUS', 'MODIS']:
    ax[0,-1].plot([-1,-1], [-1,-1], label=mod, c=cDict[mod])
  ax[0,-1].plot([-1,-1], [-1,-1], label='2σ', ls='-', color='k')
  ax[0,-1].plot([-1,-1], [-1,-1], label='σ', ls=':', color='k')
  ax[0,-1].legend(loc='upper right', edgecolor='none', facecolor='none', ncol=2)
  fig.savefig('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/albedo_distributions_BeforeJune.pdf')
  print('plotting')
  modisi = modis.loc[modis.date.dt.month>=7]
  fig, ax = plt.subplots(2,2, figsize=(6,8), layout='compressed')
  alb = ['BSA', 'WSA']
  for i in range(len(alb)):
    print(f'Sellers {alb[i]}')
    sns.kdeplot(sellers, x=f'Sellers_{alb[i]}_vis', y=f'Sellers_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['Sellers'], linestyles=['-', ':'])
    sns.kdeplot(sellers, x=f'Sellers_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['Sellers'])
    print(f'SPARTACUS {alb[i]}')
    sns.kdeplot(spartacus, x=f'SPARTACUS_{alb[i]}_vis', y=f'SPARTACUS_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['SPARTACUS'], linestyles=['-', ':'])
    sns.kdeplot(spartacus, x=f'SPARTACUS_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['SPARTACUS'])
    print(f'MODIS {alb[i]}')
    sns.kdeplot(modisi, x=f'MODIS_{alb[i]}_vis', y=f'MODIS_{alb[i]}_nir', ax=ax[0,i], levels=[1-0.9545, 1-0.6827], color=cDict['MODIS'], linestyles=['-',':'])
    sns.kdeplot(modisi, x=f'MODIS_{alb[i]}_shortwave', ax=ax[1,i], color=cDict['MODIS'])
    ax[0,i].set_xlabel(f'Visible {alb[i]}')
    ax[0,i].set_ylabel(f'NIR {alb[i]}')
    ax[0,i].set_xlim(0,1)
    ax[0,i].set_ylim(0,1)
    ax[0,i].set_aspect('equal')
    ax[1,i].set_xlabel(f'Shortwave {alb[i]}')
    ax[1,i].set_ylabel(f'Density')
    ax[1,i].set_xlim(0,1)
  for mod in ['Sellers', 'SPARTACUS', 'MODIS']:
    ax[0,-1].plot([-1,-1], [-1,-1], label=mod, c=cDict[mod])
  ax[0,-1].plot([-1,-1], [-1,-1], label='2σ', ls='-', color='k')
  ax[0,-1].plot([-1,-1], [-1,-1], label='σ', ls=':', color='k')
  ax[0,-1].legend(loc='upper right', edgecolor='none', facecolor='none', ncol=2)
  fig.savefig('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/albedo_distributions_JulyOnwards.pdf')


