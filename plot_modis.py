import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS, Proj, Transformer
import rioxarray as rx

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
from matplotlib.cm import gist_earth as cmap
import seaborn as sns
import cartopy.crs as ccrs

from scipy.stats import linregress
get_r2 = lambda x, y: linregress(x, y).rvalue**2

def getMODIS(inDir):
  modisList = sorted(glob.glob(os.path.join(inDir, '*')))
  print(os.path.split(modisList[0])[1], end='\r')
  modis = pd.read_pickle(modisList[0])
  for f in modisList[1:]:
    print(os.path.split(f)[1], end='\r')
    modis = pd.concat([modis, pd.read_pickle(f)])
  return modis

if __name__=='__main__':
  modis = getMODIS('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/modis')
  varNames = ['dateBin', 'zen', 'chm', 'cv', 'omega', 'lat', 'MODIS_LAI']
  modis['zenBin'] = (np.round(modis.zen/5)*5).astype(int)
  modis['chmBin'] = (np.round(modis.chm/5)*5).astype(int)
  modis['cvBin'] = (np.round(modis.cv/10)*10).astype(int)
  modis['omegaBin'] = [float(f'{o:.1f}') for o in (np.round(modis.omega/0.1)*0.1)]
  modis['latBin'] = (np.round(modis.lat)).astype(int)
  modis['MODIS_LAIBin'] = [float(f'{lai:.1f}') for lai in np.round(modis.MODIS_LAI/0.5)*0.5]
  modis['dateBin'] = modis.date.dt.month

  cDict = {'Sellers': cmap(0.2),
           'SPARTACUS': cmap(0.5),
           'MODIS': cmap(0.8)}
  alb = ['BSA', 'WSA']
  wv = ['vis', 'nir', 'shortwave']  
  varDict = {'leaf_r_': 'Leaf Reflectance',
             'leaf_t_': 'Leaf Transmittance',
             'SSA': 'Single Scattering Albedo',
             'alb_diffuse_': 'Diffuse Ground Albedo',
             'alb_direct_': 'Direct Ground Albedo',
             'vis': 'Visible',
             'nir': 'Near Infrared',
             'shortwave': 'Shortwave',
             'zen': 'Solar Zenith Angle (°)',
             'chm': 'Canopy Height (m)',
             'cv': 'Canopy Cover (%)',
             'omega': 'Clumping Index',
             'lat': 'Latitude (°)',
             'dateBin': 'Month',
             'MODIS_LAI': 'MODIS LAI (m²/m²)'}
   
  for varName in varNames:
   for varName2 in varNames:
    if varNames != 'dateBin':
        modis = modis.loc[modis.MODIS_LAI < 5]
    fig, ax = plt.subplots(1,1,figsize=(6,4),layout='compressed',sharex=True, sharey=True)
    sns.regplot(modis,
                x=f'{varName}', y=f'{varName2}',
                ax=ax,
                color=cDict['MODIS'], scatter_kws={"s":1, "zorder":-1}, line_kws={"color":"k", "zorder":1})
    if (varName!='dateBin')&(varName2!='dateBin'):
          r2 = get_r2(modis[varName],
                      modis[varName2])
          ax.text(0.95, 0.95, f'R² = {r2:.2f}',
                  transform=ax.transAxes, va='top', ha='right')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('')
    ax.set_ylabel(varDict[varName2])
    ax.set_xlabel('')
    ax.set_xlabel(varDict[varName])
    ax.set_rasterization_zorder(0)
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/{varName2}_regplot_{varName}.pdf', dpi=600)
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(6,4),layout='compressed',sharex=True, sharey=True)
    sns.regplot(modis.loc[modis.dateBin>=7],
                x=f'{varName}', y=f'{varName2}',
                ax=ax,
                color=cDict['MODIS'], scatter_kws={"s":1, "zorder":-1}, line_kws={"color":"k", "zorder":1})
    if (varName!='dateBin')&(varName2!='dateBin'):
          r2 = get_r2(modis[varName],
                      modis[varName2])
          ax.text(0.95, 0.95, f'R² = {r2:.2f}',
                  transform=ax.transAxes, va='top', ha='right')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('')
    ax.set_ylabel(varDict[varName2])
    ax.set_xlabel('')
    ax.set_xlabel(varDict[varName])
    ax.set_rasterization_zorder(0)
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/{varName2}_regplot_{varName}_JulyOnwards.pdf', dpi=600)
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(6,4),layout='compressed',sharex=True, sharey=True)
    sns.regplot(modis.loc[modis.dateBin<5],
                x=f'{varName}', y=f'{varName2}',
                ax=ax,
                color=cDict['MODIS'], scatter_kws={"s":1, "zorder":-1}, line_kws={"color":"k", "zorder":1})
    if (varName!='dateBin')&(varName2!='dateBin'):
          r2 = get_r2(modis[varName],
                      modis[varName2])
          ax.text(0.95, 0.95, f'R² = {r2:.2f}',
                  transform=ax.transAxes, va='top', ha='right')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('')
    ax.set_ylabel(varDict[varName2])
    ax.set_xlabel('')
    ax.set_xlabel(varDict[varName])
    ax.set_rasterization_zorder(0)
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/{varName2}_regplot_{varName}_beforeMay.pdf', dpi=600)
    plt.close()
 
  for varName in varNames:
    fig, ax = plt.subplots(3,2,figsize=(6,8),layout='compressed',sharex=True, sharey=True)
    for i in range(len(wv)):
      for j in range(len(alb)):
        sns.regplot(modis,
                    x=f'{varName}', y=f'MODIS_{alb[j]}_{wv[i]}',
                    ax=ax[i,j],
                    color=cDict['MODIS'], scatter_kws={"s":1, "zorder":-1}, line_kws={"color":"k", "zorder":1})
        if varName!='dateBin':
          r2 = get_r2(modis[varName],
                      modis[f'MODIS_{alb[j]}_{wv[i]}'])
          ax[i,j].text(0.95, 0.95, f'R² = {r2:.2f}',
                       transform=ax[i,j].transAxes, va='top', ha='right')
        ax[i,j].set_ylim(0,1)
        ax[i,j].set_yticks(np.arange(0,1.01, 0.1), [0]+['']*9+[1])
        ax[i,j].set_ylabel('')
        ax[i,0].set_ylabel(f'MODIS {varDict[wv[i]]} Albedo')
        ax[i,j].set_xlabel('')
        ax[0,j].set_title(alb[j])
        ax[-1,j].set_xlabel(varDict[varName])
        ax[i,j].set_rasterization_zorder(0)
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_albedo_regplot_{varName}.r2.pdf', dpi=600)
    plt.close()
    fig, ax = plt.subplots(3,2,figsize=(6,8),layout='compressed',sharex=True, sharey=True)
    for i in range(len(wv)):
      for j in range(len(alb)):
        sns.regplot(modis.loc[modis.dateBin>=7],
                    x=f'{varName}', y=f'MODIS_{alb[j]}_{wv[i]}',
                    ax=ax[i,j],
                    color=cDict['MODIS'], scatter_kws={"s":1, "zorder":-1}, line_kws={"color":"k", "zorder":1})
        if varName != 'dateBin':
          r2 = get_r2(modis[varName],
                      modis[f'MODIS_{alb[j]}_{wv[i]}'])
          ax[i,j].text(0.95, 0.95, f'R² = {r2:.2f}',
                       transform=ax[i,j].transAxes, va='top', ha='right')
        ax[i,j].set_ylim(0,1)
        ax[i,j].set_yticks(np.arange(0,1.01, 0.1), [0]+['']*9+[1])
        ax[i,j].set_ylabel('')
        ax[i,0].set_ylabel(f'MODIS {varDict[wv[i]]} Albedo')
        ax[i,j].set_xlabel('')
        ax[0,j].set_title(alb[j])
        ax[-1,j].set_xlabel(varDict[varName])
        ax[i,j].set_rasterization_zorder(0)
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_albedo_regplot_{varName}_JulyOnwards.r2.pdf', dpi=600)
    plt.close()
    fig, ax = plt.subplots(3,2,figsize=(6,8),layout='compressed',sharex=True, sharey=True)
    for i in range(len(wv)):
      for j in range(len(alb)):
        sns.regplot(modis.loc[modis.dateBin<5],
                    x=f'{varName}', y=f'MODIS_{alb[j]}_{wv[i]}',
                    ax=ax[i,j],
                    color=cDict['MODIS'], scatter_kws={"s":1, "zorder":-1}, line_kws={"color":"k", "zorder":1})
        if varName != 'dateBin':
          r2 = get_r2(modis[varName],
                      modis[f'MODIS_{alb[j]}_{wv[i]}'])
          ax[i,j].text(0.95, 0.95, f'R² = {r2:.2f}',
                       transform=ax[i,j].transAxes, va='top', ha='right')
        ax[i,j].set_ylim(0,1)
        ax[i,j].set_yticks(np.arange(0,1.01, 0.1), [0]+['']*9+[1])
        ax[i,j].set_ylabel('')
        ax[i,0].set_ylabel(f'MODIS {varDict[wv[i]]} Albedo')
        ax[i,j].set_xlabel('')
        ax[0,j].set_title(alb[j])
        ax[-1,j].set_xlabel(varDict[varName])
        ax[i,j].set_rasterization_zorder(0)
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_albedo_regplot_{varName}_beforeMay.r2.pdf', dpi=600)
    plt.close()

border = gpd.read_file('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ne_10m_admin_0_countries.shp')
border = border[border.ADMIN=='Finland'].to_crs(CRS.from_epsg("4326"))
minLon, minLat, maxLon, maxLat = border.geometry.total_bounds
minLon, maxLon = np.floor(minLon), np.ceil(maxLon)
minLat, maxLat = np.floor(minLat), np.ceil(maxLat)
LCfile = sorted(glob.glob(os.path.join('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MCD12Q1', '*.hdf')))[-1]
LCds = rx.open_rasterio(LCfile).sel(band=1)
MODProj = Proj('+proj=sinu +R=6371007.181')
trans = Transformer.from_proj(MODProj, CRS.from_epsg("4326").to_proj4(),
          always_xy=True)
X, Y = np.meshgrid(LCds.x, LCds.y)
x0, x1  = LCds.x.min(), LCds.x.max()
y0, y1 = LCds.y.min(), LCds.y.max()
west = [Point(x, y0) for x in np.linspace(x0, x1, 1000)]
north = [Point(x1, y) for y in np.linspace(y0, y1, 1000)]
east = [Point(x, y1) for x in np.linspace(x1, x0, 1000)]
south = [Point(x0, y) for y in np.linspace(y1, y0, 1000)]
tile = gpd.GeoDataFrame(geometry=[Polygon(west+north+east+south)],
                        crs=CRS.from_proj4(MODProj.to_proj4()))
tile = tile.to_crs(CRS.from_epsg("4326"))
unsampled = border.reset_index().difference(tile.geometry)

vDict = {'cv':{'vmin':0, 'vmax':100},
         'chm':{'vmin':0, 'vmax':np.ceil(modis.chm.max()/5)*5},
         'MODIS_LAI':{'vmin':0, 'vmax':4},
         'omega':{'vmin':0, 'vmax':1}}

fig = plt.figure(figsize=(6,8), layout='compressed')
ax = plt.subplot(111, projection=ccrs.PlateCarree())
modisi = modis.groupby(['y', 'x'])[[varName]].count().to_xarray()
s = modisi['MODIS_LAI'].plot(ax=ax, transform=ccrs.Projection(CRS.from_proj4(MODProj.to_proj4())), vmin=vDict[varName]['vmin'], add_colorbar=False, zorder=-1)
fig.colorbar(s, ax=ax, label='Number of LAI observations')
border.boundary.plot(ax=ax, fc='none', ec='k', zorder=5)
unsampled.boundary.plot(ax=ax, fc='k', alpha=.5, ec='none')
ax.set_ylabel('Latitude (°N)')
ax.set_yticks(np.arange(minLat, maxLat+1, 1), crs=ccrs.PlateCarree())
ax.set_xlabel('Longitude (°E)')
ax.set_xticks(np.arange(minLon, maxLon+1, 1), crs=ccrs.PlateCarree())
ax.set_xlim(minLon, maxLon)
ax.set_ylim(minLat, maxLat)
ax.set_rasterization_zorder(0)
fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_LAI_count_map.pdf', dpi=600)
plt.close()

for varName in ['cv', 'chm', 'MODIS_LAI', 'omega']:
  fig = plt.figure(figsize=(6,8), layout='compressed')
  ax = plt.subplot(111, projection=ccrs.PlateCarree())
  #modisi = modis.groupby(['y', 'x'])[[varName]].max().to_xarray()
  s = ax.scatter(modis.lon, modis.lat, c=modis[varName], transform=ccrs.PlateCarree(), vmin=vDict[varName]['vmin'], vmax=vDict[varName]['vmax'], s=1, zorder=-1) 
  #s = modisi[varName].plot(ax=ax, transform=ccrs.Projection(CRS.from_proj4(MODProj.to_proj4())), vmin=vDict[varName]['vmin'], vmax=vDict[varName]['vmax'], add_colorbar=False)
  fig.colorbar(s, ax=ax, label=varDict[varName])
  border.boundary.plot(ax=ax, fc='none', ec='k', zorder=5)
  unsampled.boundary.plot(ax=ax, fc='k', alpha=.5, ec='none')
  ax.set_ylabel('Latitude (°N)')
  ax.set_yticks(np.arange(minLat, maxLat+1, 1), crs=ccrs.PlateCarree())
  ax.set_xlabel('Longitude (°E)')
  ax.set_xticks(np.arange(minLon, maxLon+1, 1), crs=ccrs.PlateCarree())
  ax.set_xlim(minLon, maxLon)
  ax.set_ylim(minLat, maxLat)
  ax.set_rasterization_zorder(0)
  fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_{varName}_map.pdf', dpi=600)
  plt.close()
  fig = plt.figure(figsize=(6,8), layout='compressed')
  ax = plt.subplot(111, projection=ccrs.PlateCarree())
  modisi = modis.loc[modis.dateBin==7]#.groupby(['x','y'])[[varName]].max().to_xarray()
  s = ax.scatter(modisi.lon, modisi.lat, c=modisi[varName], transform=ccrs.PlateCarree(), vmin=vDict[varName]['vmin'], vmax=vDict[varName]['vmax'], s=1, zorder=-1)
  #s = modisi[varName].plot(ax=ax, transform=ccrs.Projection(CRS.from_proj4(MODProj.to_proj4())), vmin=vDict[varName]['vmin'], vmax=vDict[varName]['vmax'], add_colorbar=False)
  fig.colorbar(s, ax=ax, label=varDict[varName])
  border.boundary.plot(ax=ax, fc='none', ec='k', zorder=5)
  unsampled.boundary.plot(ax=ax, fc='k', alpha=.5, ec='none')
  ax.set_ylabel('Latitude (°N)')
  ax.set_yticks(np.arange(minLat, maxLat+1, 1), crs=ccrs.PlateCarree())
  ax.set_xlabel('Longitude (°E)')
  ax.set_xticks(np.arange(minLon, maxLon+1, 1), crs=ccrs.PlateCarree())
  ax.set_xlim(minLon, maxLon)
  ax.set_ylim(minLat, maxLat)
  ax.set_rasterization_zorder(0)
  fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_{varName}_map_July.pdf', dpi=600)
  plt.close()

for varName in ['cv', 'chm', 'MODIS_LAI', 'omega']:
  fig = plt.figure(figsize=(6,4), layout='compressed')
  ax = plt.subplot(111)
  sns.histplot(modis, x=varName, stat='probability')
  ax.set_xlim(vDict[varName]['vmin'], vDict[varName]['vmax'])
  ax.set_xlabel(varDict[varName])
  ax.set_ylabel('Fraction of pixels')
  fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_{varName}_distribution.pdf', dpi=600)
  plt.close()

