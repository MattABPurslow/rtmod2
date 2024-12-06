import os, glob, pdb
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS, Proj, Transformer
import rioxarray as rx

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import rgb2hex
rcParams['font.family']='Arial'
import seaborn as sns
import cartopy.crs as ccrs

fitDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/fitted'

sellersDS = {alb:{wv:{} for wv in ['vis', 'nir', 'shortwave']}
             for alb in ['BSA', 'WSA']}
spartacusDS = {alb:{wv:{} for wv in ['vis', 'nir', 'shortwave']}
             for alb in ['BSA', 'WSA']}
sellersCount = {alb:{wv:{c:0 for c in ['solved', 'unsolved']}
                for wv in ['vis', 'nir', 'shortwave']}
                for alb in ['BSA', 'WSA']}
spartacusCount = {alb:{wv:{c:0 for c in ['solved', 'unsolved']}
                  for wv in ['vis', 'nir', 'shortwave']}
                  for alb in ['BSA', 'WSA']}

for wv in ['vis', 'nir', 'shortwave']:
  for alb in ['BSA', 'WSA']:
    if alb=='BSA':
      albtype='direct'
    else:
      albtype='diffuse'
    sellersList = sorted(glob.glob(os.path.join(fitDir,
                         f'sellers.{alb}_{wv}.*')))
    spartacusList = [s.replace('sellers', 'spartacus') for s in sellersList]
    sellersAll = []
    spartacusAll = []
    sellersMean = []
    spartacusMean = []
    sellersStd = []
    spartacusStd = []
    cnt = 0
    for sellersFile, spartacusFile in zip(sellersList, spartacusList):
      cnt += 1
      print(alb, wv+':', cnt, '/', len(sellersList), end='      \r')
      sellers = pd.read_csv(sellersFile)
      spartacus = pd.read_csv(spartacusFile)
      sellers['MODIS_date'] = pd.to_datetime(sellers['MODIS_date'],
                                             format='%Y-%m-%d')
      spartacus['MODIS_date'] = pd.to_datetime(spartacus['MODIS_date'],
                                               format='%Y-%m-%d')
      sellers[f'SSA_{wv}'] = sellers[f'leaf_r_{wv}']+sellers[f'leaf_t_{wv}']
      spartacus[f'SSA_{wv}'] = spartacus[f'leaf_r_{wv}']+spartacus[f'leaf_t_{wv}']
      sellers['zenBin'] = (np.round(sellers.MODIS_zen/5)*5).astype(int)
      spartacus['zenBin'] = (np.round(spartacus.MODIS_zen/5)*5).astype(int)
      sellers['chmBin'] = (np.round(sellers.MODIS_chm/5)*5).astype(int)
      spartacus['chmBin'] = (np.round(spartacus.MODIS_chm/5)*5).astype(int)
      sellers['cvBin'] = (np.round(sellers.MODIS_cv/10)*10).astype(int)
      spartacus['cvBin'] = (np.round(spartacus.MODIS_cv/10)*10).astype(int)
      sellers['omegaBin'] = np.round(sellers.MODIS_omega/0.1)*0.1
      spartacus['omegaBin'] = np.round(spartacus.MODIS_omega/0.1)*0.1
      sellers['dateBin'] = sellers.MODIS_date.dt.month
      spartacus['dateBin'] = spartacus.MODIS_date.dt.month
      sellers['MODIS_latBin'] = (np.round(sellers.MODIS_lat)).astype(int)
      spartacus['MODIS_latBin'] = (np.round(spartacus.MODIS_lat)).astype(int)
      sellers['matched'] = ((sellers[f'alb_{albtype}_{wv}']-sellers[f'MODIS_rhog_{alb}_{wv}']).abs()<=2*sellers[f'MODIS_err_rhog_{alb}_{wv}'])&\
                           ((sellers.omega-sellers.MODIS_omega).abs()<=0.05) &\
                           ((sellers.MODIS_LAI-sellers.MODIS_MODIS_LAI) <= 0.5)&\
                           (sellers.MODIS_MODIS_LAI<5)
      spartacus['matched'] = ((spartacus[f'alb_{albtype}_{wv}']-spartacus[f'MODIS_rhog_{alb}_{wv}']).abs()<=2*spartacus[f'MODIS_err_rhog_{alb}_{wv}'])&\
                          ((spartacus.chm-spartacus.MODIS_chm).abs()<=2) &\
                          ((spartacus.cv-spartacus.MODIS_cv).abs()<=10) &\
                          ((spartacus.MODIS_LAI-spartacus.MODIS_MODIS_LAI) <= 0.5)&\
                          (spartacus.MODIS_MODIS_LAI<5)
      if sellers.matched.sum()==0:
        sellersCount[alb][wv]['unsolved'] += 1
      else:
        sellersCount[alb][wv]['solved'] += 1
      if spartacus.matched.sum()==0:
        spartacusCount[alb][wv]['unsolved'] += 1
      else:
        spartacusCount[alb][wv]['solved'] += 1
      sellersAll.append(sellers)
      spartacusAll.append(spartacus)
    sellersAll = pd.concat(sellersAll)
    spartacusAll = pd.concat(spartacusAll)
    for c in sellersAll.columns:
      if c != 'MODIS_date':
        sellersAll[c] = pd.to_numeric(sellersAll[c])
    for c in spartacusAll.columns:
      if c != 'MODIS_date':
        spartacusAll[c] = pd.to_numeric(spartacusAll[c])
    sellersDS[alb][wv] = sellersAll.loc[sellersAll.matched]
    spartacusDS[alb][wv] = spartacusAll[spartacusAll.matched]

print('Sellers count:', sellersCount)
print('SPARTACUS count:', spartacusCount)

from matplotlib.cm import gist_earth as cmap
cDict = {'Sellers': rgb2hex(cmap(0.2)),
         'SPARTACUS': rgb2hex(cmap(0.5)),
         'Prior': rgb2hex(cmap(0.8))}
alb = ['BSA', 'WSA']
wv = ['vis', 'nir', 'shortwave']
varDict = {'leaf_r_': 'Leaf Reflectance',
           'leaf_t_': 'Leaf Transmittance',
           'SSA': 'Single Scattering Albedo',
           'SSA_': 'Single Scattering Albedo',
           'alb_diffuse_': 'Diffuse Ground Albedo',
           'alb_direct_': 'Direct Ground Albedo',
           'vis': 'Visible',
           'nir': 'Near Infrared',
           'shortwave': 'Shortwave',
           'zen': 'Solar Zenith Angle (°)',
           'chm': 'Canopy Height (m)',
           'cv': 'Canopy Cover (%)',
           'omega': 'Clumping Index',
           'MODIS_lat': 'Latitude (°N)',
           'MODIS_date': 'Month',
           'veg_scale': 'Effective Crown Diameter (m)',
           'BSA': 'Black Sky Albedo',
           'WSA': 'White Sky Albedo'}

priors = pd.DataFrame({'x_mean': [1.5,
                                  0.17, 0.13,
                                  1.0,
                                  0.1, 0.5,
                                  0.7, 0.77,
                                  2.0,
                                  0.18, 0.35],
                       'σ_x':    [5.0,
                                  0.12, 0.014,
                                  0.7,
                                  0.0959, 0.346,
                                  0.15, 0.014,
                                  1.5,
                                  0.2, 0.25]},
                       index =   ['LAI',
                                  'SSA_vis', 'SSA_vis_greenleaf',
                                  'asymmetry_vis',
                                  'alb_vis', 'alb_vis_snow',
                                  'SSA_nir', 'SSA_nir_greenleaf',
                                  'asymmetry_nir',
                                  'alb_nir', 'alb_nir_snow'])

border = gpd.read_file('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ne_10m_admin_0_countries.shp')
border = border[border.ADMIN=='Finland'].to_crs(CRS.from_epsg("4326"))
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

for albi in alb:
  for wvi in wv[:-1]:
    ssa_vmin = np.floor(np.nanmin([sellersDS[albi][wvi][f'SSA_{wvi}'].min(),
                               spartacusDS[albi][wvi][f'SSA_{wvi}'].min()])/.1)*.1
    ssa_vmax = np.ceil(np.nanmax([sellersDS[albi][wvi][f'SSA_{wvi}'].max(),
                       spartacusDS[albi][wvi][f'SSA_{wvi}'].max()])/.1)*.1
    leaf_r_vmin = np.floor(np.nanmin([sellersDS[albi][wvi][f'leaf_r_{wvi}'].min(),
                           spartacusDS[albi][wvi][f'leaf_r_{wvi}'].min()])/.1)*.1
    leaf_r_vmax = np.ceil(np.nanmax([sellersDS[albi][wvi][f'leaf_r_{wvi}'].max(),
                          spartacusDS[albi][wvi][f'leaf_r_{wvi}'].max()])/.1)*.1
    leaf_t_vmin = np.floor(np.nanmin([sellersDS[albi][wvi][f'leaf_t_{wvi}'].min(),
                           spartacusDS[albi][wvi][f'leaf_t_{wvi}'].min()])/.1)*.1
    leaf_t_vmax = np.ceil(np.nanmax([sellersDS[albi][wvi][f'leaf_t_{wvi}'].max(),
                          spartacusDS[albi][wvi][f'leaf_t_{wvi}'].max()])/.1)*.1
    fig = plt.figure(figsize=(3.5,8), layout='compressed')
    ax = np.array([[plt.subplot(321, projection=ccrs.PlateCarree()),
                    plt.subplot(323, projection=ccrs.PlateCarree()),
                    plt.subplot(325, projection=ccrs.PlateCarree())],
                   [plt.subplot(322, projection=ccrs.PlateCarree()),
                    plt.subplot(324, projection=ccrs.PlateCarree()),
                    plt.subplot(326, projection=ccrs.PlateCarree())]])
    ax[0][0].set_title('Sellers')
    ax[1][0].set_title('SPARTACUS')
    ssa = ax[0][0].scatter(sellersDS[albi][wvi].MODIS_lon, sellersDS[albi][wvi].MODIS_lat, c=sellersDS[albi][wvi][f'SSA_{wvi}'], transform=ccrs.PlateCarree(), s=3, vmin=ssa_vmin, vmax=ssa_vmax, zorder=-1)
    ax[1][0].scatter(spartacusDS[albi][wvi].MODIS_lon, spartacusDS[albi][wvi].MODIS_lat, c=spartacusDS[albi][wvi][f'SSA_{wvi}'], transform=ccrs.PlateCarree(), s=3, vmin=ssa_vmin, vmax=ssa_vmax, zorder=-1)
    border.boundary.plot(ax=ax[0][0], fc='none', ec='k', zorder=5)
    border.boundary.plot(ax=ax[1][0], fc='none', ec='k', zorder=5)
    plt.colorbar(ssa, ax=ax[:,0], label=' '.join([varDict[wvi], varDict['SSA']]), orientation='vertical')
    leaf_r = ax[0][1].scatter(sellersDS[albi][wvi].MODIS_lon, sellersDS[albi][wvi].MODIS_lat, c=sellersDS[albi][wvi][f'leaf_r_{wvi}'], transform=ccrs.PlateCarree(), s=3, vmin=leaf_r_vmin, vmax=leaf_r_vmax, zorder=-1)
    ax[1][1].scatter(spartacusDS[albi][wvi].MODIS_lon, spartacusDS[albi][wvi].MODIS_lat, c=spartacusDS[albi][wvi][f'leaf_r_{wvi}'], transform=ccrs.PlateCarree(), s=3, vmin=leaf_r_vmin, vmax=leaf_r_vmax, zorder=-1)
    border.boundary.plot(ax=ax[0][1], fc='none', ec='k', zorder=5)
    border.boundary.plot(ax=ax[1][1], fc='none', ec='k', zorder=5)
    plt.colorbar(leaf_r, ax=ax[:,1], label=' '.join([varDict[wvi], varDict['leaf_r_']]), orientation='vertical')
    leaf_t = ax[0][2].scatter(sellersDS[albi][wvi].MODIS_lon, sellersDS[albi][wvi].MODIS_lat, c=sellersDS[albi][wvi][f'leaf_t_{wvi}'], transform=ccrs.PlateCarree(), s=3, vmin=leaf_t_vmin, vmax=leaf_t_vmax, zorder=-1)
    ax[1][2].scatter(spartacusDS[albi][wvi].MODIS_lon, spartacusDS[albi][wvi].MODIS_lat, c=spartacusDS[albi][wvi][f'leaf_t_{wvi}'], transform=ccrs.PlateCarree(), s=3, vmin=leaf_t_vmin, vmax=leaf_t_vmax, zorder=-1)
    border.boundary.plot(ax=ax[0][2], fc='none', ec='k', zorder=5)
    border.boundary.plot(ax=ax[1][2], fc='none', ec='k', zorder=5)
    plt.colorbar(leaf_t, ax=ax[:,2], label=' '.join([varDict[wvi], varDict['leaf_t_']]), orientation='vertical')
    for axi in np.array(ax).ravel().tolist():
      unsampled.boundary.plot(ax=axi, fc='k', alpha=.5, ec='none')
      axi.set_rasterization_zorder(0)
    for i in range(3):
      ax[0][i].set_ylabel('Latitude (°N)')
      ax[0][i].set_yticks(np.arange(60, 72, 3), crs=ccrs.PlateCarree())
    for i in range(2):
      ax[i][-1].set_xlabel('Longitude (°E)')
      ax[i][-1].set_xticks(np.arange(21, 33, 3), crs=ccrs.PlateCarree())
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/{albi}_{wvi}_map.pdf', dpi=600)
    plt.close()

fig, ax = plt.subplots(2,2,figsize=(6,6), sharex=True, sharey=True)
for i in range(len(wv[:-1])):
  for j in range(len(alb)):
    sns.histplot(sellersDS[alb[j]][wv[i]],
                 x=f'SSA_{wv[i]}',
                 ax=ax[i,j], label='Sellers',
                 stat='probability',
                 color=cDict['Sellers'],
                 bins=np.arange(0, 1.01, 0.05))
    sns.histplot(spartacusDS[alb[j]][wv[i]],
                 x=f'SSA_{wv[i]}',
                 ax=ax[i,j], label='SPARTACUS',
                 stat='probability',
                 color=cDict['SPARTACUS'],
                 bins=np.arange(0, 1.01, 0.05))
    x = np.arange(0, 1.00001, 0.01)
    y = stats.norm.pdf(x, priors.loc[f'SSA_{wv[i]}', 'x_mean'],
                          priors.loc[f'SSA_{wv[i]}', 'σ_x'])
    ax[i,j].plot(x, y/len(np.arange(0, 1.01, 0.05)), color=cDict['Prior'], label='Prior')
    ax[i,j].set_xlabel('')
    ax[i,j].set_ylabel('')
    ax[0,j].set_title(varDict[alb[j]])
    ax[-1,j].set_xlabel(varDict['SSA'])
    ax[i,0].set_ylabel(f'{varDict[wv[i]]}\nFraction of model configurations')
    ax[i,j].set_xticks(np.arange(0,1.01, 0.1), [0]+['']*9+[1])
    ax[i,j].set_xlim(0,1)
ax[0,-1].legend(loc='best', edgecolor='none', facecolor='none')
fig.tight_layout()
fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/SSA_distribution_matchedomegachmcv.pdf')
plt.close()

fig, ax = plt.subplots(2,2,figsize=(6,6), sharex=True, sharey=True)
xmax = 0
for i in range(len(wv[:-1])):
  for j in range(len(alb)):
    sns.histplot(spartacusDS[alb[j]][wv[i]],
                 x=f'veg_scale',
                 ax=ax[i,j], label='SPARTACUS',
                 stat='probability',alpha=1,
                 color=cDict['SPARTACUS'])
    sns.histplot(x=spartacusDS[alb[j]][wv[i]]['chm']/((12.9/2.7)),
                 kde=True, ax=ax[i,j],
                 color=cDict['Prior'],
                 facecolor='none', edgecolor='none',
                 stat='probability',
                 line_kws={'label':'Prior'})
    ax[i,j].set_xlabel('')
    ax[i,j].set_ylabel('')
    ax[0,j].set_title(varDict[alb[j]])
    ax[-1,j].set_xlabel(varDict['veg_scale'])
    ax[i,0].set_ylabel(f'{varDict[wv[i]]}\nFraction of model configurations')
    if spartacusDS[alb[j]][wv[i]].veg_scale.max()>xmax:
      xmax = spartacusDS[alb[j]][wv[i]].veg_scale.max()


for i in range(len(wv[:-1])):
  for j in range(len(alb)):
    ax[i,j].set_xlim(0, np.ceil(xmax))


ax[0,-1].legend(loc='best', edgecolor='none', facecolor='none')
fig.tight_layout()
fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/veg_scale_distribution_matchedomegachmcv.pdf')
plt.close()

from scipy.stats import linregress
get_r2 = lambda x, y: linregress(x, y).rvalue**2

for x in ['leaf_r_', 'leaf_t_', 'SSA_', 'alb_']:
  for varName in ['zen', 'chm', 'cv', 'omega', 'MODIS_lat', 'MODIS_date']:
    fig, ax = plt.subplots(4,3,figsize=(6,8),layout='compressed',sharex=True, sharey=True)
    for i in range(len(wv)):
      for j in range(len(alb)):
        if x[:4]=='alb_':
          if alb[j]=='BSA':
            x = 'alb_direct_'
          else:
            x = 'alb_diffuse_'
        sns.regplot(sellersDS[alb[j]][wv[i]],
                    x=f'{varName}Bin', y=x+wv[i], ax=ax[2*j,i],
                    color=cDict['Sellers'], zorder=-1,
                    scatter_kws={"s":1, "zorder":-1},
                    line_kws={"color":"k", "zorder":1})
        sns.regplot(spartacusDS[alb[j]][wv[i]],
                    x=f'{varName}Bin', y=x+wv[i], ax=ax[2*j+1,i],
                    color=cDict['SPARTACUS'], zorder=-1,
                    scatter_kws={"s":1, "zorder":-1},
                    line_kws={"color":"k", "zorder":1})
        if varName !='MODIS_date':
          r2sellers = get_r2(sellersDS[alb[j]][wv[i]][varName],
                             sellersDS[alb[j]][wv[i]][x+wv[i]])
          r2spartacus = get_r2(spartacusDS[alb[j]][wv[i]][varName],
                               spartacusDS[alb[j]][wv[i]][x+wv[i]])
          ax[2*j,i].text(0.95, 0.95, f'R² = {r2sellers:.2f}',
                         transform=ax[2*j,i].transAxes, va='top', ha='right')
          ax[2*j+1,i].text(0.95, 0.95, f'R² = {r2spartacus:.2f}',
                           transform=ax[2*j+1,i].transAxes, va='top', ha='right')
        ax[2*j,i].set_ylim(0,1)
        ax[2*j+1,i].set_ylim(0,1)
        ax[2*j,i].set_yticks(np.arange(0,1.01, 0.1), [0]+['']*9+[1])
        ax[2*j+1,i].set_yticks(np.arange(0,1.01, 0.1), [0]+['']*9+[1])
        ax[2*j,i].set_ylabel('')
        ax[2*j+1,i].set_ylabel('')
        ax[2*j,0].set_ylabel(f'Sellers {alb[j]}\n{varDict[x]}')
        ax[2*j+1,0].set_ylabel(f'SPARTACUS {alb[j]}\n{varDict[x]}')
        ax[2*j,i].set_xlabel('')
        ax[2*j+1,i].set_xlabel('')
        ax[2*j, i].set_rasterization_zorder(0)
      ax[0,i].set_title(varDict[wv[i]])
      ax[-1,i].set_xlabel(varDict[varName])
    fig.savefig(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/{x}regplot_matchedomegachmcv_{varName}.r2.pdf', dpi=600)
    plt.close()
    

