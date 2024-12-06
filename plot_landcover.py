import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely import vectorized
from pyproj import CRS, Proj, Transformer
import xarray as xr
import rioxarray as rx

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
from matplotlib.cm import gist_earth as cmap
import seaborn as sns
import cartopy.crs as ccrs

def getMODIS(inDir):
  modisList = sorted(glob.glob(os.path.join(inDir, '*')))
  print(os.path.split(modisList[0])[1], end='\r')
  modis = pd.read_pickle(modisList[0])
  for f in modisList[1:]:
    print(os.path.split(f)[1], end='\r')
    modis = pd.concat([modis, pd.read_pickle(f)])
  return modis

if __name__=='__main__':
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
             'date': 'Month',
             'MODIS_LAI': 'MODIS LAI (m²/m²)'}

border = gpd.read_file('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ne_10m_admin_0_countries.shp')
border = border[border.ADMIN=='Finland'].to_crs(CRS.from_epsg("4326"))
LCfile = sorted(glob.glob(os.path.join('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/MCD12Q1', '*.hdf')))[-1]
LCds = rx.open_rasterio(LCfile).sel(band=1)
LCds = LCds.rename({'LC_Type1':'IGBP_LC','LC_Type5':'PFT_LC'})
MODProj = Proj('+proj=sinu +R=6371007.181')
trans = Transformer.from_proj(MODProj, CRS.from_epsg("4326").to_proj4(),
          always_xy=True)
X, Y = np.meshgrid(LCds.x, LCds.y)
lon, lat = trans.transform(X, Y)
LCds['lon'] = (('y', 'x'), lon)
LCds['lat'] = (('y', 'x'), lat)
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

fig = plt.figure(figsize=(3,4), layout='compressed')
ax = plt.subplot(111, projection=ccrs.PlateCarree())
border.boundary.plot(ax=ax, fc='none', ec='k', zorder=5)
unsampled.boundary.plot(ax=ax, fc='k', alpha=.5, ec='none')
ax.set_xticks(np.arange(21, 33, 3), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(60, 72, 3), crs=ccrs.PlateCarree())
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
fig.savefig('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_tile_map.pdf')

PFT_dict = {v: k for k, v in LCds.PFT_LC.attrs.items()
            if k not in ['_FillValue', 'scale_factor', 'add_offset', 'valid_range', 'long_name']}
getLabel = np.vectorize(lambda pft: PFT_dict[int(pft)])
LCds['PFT_label'] = (('y','x'), getLabel(LCds.PFT_LC.values))

minLon, minLat, maxLon, maxLat = border.geometry.total_bounds
getWithin = np.vectorize(lambda x, y: border.contains(Point(x,y)))
finMask = vectorized.contains(border.geometry.item(), LCds.lon.values, LCds.lat.values)
fig = plt.figure(figsize=(6,8), layout='compressed')
ax = plt.subplot(111, projection=ccrs.PlateCarree())
import matplotlib as mpl
bounds = np.arange(-.5,12.5, 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
pft = ax.pcolormesh(LCds.lon, LCds.lat, LCds.PFT_LC.where(finMask), rasterized=True, cmap=cmap, norm=norm)
border.boundary.plot(ax=ax, fc='none', ec='k', zorder=5)
unsampled.boundary.plot(ax=ax, fc='k', alpha=.5, ec='none')
ax.set_xticks(np.arange(21, 33, 3), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(60, 72, 3), crs=ccrs.PlateCarree())
ax.set_xlim(np.floor(minLon), np.ceil(maxLon))
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
cbar = fig.colorbar(pft, ax=ax, label='Plant Functional Type', ticks=range(12), extend='neither')
fig.savefig('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/plots/MODIS_landcover_map_2022.pdf')
