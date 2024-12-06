import glob, os, sys, pdb
sys.path.append('/home/s1503751/src/')
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rioxarray as rx
import xarray as xr
import pvlib
from pyproj import CRS, Proj, Transformer
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
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
    self.getMODISdirs()
    self.getDOYs()
    self.getLandCover()
    self.getMODIS()

  def fitReflectanceRatio(self, modis):
    dateMask = modis.date==np.datetime64(self.date.strftime('%Y-%m-%d'))
    dlat, dlon = 0.5, 0.5
    modis['latBin'] = np.round(modis.lat/dlat)*dlat
    modis['lonBin'] = np.round(modis.lon/dlon)*dlon
    lats = np.unique(modis.latBin)
    lons = np.unique(modis.lonBin)
    for lat in lats:
      for lon in lons:
        pixelMask = (modis.latBin==lat)&(modis.lonBin==lon)
        df = modis.loc[dateMask&pixelMask]
        if df.shape[0]>10:
          alb = ['BSA', 'WSA']
          wv = ['vis', 'nir', 'shortwave']
          for i in range(len(wv)):
            for j in range(len(alb)):
              result = linregress(df.cv/100, df[f'MODIS_{alb[j]}_{wv[i]}'])
              rhog = result.intercept
              rhov = result.intercept + result.slope
              err_rhog = result.intercept_stderr
              modis.loc[dateMask&pixelMask, f'rhog_{alb[j]}_{wv[i]}'] = rhog
              modis.loc[dateMask&pixelMask, f'err_rhog_{alb[j]}_{wv[i]}'] = err_rhog
              modis.loc[dateMask&pixelMask, f'rhov_{alb[j]}_{wv[i]}'] = rhov
    return modis
  
  def getMODISdirs(self):
    """
    Identify directories containing MODIS data
    """
    self.LCdir = os.path.join(self.rootDir, 'MCD12Q1')
    self.LAIdir = os.path.join(self.rootDir, 'MCD15A3H')
    self.Snowdir = os.path.join(self.rootDir, 'MOD10A1')
    self.Albedodir = os.path.join(self.rootDir, 'MCD43A3')

  def getMODIS(self):
    """
    Read MODIS data for given day
    """
    border = gpd.read_file('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ne_10m_admin_0_countries.shp')
    self.border = border[border.ADMIN=='Finland'].to_crs(CRS.from_epsg("4326"))
    del border
    with Pool(12) as pool:
      for _ in  tqdm.tqdm(pool.imap_unordered(self.pickleMODIS, self.doyList), total=len(self.doyList)):
        pass

  def pickleMODIS(self, doy):
      print(doy, end='\n')
      self.doy = doy
      self.date = datetime.datetime.strptime(self.doy, '%Y%j')\
                                   .replace(tzinfo=datetime.timezone.utc)
      self.getMODISfiles()
      self.getLAI()
      self.getSnow()
      self.getAlbedo()
      self.mergeMODIS()

  def getMODISfiles(self):
    """
    Get lists of MODIS files
    """
    self.LAIfile = glob.glob(os.path.join(self.LAIdir,
                                          '*A%s*.hdf' % self.doy))[0]
    self.Snowfile = glob.glob(os.path.join(self.Snowdir,
                                           '*A%s*.hdf' % self.doy))[0]
    self.Albedofile = glob.glob(os.path.join(self.Albedodir,
                                             '*A%s*.hdf' % self.doy))[0]

  def getDOYs(self):
    """
    Idenitify dates with comprehensive MODIS data
    """
    LAIfiles = sorted(glob.glob(os.path.join(self.LAIdir, '*.hdf*')))
    Snowfiles = sorted(glob.glob(os.path.join(self.Snowdir, '*.hdf*')))
    Albedofiles = sorted(glob.glob(os.path.join(self.Albedodir, '*.hdf*')))
    LAIdoy = [os.path.split(f)[-1].split('.')[1][1:] for f in LAIfiles]
    Snowdoy = [os.path.split(f)[-1].split('.')[1][1:] for f in Snowfiles]
    Albedodoy = [os.path.split(f)[-1].split('.')[1][1:] for f in Albedofiles]
    self.doyList = [d for d in LAIdoy if (d in Snowdoy) & (d in Albedodoy)]
    print(len(self.doyList), 'dates from', min(self.doyList), 'to',
              max(self.doyList), end='\r')

  def getLAI(self):
    """
    Read MODIS LAI (MCD15A3H)
    """
    print('reading LAI @', datetime.datetime.now(), end='\r')
    LAI = rx.open_rasterio(self.LAIfile).sel(band=1, y=self.ySlice, x=self.xSlice)
    LAI = LAI.rename({'Lai_500m':'MODIS_LAI',
                      'FparLai_QC':'MODIS_LAI_QC',
                      'LaiStdDev_500m':'MODIS_LAI_Sigma'})
    LAI = LAI.drop(['Fpar_500m', 'FparExtra_QC', 'FparStdDev_500m'])
    LAI.coords['date'] = self.date
    LAI = LAI.expand_dims(dim={'date':1})
    checkRTMethodUsed = np.vectorize(lambda i: bin(i)[-3:-1]=='00')
    LAI['MODIS_LAI_QC'] = (('date', 'y', 'x'),
                           checkRTMethodUsed(LAI.MODIS_LAI_QC))
    LAI['MODIS_LAI'] = (LAI.MODIS_LAI-LAI.MODIS_LAI.add_offset) * \
                       LAI.MODIS_LAI.scale_factor
    LAI['MODIS_LAI_Sigma'] = (LAI.MODIS_LAI_Sigma - \
                              LAI.MODIS_LAI_Sigma.add_offset) * \
                              LAI.MODIS_LAI_Sigma
    LAI['MODIS_LAI'] = LAI.MODIS_LAI.where(LAI.MODIS_LAI<=10.)
    LAI['MODIS_LAI_Sigma'] = LAI.MODIS_LAI_Sigma.where(LAI.MODIS_LAI_Sigma<=10.)
    LAI['MODIS_LAI_eff'] = LAI.MODIS_LAI * self.LCds.omega
    ## Only use successful LAI retrievals
    LAI = LAI.where(LAI.MODIS_LAI_QC)
    ## Add DataArrays to list
    LAI.attrs.clear()
    self.LAIds = LAI

  def getSnow(self):
    """
    Read MODIS Snow Cover (MOD10A1)
    """
    print('reading snow @', datetime.datetime.now(), end='\r')
    snow = rx.open_rasterio(self.Snowfile).sel(band=1, y=self.ySlice, x=self.xSlice)
    snow = snow.rename({'NDSI_Snow_Cover':'MODIS_NDSI_Snow_Cover',
                        'Snow_Albedo_Daily_Tile':'MODIS_Snow_Albedo',
                        'NDSI_Snow_Cover_Basic_QA':'MODIS_Snow_Cover_QC'})
    snow = snow.drop(['NDSI_Snow_Cover_Algorithm_Flags_QA', 'NDSI',
                      'orbit_pnt', 'granule_pnt'])
    snow.coords['date'] = self.date
    snow = snow.expand_dims(dim={'date':1})
    snow['MODIS_Snow_Cover_QC'] = snow.MODIS_Snow_Cover_QC <= 2
    snow['MODIS_NDSI_Snow_Cover'] = snow['MODIS_NDSI_Snow_Cover']\
                             .where(snow['MODIS_NDSI_Snow_Cover'] <= 100) / 100.
    snow['MODIS_Snow_Albedo'] = snow['MODIS_Snow_Albedo']\
                              .where(snow['MODIS_Snow_Albedo'] <= 100) / 100.
    snow.attrs.clear()
    self.Snowds = snow

  def getAlbedo(self):
    """
    Read MODIS albedo (MCD43A3)
    """
    print('reading albedo @', datetime.datetime.now(), end='\r')
    Albedo = rx.open_rasterio(self.Albedofile).sel(band=1, y=self.ySlice, x=self.xSlice)
    Albedo = Albedo.rename({'Albedo_BSA_shortwave': 'MODIS_BSA_shortwave',
                            'Albedo_BSA_nir': 'MODIS_BSA_nir',
                            'Albedo_BSA_vis': 'MODIS_BSA_vis',
                            'Albedo_WSA_shortwave': 'MODIS_WSA_shortwave',
                            'Albedo_WSA_nir': 'MODIS_WSA_nir',
                            'Albedo_WSA_vis': 'MODIS_WSA_vis',
                            'BRDF_Albedo_Band_Mandatory_Quality_shortwave':
                            'MODIS_BRDF_shortwave_QC',
                            'BRDF_Albedo_Band_Mandatory_Quality_vis':
                            'MODIS_BRDF_vis_QC',
                            'BRDF_Albedo_Band_Mandatory_Quality_nir':
                            'MODIS_BRDF_nir_QC'})
    Albedo = Albedo.drop(['BRDF_Albedo_Band_Mandatory_Quality_Band1',
                          'Albedo_BSA_Band1', 'Albedo_BSA_Band2',
                          'Albedo_BSA_Band3', 'Albedo_BSA_Band4',
                          'Albedo_BSA_Band5', 'Albedo_BSA_Band6',
                          'Albedo_BSA_Band7',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band2',
                          'Albedo_WSA_Band1', 'Albedo_WSA_Band2',
                          'Albedo_WSA_Band3', 'Albedo_WSA_Band4',
                          'Albedo_WSA_Band5', 'Albedo_WSA_Band6',
                          'Albedo_WSA_Band7',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band3',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band4',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band5',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band6',
                          'BRDF_Albedo_Band_Mandatory_Quality_Band7'])
    Albedo.coords['date'] = self.date
    Albedo = Albedo.expand_dims(dim={'date':1})
    # Check Full BRDF inversion used
    Albedo['MODIS_BRDF_shortwave_QC'] = Albedo.MODIS_BRDF_shortwave_QC==0
    Albedo['MODIS_BRDF_vis_QC'] = Albedo.MODIS_BRDF_vis_QC==0
    Albedo['MODIS_BRDF_nir_QC'] = Albedo.MODIS_BRDF_nir_QC==0
    for k in list(Albedo.keys()):
      if k[-2:] != 'QC':
        Albedo[k] = (Albedo[k]-Albedo[k].add_offset)*Albedo[k].scale_factor
        Albedo[k] = Albedo[k].where(Albedo[k] <= 1.0)
    for alb in ['BSA', 'WSA']:
      for wv in ['vis', 'nir', 'shortwave']:
        Albedo[f'MODIS_{alb}_{wv}'] = Albedo[f'MODIS_{alb}_{wv}'].where(Albedo[f'MODIS_BRDF_{wv}_QC'])
    Albedo.attrs.clear()
    self.Albedods = Albedo
   
  def mergeMODIS(self):
    """
    Combine MODIS dataset into single dataset
    """
    if 'date' in self.LAIds.coords:
      self.LAIds.coords['date'] = self.LAIds.date
    self.Snowds.coords['date'] = self.Snowds.date
    self.Albedods.coords['date'] = self.Albedods.date
    lcVars = ['PFT_LC', 'lon', 'lat', 'omega', 'chm', 'cv']
    snowVars = ['MODIS_Snow_Albedo', 'MODIS_NDSI_Snow_Cover']
    laiVars = ['MODIS_LAI', 'MODIS_LAI_Sigma',
               'MODIS_LAI_eff', 'MODIS_LAI_QC']
    albedoVars = ['MODIS_BSA_vis', 'MODIS_BSA_nir', 'MODIS_BSA_shortwave',
                  'MODIS_WSA_vis', 'MODIS_WSA_nir', 'MODIS_WSA_shortwave']
    modis = xr.merge([self.LCds[lcVars],
                           self.LAIds[laiVars],
                           self.Snowds[snowVars],
                           self.Albedods[albedoVars]]).isel(date=0).copy()
    getNoon = np.vectorize(lambda lon: np.timedelta64(int((12.-(lon/15.))*60*60), 's'))
    print('getting noon @', datetime.datetime.now(), end='\r')
    utcNoon = modis.date.values + getNoon(modis.lon.values)
    print('getting zenith @', datetime.datetime.now(), end='\r')
    lat = modis.lat.values
    lon = modis.lon.values
    zen = pvlib.solarposition.get_solarposition(utcNoon.flatten(),
                                                lat.flatten(),
                                                lon.flatten(),
                                                0).zenith
    modis['zen'] = (('y', 'x'), zen.values.reshape(modis.MODIS_BSA_vis.shape))
    modis = modis.to_dataframe().dropna().reset_index()
    modis = self.fitReflectanceRatio(modis)
    modis = modis.loc[np.array([self.border.contains(Point(x, y)) for x,y in zip(modis.lon, modis.lat)])]
    modis.to_pickle(f'/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/modis/modis.{self.doy}.pkl')

  def getALS(self):
    """
    Get ALS-derived canopy height and cover
    """
    print('Retrieving ALS canopy height and cover @', datetime.datetime.now(), end='\r')
    albLon, albLat = 26.63319, 67.36198 # Location of Sodankyla IOA albedo measurement
    self.aoiLon = np.array([albLon-.25, albLon-.25, albLon+.25, albLon+.25])
    self.aoiLat = np.array([albLat-.25, albLat+.25, albLat+.25, albLat-.25])
    MODProj = Proj('+proj=sinu +R=6371007.181')
    trans = Transformer.from_crs(CRS.from_epsg("4326"), MODProj.crs, always_xy=True)
    aoiX, aoiY = trans.transform(self.aoiLon, self.aoiLat)
    if self.lidar=='als.sodankyla':
      cv = rx.open_rasterio(self.ALSfile.replace('als','finland.canopy_cover.modis')).sel(band=1)
      chm = rx.open_rasterio(self.ALSfile.replace('als','finland.ndsm.modis.%s' % self.alsThreshold)).sel(band=1)
      self.LCds['chm'] = (('y', 'x'), chm.where((chm>0.)&(chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (cv.where((cv>1.e-3)&(cv<1.)).values.astype(np.float64)*100.))
      self.xSlice = slice(aoiX.min(), aoiX.max())
      self.ySlice = slice(aoiY.max(), aoiY.min())
    elif self.lidar=='als.finland':
      cv = rx.open_rasterio(self.ALSfile.replace('als','finland.canopy_cover.modis')).sel(band=1)
      chm = rx.open_rasterio(self.ALSfile.replace('als','finland.ndsm.modis.%s' % self.alsThreshold)).sel(band=1)
      self.LCds['chm'] = (('y', 'x'), chm.where((chm>0.)&(chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (cv.where((cv>1.e-3)&(cv<1.)).values.astype(np.float64)*100.))
      self.xSlice = slice(-1e12, 1e12)
      self.ySlice = slice(1e12, -1e12)
    elif self.lidar=='icesat2.reclassified':
      ice = xr.open_dataset(os.path.join(self.rootDir, 'icesat2.nc'))
      self.LCds['chm'] = (('y', 'x'), ice.chm.where((ice.chm>0.)&(ice.chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (ice.cv.where((ice.cv>0.1)&(ice.cv<100.)).values.astype(np.float64)))
      self.xSlice = slice(aoiX.min(), aoiX.max())
      self.ySlice = slice(aoiY.max(), aoiY.min())
    elif self.lidar=='icesat2.atl08':
      ice = xr.open_dataset(os.path.join(self.rootDir, 'icesat2.atl08.nc'))
      self.LCds['chm'] = (('y', 'x'), ice.chm.where((ice.chm>0.)&(ice.chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (ice.cv.where((ice.cv>0.1)&(ice.cv<100.)).values.astype(np.float64)))
      self.xSlice = slice(aoiX.min(), aoiX.max())
      self.ySlice = slice(aoiY.max(), aoiY.min())
    elif self.lidar=='icesat2.atl08.finland':
      ice = xr.open_dataset(os.path.join(self.rootDir, 'icesat2.atl08.finland.nc'))
      self.LCds['chm'] = (('y', 'x'), ice.chm.where((ice.chm>0.)&(ice.chm<50.)).values.astype(np.float64))
      self.LCds['cv'] = (('y', 'x'), (ice.cv.where((ice.cv>0.1)&(ice.cv<100.)).values.astype(np.float64)))
      self.xSlice = slice(-1e12, 1e12)
      self.ySlice = slice(1e12, -1e12)
    self.LCds = self.LCds.sel(y=self.ySlice, x=self.xSlice)

  def getLandCover(self):
    """
    Read MODIS land cover classification (MCD12Q1)
    """
    print('reading land cover @', datetime.datetime.now(), end='\r')
    LCfile = sorted(glob.glob(os.path.join(self.LCdir,'*.hdf')))[-1]
    MODProj = Proj('+proj=sinu +R=6371007.181')
    trans = Transformer.from_crs(MODProj.crs, CRS.from_epsg("4326"),
                                 always_xy=True)
    LandCover = rx.open_rasterio(LCfile).sel(band=1)
    dropList = ['_FillValue', 'scale_factor', 'add_offset',
                'long_name', 'valid_range', 'Unclassified']
    self.lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items()
                   if k not in dropList}
    LandCover = LandCover.rename({'LC_Type1':'IGBP_LC','LC_Type5':'PFT_LC'})
    X, Y = np.meshgrid(LandCover.x, LandCover.y)
    lon, lat = trans.transform(X, Y)
    LandCover['lon'] = (('y', 'x'), lon)
    LandCover['lat'] = (('y', 'x'), lat)
    LandCover = LandCover.drop(['band', 'spatial_ref'])
    LandCover.attrs.clear()
    LandCover = LandCover.drop([k for k in list(LandCover.keys())
                                if k not in ['IGBP_LC', 'PFT_LC', 'ForestMask',
                                             'lon', 'lat']])
    LandCover['IGBP_LC'] = (('y','x'), LandCover.IGBP_LC.values.astype(np.int8))
    LandCover['PFT_LC'] = (('y','x'), LandCover.PFT_LC.values.astype(np.int8))
    self.LCds = LandCover
    self.getClumping()
    self.getALS()
    nanMask = np.isnan(self.LCds.PFT_LC.values)==False
    self.pftList = np.sort(np.unique(self.LCds.PFT_LC.values[nanMask])\
                                                     .astype(int))
    self.pftList = self.pftList[self.pftList==1]#self.pftList[self.pftList > 0]
    self.LCds['PFT_LC'] = self.LCds.PFT_LC.where(np.isin(self.LCds.PFT_LC,
                                                         self.pftList))

  def getLCdict(self, LandCover):
    """
    Get list of MODIS Plant Functional Types
    """
    dropList = ['_FillValue', 'scale_factor', 'add_offset',
                'long_name', 'valid_range', 'Unclassified']
    self.lcDict = {v: k for k, v in LandCover['LC_Type5'].attrs.items()
                   if k not in dropList}

  def getClumping(self):
    """
    Retrieve MODIS-derived clumping index
    """
    print('retrieving he et al (2012) clumping index @', datetime.datetime.now(), end='\r')
    Clumping = rx.open_rasterio(os.path.join(self.rootDir,
                                '../global_clumping_index.tif')).sel(band=1)
    LandFile2006 = sorted(glob.glob(os.path.join(self.LCdir, '*.hdf')))[0]
    LC2006 = rx.open_rasterio(LandFile2006).sel(band=1)
    ## Select Land Cover tile clumping factors
    LC2006['Clumping_Index'] = (('y', 'x'), Clumping.sel(y=LC2006.y, x=LC2006.x,
                                                     method='nearest').values)
    LC2006['Clumping_Index'] = LC2006.Clumping_Index\
                                     .where(LC2006.Clumping_Index!=255) / 100
    ## Reduce to places with same land cover type now
    noChange = self.LCds['PFT_LC']==LC2006['LC_Type5']
    LC2006 = LC2006.where(noChange)
    self.LCds['omega'] = LC2006.Clumping_Index.sel(x=self.LCds.x,
                                                   y=self.LCds.y,
                                                   method='nearest')

if __name__=='__main__':
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/als.tif'
  rtmod = RTMOD(rootDir, ALSfile, 'icesat2.atl08.finland', '2m')
  rtmod.modis.to_pickle('/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/modis.2013001.2022365.pkl')
