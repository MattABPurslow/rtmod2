import glob, os, sys, pdb
import subprocess
sys.path.append('/home/s1503751/src/')
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

from multiprocessing import Pool
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
    self.k = 0.5 # As in JULES
    self.crownRatio = 12.9/2.7 # for boreal & montane (Jucker et al 2022)
    self.wv = ['vis', 'nir', 'shortwave', 'vis', 'nir', 'shortwave']
    self.df = ['BSA', 'BSA', 'BSA', 'WSA', 'WSA', 'WSA']
    self.createSellers()
    self.sellers = np.vectorize(self.sellersUfunc)

  def main(self):
    """
    Run Sellers and SPARTACUS
    """
    self.buildDS()
    self.createSPARTACUSinput()
    if self.ds.column.shape[0]>0:
      self.getSellers()
      self.runSPARTACUS()

  def getSellers(self):
    """
    Run Sellers for all pixels in out dataset
    """
    #proportion of diffuse radiation
    for alb, wv in zip(self.df, self.wv):
      k = 'Sellers_%s_%s'%(alb,wv)
      if alb=='BSA':
        self.t.propDif = 0.
        self.ds[k] = xr.apply_ufunc(self.sellers,
                                    self.ds.cos_solar_zenith_angle,
                                    self.ds.MODIS_LAI_eff,
                                    self.ds[f'leaf_r_{wv}'],
                                    self.ds[f'leaf_t_{wv}'],
                                    self.ds[f'ground_albedo_direct_{wv}'],
                                    wv)
      else:
          self.t.propDif = 1.
          self.ds[k] = xr.apply_ufunc(self.sellers,
                                      self.ds.cos_solar_zenith_angle,
                                      self.ds.MODIS_LAI_eff,
                                      self.ds[f'leaf_r_{wv}'],
                                      self.ds[f'leaf_t_{wv}'],
                                      self.ds[f'ground_albedo_diffuse_{wv}'],
                                      wv)
  
  def createSellers(self):
    """
    Create single layer Sellers instance
    """
    self.t = twoStream()
    self.t.setupJULES()
    #single layer
    self.t.nLayers=1

  def sellersUfunc(self, mu, LAI, leaf_r, leaf_t, SA, wv):
    """
    Run Sellers for given inputs
    mu: cosine of solar zenith angle
    LAI: effective LAI
    SA: subcanopy albedo
    wv: waveband
    """
    check = np.isnan(mu)|np.isnan(LAI)|(LAI<0.01)|np.isnan(leaf_r)|np.isnan(leaf_t)|np.isnan(SA)|(SA==0)
    if check:
      return np.nan
    else:
      #cosine of solar zenith:
      self.t.mu=mu
      #leaf area index
      self.t.lai=LAI
      #leaf reflectance & tranmisttance
      self.t.leaf_r = leaf_r
      self.t.leaf_t = leaf_t
      #soil reflectance
      self.t.soil_r=SA
      #do the radiative transfer calculation:
      Iup, _, _, _ = self.t.getFluxes()
      return Iup[0]
  
  def eigsorted(self, cov):
      vals, vecs = np.linalg.eigh(cov)
      order = vals.argsort()[::-1]
      return vals[order], vecs[:,order]

  def getAlbDict(self, priors, xlab, ylab, C, visFrac, Nσ=1.5):
    cov = [[priors.loc[xlab, 'σ_x']**2,
            C*priors.loc[xlab, 'σ_x']*priors.loc[ylab, 'σ_x']],
           [C*priors.loc[xlab, 'σ_x']*priors.loc[ylab, 'σ_x'],
            priors.loc[ylab, 'σ_x']**2]]
    vals, vecs = self.eigsorted(cov)
    theta = np.rad2deg(np.arctan2(*vecs[:,0][::-1]))
    a, b = Nσ*np.sqrt(vals)
    x0 = priors.loc[xlab, 'x_mean']
    y0 = priors.loc[ylab, 'x_mean']
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=(x0, y0),
                  width=2*a, height=2*b,
                  angle=theta, facecolor='none', edgecolor='k')
    tmpfig, tmpax = plt.subplots(1,1)
    tmpax.add_artist(ell)
    x = np.random.rand(1)[0]
    y = np.random.rand(1)[0]
    while ell.contains_point(tmpax.transData.transform((x,y)))==False:
      x = np.random.rand(1)[0]
      y = np.random.rand(1)[0]
    albDict = {'diffuse': {'vis':x, 'nir':y}}
    tmpax.scatter(x,y)
    x = np.random.rand(1)[0]
    y = np.random.rand(1)[0]
    while ell.contains_point(tmpax.transData.transform((x,y)))==False:
      x = np.random.rand(1)[0]
      y = np.random.rand(1)[0]
    albDict['direct'] = {'vis':x, 'nir':y}
    albDict['diffuse']['shortwave'] = albDict['diffuse']['vis']*visFrac+\
                                      albDict['diffuse']['nir']*(1-visFrac)
    albDict['direct']['shortwave'] = albDict['direct']['vis']*visFrac+\
                                     albDict['direct']['nir']*(1-visFrac)
    plt.close()
    return albDict

  def rand_Nσ(self, x_mean, σ_x, N, Nσ=1.5):
    """
    val = [-1]
    while (np.min(val) < 0):
      val = x_mean + ((np.random.rand(N)-0.5)*2.)*Nσ*σ_x
    if N==1:
      return float(val)
    else:
      return val.astype(float)
    """
    val = [-1]
    while (np.min(val) < 0):
      val = stats.norm.rvs(loc=x_mean,
                           scale=σ_x,
                           size=N)
    if N==1:
      return float(val)
    else:
      return val.astype(float)

  def buildDS(self):
    priors = pd.DataFrame({'x_mean': [1.5,
                                      0.17, 0.13,
                                      1.0,
                                      0.1, 0.5,
                                      0.7, 0.77,
                                      2.0,
                                      0.18, 0.35],
                           'σ_x':  [5.0,
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
    
    visFrac = (0.7-0.3)/(3-0.3)
    self.albDict = self.getAlbDict(priors, 'alb_vis', 'alb_nir', 0.8862, visFrac)
    self.snowDict = self.getAlbDict(priors, 'alb_vis_snow', 'alb_nir_snow', 0.8670, visFrac)
    self.rtDict = {'ssa': {'vis': self.rand_Nσ(priors.loc['SSA_vis','x_mean'],
                                          priors.loc['SSA_vis','σ_x'], 1),
                           'nir': self.rand_Nσ(priors.loc['SSA_nir','x_mean'],
                                          priors.loc['SSA_nir','σ_x'], 1)},
                   'ratio': {'vis': self.rand_Nσ(priors.loc['asymmetry_vis','x_mean'],
                                            priors.loc['asymmetry_vis','σ_x'], 1),
                             'nir': self.rand_Nσ(priors.loc['asymmetry_nir','x_mean'],
                                            priors.loc['asymmetry_nir','σ_x'], 1)}}
    self.rtDict['r'] = {c: self.rtDict['ssa'][c] / (1+(1/self.rtDict['ratio'][c]))
                        for c in ['vis', 'nir']}
    self.rtDict['t'] = {c: self.rtDict['ssa'][c] - self.rtDict['r'][c]
                        for c in ['vis', 'nir']}
    self.rtDict['r']['shortwave'] =  self.rtDict['r']['vis']*visFrac+\
                                     self.rtDict['r']['nir']*(1-visFrac)
    self.rtDict['t']['shortwave'] =  self.rtDict['t']['vis']*visFrac+\
                                     self.rtDict['t']['nir']*(1-visFrac)
    self.ds = xr.Dataset(coords={'PFT_LC': [1],
                                 'omega': np.sort(np.random.rand(5)),
                                 'chm': np.sort(np.random.rand(5))*40,
                                 'cv': np.sort(np.random.rand(5))*100,
                                 'MODIS_LAI': self.rand_Nσ(priors.loc['LAI', 'x_mean'],
                                                      priors.loc['LAI', 'σ_x'],
                                                      5),
                                 'MODIS_LAI_Sigma': np.sort(np.random.rand(5)),
                                 'MODIS_Snow_Albedo': [-1],
                                 'MODIS_NDSI_Snow_Cover': [-1],
                                 'crownRatio': self.crownRatio + ((np.sort(np.random.rand(7)-0.5))*(self.crownRatio/2)),
                                 'zen': np.sort(np.random.rand(45))*90})
    self.ds['MODIS_LAI_eff'] = self.ds.MODIS_LAI * self.ds.omega
    self.ds['cos_solar_zenith_angle'] = np.cos(np.deg2rad(self.ds.zen))
    self.ds = self.ds.to_dataframe()
    for wv in ['vis', 'nir', 'shortwave']:
      self.ds[f'leaf_r_{wv}'] = self.rtDict['r'][wv]
      self.ds[f'leaf_t_{wv}'] = self.rtDict['t'][wv]
      self.ds[f'alb_diffuse_{wv}'] = self.snowDict['diffuse'][wv]
      self.ds[f'alb_direct_{wv}'] = self.snowDict['direct'][wv]
    self.ds = self.ds.sample(frac=0.0001)
    self.ds = self.ds.reset_index()
    self.ds['column'] = self.ds.index.values
    self.ds = self.ds.set_index(['column']).to_xarray()
  
  def getExtinction(self):
    """
    Calculate vegetation extinction coefficient
    """
    self.ds['omega'] = self.ds.omega.where((self.ds.omega > 0) & (self.ds.omega <= 1))
    self.ds['veg_extinction'] = (('layer', 'column'), [(self.k*self.ds.MODIS_LAI)/self.ds.chm])

  def getFSD(self):
    """
    Define variability in LAI
    """
    self.ds['veg_fsd'] = (('layer', 'column'), [self.ds.MODIS_LAI_Sigma.where(self.ds.MODIS_LAI_Sigma>0)])

  def getSA(self, Nsw):
    """
    Retrieve subcanopy albedo
    """
    ground_sw_albedo = []
    ground_sw_albedo_direct = []
    for wv in ['vis', 'nir', 'shortwave']:
      self.ds[f'ground_albedo_diffuse_{wv}'] = self.ds[f'alb_diffuse_{wv}']
      self.ds[f'ground_albedo_direct_{wv}'] = self.ds[f'alb_direct_{wv}']
    self.ds['ground_sw_albedo'] = (('sw', 'column'), [self.ds[f'ground_albedo_diffuse_{wv}'].values for wv in self.wv])
    self.ds['ground_sw_albedo_direct'] = (('sw', 'column'), [self.ds[f'ground_albedo_direct_{wv}'].values for wv in self.wv])
  
  def getSSA(self):
    """
    Get vegetation single scattering albedo
    """
    ssa_vis = (self.ds.leaf_r_vis + self.ds.leaf_t_vis).values
    ssa_nir = (self.ds.leaf_r_nir + self.ds.leaf_t_nir).values
    ssa_short = (self.ds.leaf_r_shortwave + self.ds.leaf_t_shortwave).values
    self.ds['veg_sw_ssa'] = (('layer', 'sw', 'column'), [[ssa_vis, ssa_nir, ssa_short, ssa_vis, ssa_nir, ssa_short]])

  def createSPARTACUSinput(self):
    """
    Create dataset in format required for SPARTACUS runs
    """
    self.ds['height'] = (('layer_int','column'), [np.full(self.ds.chm.shape,0.),
                                                 self.ds.chm])
    self.ds['veg_fraction'] = (('layer', 'column'), [self.ds.cv])
    self.ds['veg_fraction'] = self.ds.veg_fraction.where(self.ds.veg_fraction>0.01)
    self.ds['veg_scale'] = (('layer', 'column'), [self.ds.chm / self.ds.crownRatio])
    ## Default values
    Nc = self.ds.column.shape[0]
    Nsw = 6
    self.ds['surface_type'] = (('column'), np.full((Nc), int(1)))
    self.ds['nlayer'] = (('column'), np.full((Nc), int(1)))
    self.ds['veg_contact_fraction'] = (('column', 'layer'), np.full((Nc, 1), 0.))
    self.ds['building_fraction']  = (('column', 'layer'), np.full((Nc, 1), 0.))
    self.ds['building_scale'] = (('column', 'layer'), np.full((Nc, 1), 0.))
    self.ds['clear_air_temperature'] = (('column', 'layer'), np.full((Nc, 1), 273.))
    self.ds['veg_temperature'] = (('column', 'layer'), np.full((Nc, 1), 273.))
    self.ds['veg_air_temperature'] = (('column', 'layer'), np.full((Nc, 1), 273.))
    self.ds['air_temperature'] = (('column', 'layer'), np.full((Nc, 1), 273.))
    self.ds['ground_temperature'] = (('column'), np.full((Nc), 273.))
    self.ds['roof_temperature'] = (('column', 'layer'), np.full((Nc, 1), 273.))
    self.ds['wall_temperature'] = (('column', 'layer'), np.full((Nc, 1), 273.))
    self.ds['ground_lw_emissivity'] = (('column', 'lw'), np.full((Nc, 1), 1.))
    self.ds['veg_lw_ssa'] = (('column', 'layer', 'lw'), np.full((Nc, 1, 1), 1.))
    self.ds['roof_sw_albedo'] = (('column', 'sw'), np.full((Nc, Nsw), 1.))
    self.ds['roof_sw_albedo_direct'] = (('column', 'sw'), np.full((Nc, Nsw), 1.))
    self.ds['roof_lw_emissivity'] = (('column', 'lw'), np.full((Nc, 1), 1.))
    self.ds['wall_sw_albedo'] = (('column', 'sw'), np.full((Nc, Nsw), 1.))
    self.ds['wall_sw_albedo_direct'] = (('column', 'sw'), np.full((Nc, Nsw), 1.))
    self.ds['wall_lw_emissivity'] = (('column', 'lw'), np.full((Nc, 1), 1.))
    self.ds['sky_temperature'] = (('column'), np.full((Nc), 273.))
    self.ds['top_flux_dn_sw'] = (('column', 'sw'), np.full((Nc, Nsw), 1.))
    self.ds['top_flux_dn_direct_sw'] = (('column', 'sw'), np.full((Nc, Nsw), ([1.]*(Nsw//2))+([0.]*(Nsw//2))))
    self.getExtinction()
    self.getFSD()
    self.getSA(Nsw)
    self.getSSA()
    self.ds = self.ds.copy()
    varList = list(self.ds.variables.keys())
    for v in varList:
      Nd = len(self.ds[v].dims)
      if Nd > 1:
        dims = sorted(self.ds[v].dims)
        if Nd == 2:
          self.ds[v] = self.ds[v].transpose(dims[0], dims[1])
        elif Nd == 3:
          self.ds[v] = self.ds[v].transpose(dims[0], dims[1], dims[2])
        else:
          raise Exception('Too many dims on %s ' % v)
      else:
        pass
    self.spartVars = ['surface_type',
                      'nlayer',
                      'height',
                      'veg_fraction',
                      'veg_scale',
                      'veg_extinction',
                      'veg_fsd',
                      'veg_contact_fraction',
                      'building_fraction',
                      'building_scale',
                      'clear_air_temperature',
                      'veg_temperature',
                      'veg_air_temperature',
                      'air_temperature',
                      'ground_temperature',
                      'roof_temperature',
                      'wall_temperature',
                      'ground_sw_albedo',
                      'ground_sw_albedo_direct',
                      'ground_lw_emissivity',
                      'veg_sw_ssa',
                      'veg_lw_ssa',
                      'roof_sw_albedo',
                      'roof_sw_albedo_direct',
                      'roof_lw_emissivity',
                      'wall_sw_albedo',
                      'wall_sw_albedo_direct',
                      'wall_lw_emissivity',
                      'sky_temperature',
                      'top_flux_dn_sw',
                      'top_flux_dn_direct_sw',
                      'column']
    self.ds = self.ds.dropna(dim='column')
    self.ds['veg_fraction'] = self.ds.veg_fraction / 100.
    self.ds = self.ds.transpose('column', 'layer', 'layer_int', 'lw', 'sw')

  def runSPARTACUS(self):
    """
    Save SPARTACUS input files and run
    """
    ## Remove any pixels with persistent missing data 
    if self.ds.column.shape[0]>0:
      for i in range(self.ds.sw.shape[0]):
        inFile = os.path.join(self.rootDir, f'spartacusInSensitivity_{self.index}/sodankyla.%s_%s.nc' % (self.df[i], self.wv[i]))
        if os.path.exists(os.path.split(inFile)[0])==False:
          os.mkdir(os.path.split(inFile)[0])
        self.ds.isel(sw=[i]).to_netcdf(inFile, engine='scipy', format='NETCDF3_CLASSIC')
        outFile = inFile.replace('spartacusInSensitivity', 'spartacusOutSensitivity')
        if os.path.exists(os.path.split(outFile)[0])==False:
          os.mkdir(os.path.split(outFile)[0])
        command = ['spartacus_surface', 'config.nam', inFile, outFile]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def readSPARTACUS(outFile):
  ds = xr.open_dataset(outFile.replace('spartacusOutSensitivity', 'spartacusInSensitivity')).isel(layer=-1, layer_int=-1)
  labLoaded = []
  for df in ['BSA', 'WSA']:
    for wv in ['nir', 'vis', 'shortwave']:
      lab = '%s_%s' % (df, wv)
      if (os.path.exists(outFile.replace('BSA_nir', lab))):
        out = xr.open_dataset(outFile.replace('BSA_nir', lab)).isel(layer=-1, layer_interface=-1)
        ds['SPARTACUS_%s' % lab] = out.flux_up_layer_top_sw
        labLoaded.append(lab)
  return ds.to_dataframe().reset_index()

def run(i):
  rootDir = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections'
  ALSfile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla/ALS/als.tif'
  lidar = 'icesat2.atl08'
  threshold = '2m'
  rtmod = RTMOD(rootDir, ALSfile, lidar, threshold)
  rtmod.index = i
  np.random.RandomState(i)
  rtmod.main()
  inDir = os.path.join(rootDir, f'spartacusInSensitivity_{rtmod.index}')
  outDir = os.path.join(rootDir, f'spartacusOutSensitivity_{rtmod.index}')
  outList = sorted(glob.glob(os.path.join(outDir, '*BSA_nir.nc')))
  df = pd.DataFrame()
  cnt = 1
  for outFile in outList:
    df = pd.concat([df, readSPARTACUS(outFile)], ignore_index=True)
  os.system(f'rm -r {inDir} {outDir}')
  columns = ['leaf_r_vis', 'leaf_r_nir', 'leaf_r_shortwave',
             'leaf_t_vis', 'leaf_t_nir', 'leaf_t_shortwave',
             'alb_diffuse_vis', 'alb_diffuse_nir', 'alb_diffuse_shortwave',
             'alb_direct_vis', 'alb_direct_nir', 'alb_direct_shortwave',
             'omega', 'chm', 'cv', 'zen', 'veg_scale', 'veg_fsd',
             'MODIS_LAI', 'MODIS_LAI_Sigma', 'MODIS_LAI_eff',
             'Sellers_BSA_vis', 'Sellers_BSA_nir', 'Sellers_BSA_shortwave',
             'Sellers_WSA_vis', 'Sellers_WSA_nir', 'Sellers_WSA_shortwave',
             'SPARTACUS_BSA_nir', 'SPARTACUS_BSA_vis', 'SPARTACUS_BSA_shortwave',
             'SPARTACUS_WSA_nir', 'SPARTACUS_WSA_vis', 'SPARTACUS_WSA_shortwave']
  outFile = '/exports/csce/datastore/geos/users/s1503751/MODIS/sodankyla_corrections/model.pinty.paramspace.csv'
  if df.columns.all() in columns:
    if os.path.exists(outFile)==False:
      df[columns].to_csv(outFile, index=False, float_format='%.3f')
    else:
      df[columns].to_csv(outFile, index=False, mode='a', header=False, float_format='%.3f')

if __name__=='__main__':
  import tqdm
  Niter = 50000
  with Pool(12) as pool:
    for _ in  tqdm.tqdm(pool.imap_unordered(run, np.random.randint(0, (2**32)-1, Niter)), total=Niter):
      pass
