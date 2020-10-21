#from netCDF4 import Dataset
from osgeo import osr
#from datetime import datetime
import wradlib as wrl
import wradlib.georef as georef
import wradlib.io as io
import wradlib.util as util
#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as pl
import wradlib as wrl
import numpy as np
import os,math
import warnings
import sys
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
#warnings.filterwarnings("ignore", category=RuntimeWarning)

#setting
res=500.
nelevation=6
average_grid=3
a=200;b=1.6
rmax=10000.

def BARON(fpath,res,nelevation,rmax):
	'''
 -------------------------------------------------------------------------------------
   radar_reader.py
   Purpose: These are a set of wrap function to read RADAR
			data from various vendor to make your life easier
			which is outputting the data in cartesian coordinate
			
   Status: Development 
   History:
	  01/2017 - Abdullah Ali - Write radar reader for python2
	  05/2020 - Muhammad Ryan - Make it usable for python3
	  10/2020 - Muhammad Ryan - Multi level CAPPI, edit line 185-189
   Copyright 2020. BMKG. All rights reserved. 
 ------------------------------------------------------------------------------------
  '''
	f = wrl.util.get_wradlib_data_file(fpath)
	raw, metaraw = wrl.io.read_gamic_hdf5(f)
	print('sukses')

	# load site radar lat lon
	llon=float(metaraw['VOL']['Longitude'])
	llat=float(metaraw['VOL']['Latitude'])
	lalt=float(metaraw['VOL']['Height'])
	sitecoords=(llon,llat,lalt)
	
	# define your cartesian reference system
	proj = osr.SpatialReference()
	
	# select spatial reference for Indonesia from EPSG (http://www.spatialreference.org/ref/?search=indonesia)
	if (llat >=0): # north hemisphere
		if (llon < 96):
			epsg_proj = 23846
		elif (llon >= 96 and llon < 102):
			epsg_proj = 23847
		elif (llon >= 102 and llon < 108):
			epsg_proj = 23848
		elif (llon >= 108 and llon < 114):
			epsg_proj = 23849
		elif (llon >= 114 and llon < 120):
			epsg_proj = 23850
		elif (llon >= 120 and llon < 126):
			epsg_proj = 23851
		elif (llon >= 126 and llon < 132):
			epsg_proj = 23852
		else:
			epsg_proj = 4238
	else: # south hemisphere
		if (llon >= 96 and llon < 102):
			epsg_proj = 23887
		elif (llon >= 102 and llon < 108):
			epsg_proj = 23888
		elif (llon >= 108 and llon < 114):
			epsg_proj = 23889
		elif (llon >= 114 and llon < 120):
			epsg_proj = 23890
		elif (llon >= 120 and llon < 126):
			epsg_proj = 23891
		elif (llon >= 126 and llon < 132):
			epsg_proj = 23892
		elif (llon >= 132 and llon < 138):
			epsg_proj = 23893
		elif (llon >= 138):
			epsg_proj = 23894 
		else:
			epsg_proj = 4238
			
	proj.ImportFromEPSG(epsg_proj)	
	
	xyz  = np.array([]).reshape((-1, 3))
	data_cappi_z, data_cappi_v, data_cappi_w = np.array([]), np.array([]), np.array([])
	
	res_coords=res/111229.
	xmax,xmin=llon+(rmax/111229),llon-(rmax/111229)
	ymax,ymin=llat+(rmax/111229),llat-(rmax/111229)
	n_grid=np.floor(((xmax-xmin)/res_coords)+1)
	n_grid = int(n_grid)
	x_grid=np.linspace(xmax,xmin,n_grid)
	y_grid=np.linspace(ymax,ymin,n_grid)			
	all_data = np.zeros((len(x_grid),len(y_grid)))
	all_data_v = np.zeros((len(x_grid),len(y_grid)))
	all_data_w = np.zeros((len(x_grid),len(y_grid)))

	# load elevation
	nelevangle=len(raw)
	if nelevation>nelevangle:
		nelevation=nelevangle
		
	for i in range(nelevation):
		sweep='SCAN'+str(i)
		elevation=float('{0:.1f}'.format(metaraw[sweep]['elevation']))
		strsweep=str(i+1)
		print('Extracting radar data : SWEEP-'+strsweep +' at Elevation Angle '+str(elevation)+'  deg ...')

		azi=metaraw[sweep]['az']
		r=metaraw[sweep]['r']
		
		# data masking and preprocessing
		data=raw[sweep]['Z']['data']
		data_v=raw[sweep]['V']['data']
		data_w=raw[sweep]['W']['data']
		print('CHECK RANGE', r)
		
		clutter=wrl.clutter.filter_gabella(data, tr1=6, n_p=6, tr2=1.3, rm_nans=False)
		data_noclutter=wrl.ipol.interpolate_polar(data, clutter, ipclass = wrl.ipol.Linear)
		data=data_noclutter
		
		# prepare CAPPI
		xyz_ = wrl.vpr.volcoords_from_polar(sitecoords, elevation, azi, r, proj) 
		xyz = np.vstack((xyz, xyz_))
		data_cappi_z = np.append(data_cappi_z,data_noclutter.ravel())
		data_cappi_v = np.append(data_cappi_v,data_v.ravel())
		data_cappi_w = np.append(data_cappi_w,data_w.ravel())
		
		# georeferencing
		polargrid=np.meshgrid(r,azi)
		#x,y,z=wrl.georef.spherical_to_xyz(polargrid[0],polargrid[1],elevation,sitecoords) #(360, 889, 3)
		xyz_pack = wrl.georef.spherical_to_xyz(polargrid[0],polargrid[1],elevation,sitecoords)[0] #(360, 889, 3)
		x = xyz_pack[0, :, :, 0]
		y = xyz_pack[0, :, :, 1]
		z = xyz_pack[0, :, :, 2]

		# regriding
		grid_xy=np.meshgrid(x_grid,y_grid)
		xgrid=grid_xy[0]
		ygrid=grid_xy[1]
		grid_xy = np.vstack((xgrid.ravel(), ygrid.ravel())).transpose()
		xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
		radius=r[np.size(r)-1]
		center=[x.mean(),y.mean()]
		# interpolate polar coordinate data to cartesian cordinate
		# option : Idw, Linear, Nearest
		print(np.shape(xyz_pack), z)
		gridded = wrl.comp.togrid(xy, grid_xy, radius, center, data.ravel(), wrl.ipol.Nearest)	
		gridded_data = np.ma.masked_invalid(gridded).reshape((len(x_grid), len(y_grid)))
		all_data=np.dstack((all_data,gridded_data))
		
		gridded_v = wrl.comp.togrid(xy, grid_xy, radius, center, data_v.ravel(), wrl.ipol.Nearest)	
		gridded_data_v = np.ma.masked_invalid(gridded_v).reshape((len(x_grid), len(y_grid)))
		all_data_v=np.dstack((all_data_v,gridded_data_v))

		gridded_w = wrl.comp.togrid(xy, grid_xy, radius, center, data_w.ravel(), wrl.ipol.Nearest)	
		gridded_data_w = np.ma.masked_invalid(gridded_w).reshape((len(x_grid), len(y_grid)))
		all_data_w = np.dstack((all_data_w,gridded_data_w))
	
	print('Calculate MAX..')
	data=np.nanmax(all_data[:,:,:],axis=2)
	data_v=np.nanmax(all_data_v[:,:,:],axis=2)
	data_w=np.nanmax(all_data_w[:,:,:],axis=2)
	data[data<0]=np.nan;data[data>100]=np.nan
	
	print('Calculate CAPPI..')
	global maxalt, horiz_res, vert_res
	minelev = 0.2
	maxelev = 20.
	maxalt = 1000.
	horiz_res = 500.
	vert_res = 200.
	trgxyz, trgshape = wrl.vpr.make_3d_grid(sitecoords, proj, rmax, maxalt, horiz_res, vert_res)
	
	# interpolate to Cartesian 3-D volume grid
	gridder_cappi = wrl.vpr.CAPPI(xyz, trgxyz, trgshape, rmax, minelev, maxelev, ipclass=wrl.ipol.Nearest)
	cappi_z = np.ma.masked_invalid(gridder_cappi(data_cappi_z).reshape(trgshape))
	#cappi_v = np.ma.masked_invalid(gridder_cappi(data_cappi_v).reshape(trgshape))
	cappi_v = gridder_cappi(data_cappi_v).reshape(trgshape)
	cappi_w = np.ma.masked_invalid(gridder_cappi(data_cappi_w).reshape(trgshape))
	cappi_z[cappi_z<0]=0
	return data,data_v,data_w,cappi_z,cappi_v,cappi_w,x_grid,y_grid

if len(sys.argv) < 2:
	print('please provide file name')
	exit()

nama_file = sys.argv[1]
print(nama_file)

baron_data = BARON(nama_file,res,nelevation,rmax)
print(baron_data[4], np.shape(baron_data[4]), np.nanmean(baron_data[4]))

wind_data = baron_data[4]
x_grid_wind = baron_data[6]
y_grid_wind = baron_data[7]

hshear_data = np.zeros((len(wind_data), len(wind_data[0])-1, len(wind_data[0,0])-1))
for i in range(len(wind_data)):
	for j in range(1, len(wind_data[0])-1):
		for k in range(1, len(wind_data[0,0])-1):
			if np.isnan(wind_data[i, j-1:j+2, k]).any() or np.isnan(wind_data[i, j, k-1:k+2]).any():
				hshear_data[i,j,k] = np.nan
				continue
			#print(wind_data[i, j-1:j+2, k], wind_data[i, j, k-1:k+2])
			hshear_data[i,j,k] = (((wind_data[i, j+1, k]-wind_data[i, j-1, k])/2*res)**2 
			+ ((wind_data[i, j, k+1]-wind_data[i, j, k-1])/2*res)**2)**0.5

for i in range(len(hshear_data)):
	print('total bukan nan lapisan %s'%(i), np.count_nonzero(~np.isnan(hshear_data[i])), np.nansum(hshear_data[i]),
	np.nanmean(hshear_data[i]))
	print(hshear_data[i])

print('ini grid')
print(x_grid_wind)
print(y_grid_wind)

#plotting section
m = Basemap(llcrnrlon=x_grid_wind[-2],llcrnrlat=y_grid_wind[-2],urcrnrlon=x_grid_wind[1],
urcrnrlat=y_grid_wind[1],resolution='h',projection='merc')

m.fillcontinents(color='coral',lake_color='aqua')
# draw lat/lon grid lines every 30 degrees.
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))

for i in range(len(hshear_data)):
	fig, ax = plt.subplots(1)
	cs = m.pcolormesh(x_grid_wind,y_grid_wind,hshear_data[i], latlon=True)
	cbar = m.colorbar(cs, location='bottom', pad="10%")
	plt.title('Radar Horizontal Wind Shear (%s meter above radar)'%(i*200))
	plt.savefig('hshear_%s.png'%(i*200))
	plt.close()


