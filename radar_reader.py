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

   Copyright 2020. BMKG. All rights reserved. 
 ------------------------------------------------------------------------------------
 '''

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
import warnings,os,math
import warnings
#warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
#warnings.filterwarnings("ignore", category=RuntimeWarning)


def EEC(fpath,res,nelevation):
	f=wrl.util.get_wradlib_data_file(fpath)
	raw=wrl.io.read_generic_netcdf(f)
	
	#load site radar lon lat
	llon = float(raw['variables']['longitude']['data'])
	llat = float(raw['variables']['latitude']['data'])
	lalt = float(raw['variables']['altitude']['data'])
	sitecoords = (llon, llat,lalt)
	res_coords=res/111229.
	rmax=250000.
	xmax,xmin=llon+(rmax/111229),llon-(rmax/111229)
	ymax,ymin=llat+(rmax/111229),llat-(rmax/111229)
	n_grid=np.floor(((xmax-xmin)/res_coords)+1)
	n_grid = int(n_grid)
	x_grid=np.linspace(xmax,xmin,n_grid)
	y_grid=np.linspace(ymax,ymin,n_grid)			
	all_data = np.zeros((len(x_grid),len(y_grid)))
	
	#load range data
	r_all = raw['variables']['range']['data'] 
	r=r_all
	
	#load elevation
	nelevangle = np.size(raw['variables']['fixed_angle']['data'])
	sweep_start_idx = raw['variables']['sweep_start_ray_index']['data']
	sweep_end_idx = raw['variables']['sweep_end_ray_index']['data']
	n_azi = np.size(raw['variables']['azimuth']['data'])/nelevangle	
	
	#flag option for loading EEC radar data
	sweep_start_idx = raw['variables']['sweep_start_ray_index']['data']
	sweep_end_idx = raw['variables']['sweep_end_ray_index']['data']
	try:
		if raw['gates_vary']=='true':
			ray_n_gates=raw['variables']['ray_n_gates']['data']
			ray_start_index=raw['variables']['ray_start_index']['data']
			flag='true'
		elif raw['gates_vary']=='false':
			flag='false'
	except :
		if raw['n_gates_vary']=='true':
			ray_n_gates=raw['variables']['ray_n_gates']['data']
			ray_start_index=raw['variables']['ray_start_index']['data']
			flag='true'
		elif raw['n_gates_vary']=='false':
			flag='false'
			
	if nelevation>nelevangle:
		nelevation=nelevangle
	for i in range(nelevation):
		elevation=float('{0:.1f}'.format(raw['variables']['fixed_angle']['data'][i]))		
		strsweep=str(i+1) 
		print('Extracting radar data : SWEEP-'+strsweep +' at Elevation Angle '+str(elevation)+'  deg ...')

		#load azimuth data  
		azi = raw['variables']['azimuth']['data'][sweep_start_idx[i]:sweep_end_idx[i]]   
			
		#load radar data	 
		if flag == 'false':
			data_ = raw['variables']['DBZH']['data'][sweep_start_idx[i]:sweep_end_idx[i], :]
			# create range array
			r = r_all	
		else: #flag = true				
			data_ = np.array([])
			n_azi = sweep_end_idx[i]-sweep_start_idx[i]		
			try:
				for ll in range(sweep_start_idx[i],sweep_end_idx[i]):
					data_ = np.append(data_,raw['variables']['DBZH']['data'][ray_start_index[ll]:ray_start_index[ll+1]])
				data_ = data_.reshape((n_azi,ray_n_gates[sweep_start_idx[i]]))
			except:
				for ll in range(sweep_start_idx[i],sweep_end_idx[i]):
					data_ = np.append(data_,raw['variables']['UH']['data'][ray_start_index[ll]:ray_start_index[ll+1]])
				data_ = data_.reshape((n_azi,ray_n_gates[sweep_start_idx[i]]))
			# create range array
			r = r_all[0:ray_n_gates[sweep_start_idx[i]]]
			
		#remove clutter (using wradlib algorithm)
		data=data_
		clutter=wrl.clutter.filter_gabella(data, tr1=6, n_p=6, tr2=1.3, rm_nans=False)
		data_noclutter=wrl.ipol.interpolate_polar(data, clutter, ipclass = wrl.ipol.Linear)
		data=data_noclutter
		#convert polar to cartesian coordinate
		polargrid=np.meshgrid(r,azi)
		#x,y,z=wrl.georef.polar2lonlatalt_n(polargrid[0],polargrid[1],elevation,sitecoords)
		xyz_pack = wrl.georef.polar.spherical_to_proj(polargrid[0],polargrid[1],elevation,sitecoords)
		x = xyz_pack[:, :, 0]
		y = xyz_pack[:, :, 1]
		z = xyz_pack[:, :, 2]
#		print('shape data ->', np.shape(xyz_pack))
#		print(x_grid)
		#regriding
		grid_xy=np.meshgrid(x_grid,y_grid)
		xgrid=grid_xy[0]
		ygrid=grid_xy[1]
		grid_xy = np.vstack((xgrid.ravel(), ygrid.ravel())).transpose()
		xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
		radius=r[np.size(r)-1]
		center=[x.mean(),y.mean()]
		#interpolate polar coordinate data to cartesian cordinate
		#option : Idw, Linear, Nearest
		gridded = wrl.comp.togrid(xy, grid_xy, radius, center, data.ravel(), wrl.ipol.Linear)	
		gridded_data = np.ma.masked_invalid(gridded).reshape((len(x_grid), len(y_grid)))
		all_data=np.dstack((all_data,gridded_data))
		
	data=np.nanmax(all_data[:,:,:],axis=2)
	data[data<0]=np.nan;data[data>100]=np.nan
	
	x,y=xgrid,ygrid
	xymaxmin=(xmax,xmin,ymax,ymin)
	print('Finish extracting data.')
	return data,sitecoords,x,y,xymaxmin

def GEMA(listpcdfpath,radarpath,res,rmax,sweep):
	f0=wrl.util.get_wradlib_data_file(listpcdfpath[0])
	raw=wrl.io.read_Rainbow(f0)
	try :	
		llon = float(raw['volume']['sensorinfo']['lon'])
		llat = float(raw['volume']['sensorinfo']['lat'])
		lalt = float(raw['volume']['sensorinfo']['alt'])
	except :
		llon = float(raw['volume']['radarinfo']['@lon'])
		llat = float(raw['volume']['radarinfo']['@lat'])
		lalt = float(raw['volume']['radarinfo']['@alt'])
	res_coords=res/111229.
	xmax,xmin=llon+(rmax/111229),llon-(rmax/111229)
	ymax,ymin=llat+(rmax/111229),llat-(rmax/111229)
	n_grid=np.floor(((xmax-xmin)/res_coords)+1)
	x_grid=np.linspace(xmax,xmin,n_grid)
	y_grid=np.linspace(ymax,ymin,n_grid)			

	for fpath in listpcdfpath:
		filename=fpath[len(radarpath)+1:]
		f=wrl.util.get_wradlib_data_file(fpath)
		raw=wrl.io.read_Rainbow(f)
		
		# load site radar attribute
		try :	
			llon = float(raw['volume']['sensorinfo']['lon'])
			llat = float(raw['volume']['sensorinfo']['lat'])
			lalt = float(raw['volume']['sensorinfo']['alt'])
		except :
			llon = float(raw['volume']['radarinfo']['@lon'])
			llat = float(raw['volume']['radarinfo']['@lat'])
			lalt = float(raw['volume']['radarinfo']['@alt'])
		sitecoords=(llon,llat,lalt)

		i=sweep
		try :	
			elevation = float(raw['volume']['scan']['slice'][i]['posangle'])
		except :
			elevation = float(raw['volume']['scan']['slice'][0]['posangle'])
		strsweep=str(i+1) 

		# load azimuth data
		try:
			azi = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['data']
			azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@depth'])
			azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@rays'])  
		except:
			azi0 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['data']
			azi1 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][1]['data']
			azi = (azi0/2) + (azi1/2)
			del azi0, azi1
			azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@depth'])
			azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@rays'])			
		try:
			azires = float(raw['volume']['scan']['slice'][i]['anglestep'])
		except:
			azires = float(raw['volume']['scan']['slice'][0]['anglestep'])
		azi = (azi * azirange / 2**azidepth) * azires

		# load range data
		try:
			stoprange = float(raw['volume']['scan']['slice'][i]['stoprange'])
			rangestep = float(raw['volume']['scan']['slice'][i]['rangestep'])
		except:
			stoprange = float(raw['volume']['scan']['slice'][0]['stoprange'])
			rangestep = float(raw['volume']['scan']['slice'][0]['rangestep'])
		r = np.arange(0, stoprange, rangestep)*1000
			
		# projection for lat and lon value
		polargrid=np.meshgrid(r,azi)
		#x,y,z=wrl.georef.polar2lonlatalt_n(polargrid[0],polargrid[1],elevation,sitecoords)
		xyz_pack = wrl.georef.polar.spherical_to_proj(polargrid[0],polargrid[1],elevation,sitecoords)
		x = xyz_pack[:, :, 0]
		y = xyz_pack[:, :, 1]
		z = xyz_pack[:, :, 2]

		# regriding
		grid_xy=np.meshgrid(x_grid,y_grid)
		xgrid=grid_xy[0]
		ygrid=grid_xy[1]
		grid_xy = np.vstack((xgrid.ravel(), ygrid.ravel())).transpose()
		xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
		radius=r[np.size(r)-1]
		center=[x.mean(),y.mean()]
		
		# load radar data (depend on what file you load)
		if fpath[-7:]=='dBZ.vol':
			print 'Extracting data '+filename+' : SWEEP-'+strsweep +' at Elevation Angle '+str(elevation)+'  deg ...'
			data = raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['data']
			datadepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@depth'])
			datamin = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@min'])
			datamax = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@max'])
			data = datamin + data * (datamax - datamin) / 2 ** datadepth
			# data masking and preprocessing
			clutter=wrl.clutter.filter_gabella(data, tr1=6, n_p=6, tr2=1.3, rm_nans=False)
			try:
				data_noclutter=wrl.ipol.interpolate_polar(data, clutter, ipclass = wrl.ipol.Linear)
			except:
				data_noclutter=data
			data_dbz=data_noclutter
			# if the shape of data and georeferencing is not the same	
			a,b=np.shape(x)
			c,d=np.shape(data_dbz)
			selisih1=a-c
			selisih2=b-d
			if selisih2>0:
				print ('Matching data and coordinate shape...')
				data_=np.zeros((a,b))
				for k in range(c):
					for j in range(d):
						data_[k,j]=data_dbz[k,j]
					for ii in range(selisih2):
						data_[c-1,d+ii]=np.nan
					data_dbz=data_
			if selisih1>0:
				print ('Matching data and coordinate shape...')
				for ii in range(selisih1):
					data_[c+ii,d-1]=np.nan
				data_dbz=data_
			# interpolate polar coordinate data to cartesian cordinate
			# option : Idw, Linear, Nearest
			gridded_dbz = wrl.comp.togrid(xy, grid_xy, radius, center, data_dbz.ravel(), wrl.ipol.Linear)	
			gridded_datadbz = np.ma.masked_invalid(gridded_dbz).reshape((len(x_grid), len(y_grid)))
							
		elif fpath[-5:]=='V.vol':
			print 'Extracting data '+filename+'   : SWEEP-'+strsweep +' at Elevation Angle '+str(elevation)+'  deg ...'
			data = raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['data']
			datadepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@depth'])
			datamin = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@min'])
			datamax = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@max'])
			data = datamin + data * (datamax - datamin) / 2 ** datadepth
			data_v = data
			# interpolate polar coordinate data to cartesian cordinate
			# option : Idw, Linear, Nearest
			gridded_v = wrl.comp.togrid(xy, grid_xy, radius, center, data_v.ravel(), wrl.ipol.Linear)	
			gridded_datav = np.ma.masked_invalid(gridded_v).reshape((len(x_grid), len(y_grid)))
				
	return gridded_datadbz,gridded_datav,sitecoords,datetime,elevation,xgrid,ygrid,xmin,xmax,ymin,ymax,z

def BARON(fpath,res,nelevation,rmax):
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
		
		
		clutter=wrl.clutter.filter_gabella(data, tr1=6, n_p=6, tr2=1.3, rm_nans=False)
		data_noclutter=wrl.ipol.interpolate_polar(data, clutter, ipclass = wrl.ipol.Linear)
		data=data_noclutter
		
		# perpare CAPPI
		xyz_ = wrl.vpr.volcoords_from_polar(sitecoords, elevation, azi, r, proj) 
		xyz = np.vstack((xyz, xyz_))
		data_cappi_z = np.append(data_cappi_z,data_noclutter.ravel())
		data_cappi_v = np.append(data_cappi_v,data_v.ravel())
		data_cappi_w = np.append(data_cappi_w,data_w.ravel())
		
		# georeferencing
		polargrid=np.meshgrid(r,azi)
		#x,y,z=wrl.georef.spherical_to_xyz(polargrid[0],polargrid[1],elevation,sitecoords) #(360, 889, 3)
		xyz_pack = wrl.georef.spherical_to_xyz(polargrid[0],polargrid[1],elevation,sitecoords) #(360, 889, 3)
		x = xyz_pack[:, :, 0]
		y = xyz_pack[:, :, 1]
		z = xyz_pack[:, :, 2]

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
	vert_res = 1000.
	trgxyz, trgshape = wrl.vpr.make_3D_grid(sitecoords, proj, rmax, maxalt, horiz_res, vert_res)
	
	# interpolate to Cartesian 3-D volume grid
	gridder_cappi = wrl.vpr.CAPPI(xyz, trgxyz, trgshape, rmax, minelev, maxelev, Ipclass=wrl.ipol.Nearest)
	cappi_z = np.ma.masked_invalid(gridder_cappi(data_cappi_z).reshape(trgshape))
	cappi_v = np.ma.masked_invalid(gridder_cappi(data_cappi_v).reshape(trgshape))
	cappi_w = np.ma.masked_invalid(gridder_cappi(data_cappi_w).reshape(trgshape))
	cappi_z[cappi_z<0]=0
	return data,data_v,data_w,cappi_z,cappi_v,cappi_w,x_grid,y_grid
