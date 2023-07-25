
"""
Decompose NWP-forecast (ALARO/AROME) in cascade and calculate UV-field
for use in operational pySTEPS
Author: Michiel Van Ginderachter 02/05/2022
michiel.vanginderachter@meteo.be
"""
import argparse
import os
import sys
# Parse arguments before loading heavy pysteps module 
parser = argparse.ArgumentParser(
    description='Decompose NWP-forecast (ALARO/AROME) in cascade and calculate UV-field ' +
    'for use in operational pySTEPS.'
    )

parser.add_argument('filenames',
    metavar='FNAME',
    type=str,
    help='file(s) that need(s) to be decomposed.',
    nargs = '+'
    )

parser.add_argument('-n',
    dest='nworkers',
    type=int,
    default=1,
    help='number of dask workers used to decompose the fields and calculate the UV-field (default = 1).'
    )

parser.add_argument('-c',
    dest='ncascades',
    type=int,
    default=8,
    help='number of cascade levels (default = 8).'
    )

parser.add_argument('-t',
    dest='threshold',
    type=float,
    default=0.1,
    help='threshold (in mm/h) below which the precipitation is set to 0 (default = 0.1)'
    )

parser.add_argument('--dir-cascade',
    dest='dir_cascade',
    default='/data',
    help='directory where to store the decomposed cascade (default = os.getcwd())'
    )

parser.add_argument('--dir-motion',
    dest='dir_motion',
    default='/data',
    help='directory where to store the UV-field (default = os.getcwd())'
    )

args = parser.parse_args()

# Only continue when arguments are parsed correctly
import logging
import datetime
import netCDF4
import numpy as np
import pprint
import sys
import time
import scipy
from pathlib import Path
import dask
import pysteps
import xarray as xr

# - QPE radar product metadata
metadata_radar = {
    'projection' : '+proj=lcc '\
        '+lat_1=49.83333333333334 +lat_2=51.16666666666666 +lat_0=50.797815 +lon_0=4.359215833333333 '\
        '+x_0=649328 +y_0=665262 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ',
    'll_lon' : -0.2666973996088157,
    'll_lat' : 47.41679117656605,
    'ur_lon' : 9.664159875778674,
    'ur_lat' : 53.69199685747096,
    'x1' : 300000.00000639487,
    'y1' : 300000.00010250637,
    'x2': 999999.9999935811,
    'y2': 1000000.0000994017,
    'xpixelsize': 1000.0,
    'ypixelsize': 1000.0,
    'cartesian_unit': 'm',
    'yorigin': 'upper',
    'institution': 'Odyssey datacentre',
    'accutime': 5.0,
    'unit': 'mm/h',
    'transform': None,
    'zerovalue': 0.0,
    'threshold': 6.064287561002857e-08
    }
r_radar = np.zeros((700, 700))

if __name__ == '__main__':

    # Set logger options
    logging.basicConfig(format='[%(levelname)s]: %(asctime)s - %(message)s', level=logging.INFO)

    # Set general options
    fnames = args.filenames
    ncores = args.nworkers
    ncascades = args.ncascades
    threshold = args.threshold
    dir_cascade = args.dir_cascade
    dir_motion = args.dir_motion
    
    start_time_script = time.time()
    # Make sure the output directories exist
    os.makedirs(dir_cascade,exist_ok=True)
    os.makedirs(dir_motion,exist_ok=True)
    erno = 0
    # Loop over the files 
    for fname in fnames:
        if not os.path.exists(fname):
            logging.error("file %s does not exist, no output has been generated. Moving on the the next file", fname)
            erno = erno+1
            erno = erno/erno
        else:
            logging.info("Started decomposition with:")
            logging.info(r'    File: %s' % fname)
            logging.info(r'    Number of workers: %i' % ncores)
            logging.info(r'    Rain/No-rain threshold: %.2f' % threshold)
            logging.info(r'    Number of cascade levels: %i' % ncascades)
            logging.info(r'    Motion vectors are stored in: %s' % dir_motion)
            logging.info(r'    Cascade decompositions are stored in: %s' % dir_cascade)
        
            # Set the data source
            data_src_nwp = "rmi_nwp"
            
            # Importer the NPW data
            importer_nwp = pysteps.io.get_method(data_src_nwp, "importer")
            r_nwp, _, metadata_nwp = importer_nwp(fname)
            validtime_nwp = metadata_nwp['time_stamps']

            # For older NWP files, the proj4 string starts with '++...' instead of '+...'
            # correct for this:
            if metadata_nwp['projection'][:2] == '++':
                metadata_nwp['projection'] = metadata_nwp['projection'][1:]


            # Reproject NWP forecast to RADAR domain and transform to mm/h
            # - reproject the NWP data to the radar grid
            #!TODO: Can I speed this up with dask.delayed?
            r_nwp, metadata_nwp = pysteps.utils.reprojection.reproject_grids(
                    src_array = r_nwp,
                    dst_array = r_radar,
                    metadata_src = metadata_nwp,
                    metadata_dst = metadata_radar
            )

            # - convert NWP data to rainrate
            converter = pysteps.utils.get_method("mm/h")
            r_nwp, metadata_nwp = converter(r_nwp, metadata_nwp)
            
            # - set threshold
            r_nwp[r_nwp < threshold] = 0.0
            metadata_nwp["threshold"] = threshold 
            
            # Compute and store the motion
            logging.info('Computing and storing the motion vectors...')
            
            # - transform NWP data to dB-scale
            transformer = pysteps.utils.get_method("dB")
            r_nwp, metadata_nwp = transformer(r_nwp, metadata_nwp, threshold=metadata_nwp["threshold"])
            
            # - initialize the optical flow method
            oflow_method = pysteps.motion.get_method("LK")
            
            # - define the dask worker
            def worker(j):
                V_ = oflow_method(r_nwp[j-1:j+1,:,:])
                return V_
            
            # - distribute the tasks
            res = []
            for i in range(1,r_nwp.shape[0]):
                res.append(dask.delayed(worker)(i))
            
            # - compute the tasks and gather the results
            num_workers_ = len(res) if ncores > len(res) else ncores
            V_ = dask.compute(*res,num_workers=num_workers_)
            res = None
            
            v_nwp = np.zeros((r_nwp.shape[0], 2, r_nwp.shape[1], r_nwp.shape[2]))
            for i in range(1,r_nwp.shape[0]):
                v_nwp[i,:,:,:] = V_[i-1]
            v_nwp[0]=v_nwp[1]
    
            # - save the result in pickle file  
            model = os.path.basename(fname)[0:4]    
            output_date = f"{validtime_nwp[0].astype('datetime64[us]').astype(datetime.datetime):%Y%m%d%H%M%S}"
            outfn = Path(dir_motion) / f"motion_{model}_{output_date}.npy"
            np.save(outfn, v_nwp)
            
            logging.info('done!')
            
            # Decompose the NWP forecast
            logging.info('Decomposing the NWP forecast...')
            
            # - initialize the filtering method, fft method and decomposition method
            filter_cascade = pysteps.cascade.bandpass_filters.filter_gaussian(r_nwp.shape[1:],ncascades)
            fft = pysteps.utils.get_method('scipy', shape=r_nwp.shape[1:], n_threads=1)
            decomp_method, _ = pysteps.cascade.get_method('fft')
            
            # - define the worker used by dask.delayed
            # - a worker performs a cascade decomposisition for one timestep
            def worker(j):
                R_ = decomp_method(
                    field=r_nwp[j,:,:],
                    bp_filter=filter_cascade,
                    fft_method=fft,
                    input_domain='spatial',
                    output_domain='spatial',
                    normalize=True,
                    compute_stats=True,
                    compact_output=False
                )
                return R_
            
            # - distribute the tasks
            res=[]
            for i in range(r_nwp.shape[0]):
                res.append(dask.delayed(worker)(i))
            
            # - compute the tasks and gather the results
            num_workers_ = len(res) if ncores > len(res) else ncores
            R_ = dask.compute(*res,num_workers=num_workers_)
            res = None
            
            logging.info('done!')
            
            # Build the xarray for faster writing to netCDF
            logging.info('Building xarray and saving to netCDF...')
            
            # - initialize the final arrays
            R_d = np.ones((r_nwp.shape[0],ncascades,r_nwp.shape[1],r_nwp.shape[2]))
            means = np.ones((r_nwp.shape[0],ncascades))
            stds = np.ones((r_nwp.shape[0],ncascades))
            v_times = np.ones((r_nwp.shape[0],))
            
            # - reformat the results
            for i in range(r_nwp.shape[0]):
              R_d[i,:,:,:] = R_[i]['cascade_levels']
              means[i,:] = R_[i]['means']
              stds[i,:] = R_[i]['stds']
            
            # - set the initial and valid times
            zero_time = np.datetime64("1970-01-01T00:00:00", "ns")
            valid_times = np.array(validtime_nwp) - zero_time  
            analysis_time = validtime_nwp[0] - zero_time
            v_times[:] = np.array([np.float64(valid_times[i]) for i in range(len(valid_times))])
            timestep = metadata_nwp["accutime"] 
        
            # - define the xarray dataset variables, coordinates and attributes
            data_vars = {
                'pr_decomposed' : (['time','cascade_levels','y','x'],np.float32(R_d)),
                'means' : (['time','cascade_levels'],means),
                'stds' : (['time','cascade_levels'],stds),
                'valid_times' : (['time'],v_times,{'units': "nanoseconds since 1970-01-01 00:00:00"})
                }
            
            attrs = {
                'domain' : 'spatial',
                'normalized' : int(True),
                'compact_output' : int(False),
                'analysis_time' : int(validtime_nwp[0]),
                'timestep' : int(timestep)
                }      
                              
            # - create the xarray dataset
            ds = xr.Dataset(data_vars=data_vars,attrs=attrs)
            
            # - define the output path and save as netCDF
            output_date = f"{validtime_nwp[0].astype('datetime64[us]').astype(datetime.datetime):%Y%m%d%H%M%S}"
            outfn = Path(dir_cascade) / f"cascade_{model}_{output_date}.nc"
            
            ds.to_netcdf(
                path=outfn,
                mode='w',
                format='NETCDF4',
                encoding={'pr_decomposed' : {"zlib" : True, "complevel" : 4}}
                )
            
            logging.info('done!')
    # End loop over files   
    
    end = time.time()
    logging.info('Total process took %.2f minutes', (end - start_time_script)/60.0)
    sys.exit(erno)
