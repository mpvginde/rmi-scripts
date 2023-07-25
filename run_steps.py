#!/bin/env python3

"""
Run a stochastic nowcast blended with NWP forecast based on STEPS (Bowler et. al 2006)
This script assumes the NWP forecast cascade decomposition is already performed with
decompose_nwp.py
Author: Michiel Van Ginderachter 25/02/2022
michiel.vanginderachter@meteo.be
"""
"""
Scipt needs to be run as:
run_steps.py YYYYMMDDHHMM FC NENS CPUS
with 
YYYYMMDDHHMM : startdate of the forecast (also used in .nc filename)
FC           : forecast length (number of timesteps)
NENS         : number of ensemble members
CPUS         : number of workers
CASESTYPE    : the type of case (flooding/no-radar-rain-no-nwp-rain/no-radar-rain-nwp-rain)
eg.: run_steps.py 202101270600 72 10 4 flooding
"""

import os
import datetime
import netCDF4
import numpy as np
import pprint
import sys
import time
import matplotlib.pyplot as plt
import imageio

arg_list = sys.argv
if len(arg_list) < 2:
    print("Usage: run_steps.py YYYYMMDDHHMM FC NENS CPUS CASESTYPE")
    sys.exit(1)

import dask
import pysteps

# 1. Load the command line arguments

startdate = datetime.datetime.strptime(arg_list[1],"%Y%m%d%H%M")
fc_length = int(arg_list[2])
nens = int(arg_list[3])
ncores = int(arg_list[4])
casetype = arg_list[5]
#startdate = datetime.datetime.strptime('202101270200','%Y%m%d%H%M')
#nens=5
#fc_length=72
#ncores = 10

threshold = 0.1
ncascade = 8
dir_cascade = os.path.join(f'/home/michielv/pysteps/hackaton/nwp/{casetype}',startdate.strftime('%Y%m%d'))
dir_motion = dir_cascade
dir_skill = '/home/michielv/pysteps/hackaton/skill'
dir_gif = '/home/michielv/pysteps/hackaton/gifs'
dir_nwc = '/home/michielv/pysteps/hackaton/nwc'
print("Started nowcast with:")
print(r' Startdate: %s' % startdate.strftime("%Y-%m-%d %H:%M"))
print(r' Forecast length: %i timesteps' % fc_length)
print(r' Number of ensemble members: %i' % nens)
print(r' Number of workers: %i' % ncores)
print(r' Rain/No-rain threshold: %.2f' % threshold)
print(r' Number of cascade levels: %i' % ncascade)
print(r' Motion vectors are loaded from: %s' % dir_motion)
print(r' Cascade decompositions are loaded from: %s' % dir_cascade)
print(r' NWP skill is saved in: %s' % dir_skill)
print(r' Nowcast netCDF file is saved in: %s' % dir_nwc)
print('')
# 2. Set the directories and data sources
data_src_radar = "rmi"
data_src_nwp = "rmi_nwp"

# 3. Load the radar analyses 
root_path = '/home/michielv/pysteps/hackaton/radar' #pysteps.rcparams.data_sources[data_src_radar]["root_path"]
path_fmt = f'{casetype}/%Y%m%d' #pysteps.rcparams.data_sources[data_src_radar]["path_fmt"]
fn_pattern = '%Y%m%d%H%M%S.rad.bhbjbwdnfa.comp.rate.qpe2' #pysteps.rcparams.data_sources[data_src_radar]["fn_pattern"]
fn_ext = 'hdf' #pysteps.rcparams.data_sources[data_src_radar]["fn_ext"]
importer_name = pysteps.rcparams.data_sources[data_src_radar]["importer"]
importer_kwargs = pysteps.rcparams.data_sources[data_src_radar]["importer_kwargs"]
timestep = pysteps.rcparams.data_sources[data_src_radar]["timestep"]

print('Loading and preprocessing radar analysis...')
fn_radar = pysteps.io.find_by_date(
        date = startdate,
        root_path = root_path,
        path_fmt = path_fmt,
        fn_pattern = fn_pattern,
        fn_ext = fn_ext,
        timestep = timestep,
        num_prev_files = 2
)

importer_radar = pysteps.io.get_method(importer_name,"importer")
r_radar, _, metadata_radar = pysteps.io.read_timeseries(
        inputfns = fn_radar,
        importer = importer_radar,
        legacy=False
)

metadata_nwc = metadata_radar.copy()
metadata_nwc['shape'] = r_radar.shape[1:]

# 4. Prepare the radar analyses
converter = pysteps.utils.get_method("mm/h")
r_radar, metadata_radar = converter(r_radar,metadata_radar)

r_radar[r_radar < threshold] = 0.0
metadata_radar["threshold"] = threshold

transformer = pysteps.utils.get_method("dB")
r_radar, metadata_radar = transformer(
        R = r_radar,
        metadata = metadata_radar,
        threshold = threshold,
#        zerovalue=-10.0
)

oflow_method = pysteps.motion.get_method("LK")
v_radar = oflow_method(r_radar)
print('done!')
print('')
# 5. Get the available NWP dates, select the closest one and load the velocities and cascade
fcsttimes_nwp = []
for file in os.listdir(dir_motion):
    fcsttimes_nwp.append(
            datetime.datetime.strptime(file.split("_")[2].split('.')[0],'%Y%m%d%H%M%S')
    )

startdate_nwp = startdate + datetime.timedelta(minutes=timestep)
date_nwp = startdate_nwp + max([nwptime - startdate_nwp for nwptime in fcsttimes_nwp if nwptime <= startdate_nwp]) 

model='ao13'
fn_motion = os.path.join(dir_motion,
        r'motion_%s_%s.npy' % (model,date_nwp.strftime('%Y%m%d%H%M%S'))
)
fn_cascade = os.path.join(dir_cascade,
        r'cascade_%s_%s.nc' % (model,date_nwp.strftime('%Y%m%d%H%M%S'))
)

if not os.path.exists(fn_cascade):
    raise Exception('Cascade file %s accompanying motion file %s does not exist' % (fn_cascade,fn_motion))
print(r'Loading NWP cascade and velocities for run started at %s...' % date_nwp.strftime('%Y-%m-%d %H:%M'))
r_decomposed_nwp, v_nwp = pysteps.blending.utils.load_NWP(
        input_nc_path_decomp = fn_cascade,
        input_path_velocities = fn_motion,
        start_time=np.datetime64(startdate_nwp), 
        n_timesteps=fc_length
)

# 5.bis Make sure the NWP cascade and velocity fields have an extra 'n_models' dimension
r_decomposed_nwp = np.stack([r_decomposed_nwp])
v_nwp = np.stack([v_nwp])
print('done!')
# 6. Prepare the netCDF exporter-function
def write_netCDF(R):
    R, _ = converter(R, metadata_radar)
    pysteps.io.export_forecast_dataset(R, exporter)

exporter = pysteps.io.initialize_forecast_exporter_netcdf(
        outpath = dir_nwc,
        outfnprefix = 'blended_nowcast_%s' % startdate.strftime("%Y%m%d%H%M"),
        startdate = startdate_nwp,
        timestep = timestep,
        n_timesteps = fc_length,
        shape = metadata_nwc['shape'],
        n_ens_members = nens,
        metadata = metadata_nwc,
        incremental = 'timestep'
)

# 6. Start the nowcast
nwc_method = pysteps.blending.get_method("steps")
r_nwc = nwc_method(
        precip = r_radar,
        precip_models = r_decomposed_nwp,
        velocity = v_radar,
        velocity_models = v_nwp,
        timesteps = fc_length,
        timestep = timestep,
        issuetime = startdate,
        n_ens_members = nens,
        n_cascade_levels = ncascade,
        blend_nwp_members = False,
        precip_thr = metadata_radar['threshold'],
        kmperpixel = metadata_radar['xpixelsize']/1000.0,
        extrap_method = 'semilagrangian',
        decomp_method = 'fft',
        bandpass_filter_method = 'gaussian',
        noise_method = 'nonparametric',
        noise_stddev_adj = 'auto',
        ar_order = 2,
        vel_pert_method = None,
        weights_method = 'bps',
        conditional = False,
        probmatching_method = 'cdf',
        mask_method = 'incremental',
        callback = write_netCDF,
        return_output = True,
        seed = 24,
        num_workers = ncores,
        fft_method = 'numpy',
        domain = 'spatial',
        outdir_path_skill = dir_skill,
        extrap_kwargs = None,
        filter_kwargs = None,
        noise_kwargs = None,
        vel_pert_kwargs = None,
        clim_kwargs = None,
        mask_kwargs = None,
        measure_time = False
)

r_nwc, metadata_nwc = transformer(
        R = r_nwc,
        threshold = -10,
        inverse = True
)

pysteps.io.close_forecast_files(exporter)


# 7. Build GIF

filenames = []
for i in range(r_nwc.shape[1]):
    title = 'Precipitation nowcast %s + %i min\nvaliddate: %s' % (startdate.strftime("%Y-%m-%d %H:%M"), (i+1)*timestep, startdate+datetime.timedelta(minutes=(i+1)*5))
#    datestr = datetime.datetime.strftime(fns[1][i],"%Y-%m-%d %H:%M UTC")
    plt.figure(figsize = (16,12))
    for j in range(r_nwc.shape[0]):
        plt.subplot(231+j)
        pysteps.visualization.plot_precip_field(
            r_nwc[j,i,:,:],
            geodata = metadata_radar,
            title = r'Member %i' % j
        )
    plt.suptitle(title)
    plt.tight_layout()
    filename = f'{i}.png'
    filenames.append(filename)
    plt.savefig(filename,dpi=72)
    plt.close()


# build gif
kargs = { 'duration' : 0.4 }
with imageio.get_writer(os.path.join(dir_gif,
    r'forecast_%s_nwp_%s.gif' % (startdate.strftime('%Y%m%d%H%M'),
    date_nwp.strftime('%Y%m%d%H%M'))), mode='I', **kargs) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# remove files
for filename in set(filenames):
    os.remove(filename)


