"""
This script plots the results of a DeepSDs run.
"""
import os
import math
import argparse
from os.path import join

try:
    import configparser as ConfigParser
except ImportError:
    import ConfigParser

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import prism

from utils import init_mpl, set_size


def load_data(year, scale1, n_stacked, config, prism_dir, output_dir):
    print("BEFORE LOADING DATASET")
    # read prism dataset
    # resnet parameter will not re-interpolate X
    dataset = prism.PrismSuperRes(prism_dir, year, config.get('Paths', 'elevation'), model='srcnn')

    X, elev, Y, lats, lons, times = dataset.make_test(scale1=scale1,
                                                      scale2=1./config.getint('DeepSD', 'upscale_factor')**n_stacked)

    #  resize x
    n, h, w, c = X.shape

    downscaled = xr.open_dataset(join(output_dir, 'precip_%s_downscaled.nc' % year))
    downscaled = downscaled['precip'].values[:,:,:,np.newaxis]

    return X, Y, downscaled


def plot_scatter(Y, downscaled):

    plt.figure(figsize=set_size())
    plt.scatter(Y.flatten(), downscaled.flatten())
    plt.show()


def plot(X, Y, downscaled, day, output_dir=''):
    mask = (Y[day, :, :, 0]+1)/(Y[day, :, :, 0] + 1)
    error = downscaled[day, :, :, 0] * mask - Y[day, :, :, 0]

    # Calculate the maximum error distance from 0 so that I can center the colormap at 0 for the error plot
    abs_min_error = np.abs(np.nanmin(error))
    abs_max_error = np.abs(np.nanmax(error))
    vmaxmin_err = np.maximum(abs_min_error, abs_max_error)

    fig, axs = plt.subplots(4, 1, figsize=set_size(subplot=[4, 1]))
    ymax = np.nanmax(Y)
    axs = np.ravel(axs)
    im0 = axs[0].imshow(Y[day, :, :, 0], vmax=ymax)
    axs[0].axis('off')
    axs[0].set_title("Observed")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(X[day, :, :, 0])
    axs[1].axis('off')
    axs[1].set_title("Input")
    fig.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(downscaled[day, :, :, 0] * mask, vmax=ymax)
    axs[2].axis('off')
    axs[2].set_title("Downscaled")
    fig.colorbar(im2, ax=axs[2])
    im3 = axs[3].imshow(error, vmin=-vmaxmin_err, vmax=vmaxmin_err, cmap='RdBu')
    axs[3].axis('off')
    axs[3].set_title("Error")
    fig.colorbar(im3, ax=axs[3])
    plt.savefig(os.path.join(output_dir, 'res_day_%d.pdf' % day), format='pdf', bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the predictions of a DeepSD model.')
    parser.add_argument('config', type=str,
                        help='A DeepSD config file')
    parser.add_argument('-d', '--day', type=int, default=0,
                        help='Sets which day to plot')

    args = parser.parse_args()

    init_mpl()

    config = ConfigParser.ConfigParser()
    config.read(args.config)

    output_dir = os.path.expanduser(os.path.join(config.get('SRCNN', 'scratch'),
                                                 config.get('DeepSD', 'model_name'), 'outputs'))

    if os.uname()[0] == 'DARWIN':
        # We are on MacOS and most likely on my laptop where prism data lies in ~/data/prism
        prism_dir = os.path.expanduser(os.path.join('~', 'data', 'prism'))
        print('Using local prism files in', prism_dir)
    else:
        # We are probably in Kubernetes or something
        prism_dir = os.path.expanduser(os.path.join(config.get('Paths', 'prism'), 'ppt', 'raw'))

    highest_resolution = 4.
    hr_resolution_km = config.getint('DeepSD', 'high_resolution')
    lr_resolution_km = config.getint('DeepSD', 'low_resolution')
    start = highest_resolution / hr_resolution_km
    N = int(math.log(lr_resolution_km / hr_resolution_km, config.getint('DeepSD', 'upscale_factor')))

    year1 = config.getint('DataOptions', 'max_train_year')+1
    yearlast = config.getint('DataOptions', 'max_year')
    X, Y, downscaled = load_data(year1, scale1=start, n_stacked=N,
                                 config=config, prism_dir=prism_dir, output_dir=output_dir)
    # for y in range(year1+1, yearlast+1):
    #     X_new, Y_new, downscaled_new = load_data(y, scale1=start, n_stacked=N,
    #                                              config=config, prism_dir=prism_dir, output_dir=output_dir)
    #     X = np.concatenate([X, X_new], axis=0)
    #     Y = np.concatenate([Y, Y_new], axis=0)
    #     for model_name in downscaled:
    #         downscaled[model_name]['ds'] = np.concatenate([downscaled[model_name]['ds'],
    #                                                        downscaled_new[model_name]['ds']], axis=0)

    # Just plot the days of the first year right now
    for day in range(365):
        print('day', day)
        plot(X, Y, downscaled, day, output_dir=output_dir)
    # plot_scatter(Y, downscaled)
