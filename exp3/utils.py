import errno
import os
import xarray as xr
import numpy as np
import cv2
import scipy.interpolate

import matplotlib as mpl
# Make sure matplotlib works when running in kubernetes cluster without X server
# See: https://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server
if os.uname()[0].lower() != 'darwin':
    mpl.use('Agg')

def fillmiss(x):
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0

def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    step = (x[1]-x[0])/scale
    y = np.arange(x0, xlast+step*scale, step)
    return y

def interp_tensor(X, scale, fill=True):
    nlt = int(X.shape[1]*scale)
    nln = int(X.shape[2]*scale)
    newshape = (X.shape[0], nlt, nln)
    scaled_tensor = np.empty(newshape)
    for j, im in enumerate(X):
        # fill im with nearest neighbor
        if fill:
            #im = fillmiss(im)
            im[np.isnan(im)] = 0

        scaled_tensor[j] = cv2.resize(im, (newshape[2], newshape[1]),
                                     interpolation=cv2.INTER_CUBIC)
    return scaled_tensor

def interp_da(da, scale):
    '''
    Assume da is of dimensions ('time','lat', 'lon')
    '''
    tensor = da.values

    # lets store our interpolated data
    scaled_tensor = interp_tensor(tensor, scale, fill=True)

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, scale)
    lonnew = interp_dim(da[da.dims[2]].values, scale)
    if latnew.shape[0] != scaled_tensor.shape[1]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[da[da.dims[0]].values, latnew, lonnew],
                 dims=da.dims)

def interp_da2d(da, scale, fillna=False):
    '''
    Assume da is of dimensions ('time','lat', 'lon')
    '''
    # lets store our interpolated data
    newshape = (int(da.shape[0]*scale),int(da.shape[1]*scale))
    im = da.values
    scaled_tensor = np.empty(newshape)
    # fill im with nearest neighbor
    if fillna:
        filled = fillmiss(im)
    else:
        filled = im
    scaled_tensor = cv2.resize(filled, dsize=(0,0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[0]].values, scale)
    lonnew = interp_dim(da[da.dims[1]].values, scale)
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[latnew, lonnew],
                 dims=da.dims)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_size(width=347.12354, fraction=1., subplot=[1, 1]):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
            Default value 347.12354 is textwidth for Springer llncs
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    From: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def init_mpl(usetex=True):
    nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": usetex,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
    }

    mpl.rcParams.update(nice_fonts)


if __name__=="__main__":
    import matplotlib.pyplot as plt

    fhigh = '/raid/prism/ppt_0.125x0.125/prism_ppt_interp_1981.nc'
    var='ppt'

    dshigh = xr.open_dataset(fhigh)
    dshigh = dshigh.isel(time=[0,1])
    #dshigh['ppt'] = dshigh.ppt.fillna(0)
    dalow = interp_da(dshigh.ppt, 1./8)
    danew = interp_da(dalow, 8.)

    plt.figure(figsize=(8,20))
    plt.subplot(3,1,1)
    danew.isel(time=0).plot()
    plt.subplot(3,1,2)
    dshigh.isel(time=0).ppt.plot()
    plt.subplot(3,1,3)
    dalow.isel(time=0).plot()
    plt.show()
