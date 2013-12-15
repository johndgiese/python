import os
from os import path
from re import search
import re
import itertools
import unittest

from numpy import array
from pylab import * # TODO: eventually only use necessary imports
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splrep, sproot, splev
import numpy as np

# TODO: move this back to its PEP8 spot once done switching to np.*
import copy

__all__ = ["corr", "autocorr", "linescan", "register", "upsampled_idft2",
"zpadf", "circshift", "interp_max", "findpeaks", "findvalleys", "fwhm",
"savenextfig", "interact", "run_and_plot", "scatter3", "plot_fwhm",
"closest_in_grid", "findf", "meshgridn", "dot", "outer", "iseven", "isodd", "eachimg",
"MultiplePeaks", "NoPeaksFound", "NumericTestCase"]
        
PLOTTING = False

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

## IMAGE PROCESSING

def corr(a, b):
    """Circular correlation using fourier transform."""
    A = fftn(a)
    A[0, 0] = 0.0
    A = A/std(a)
    if a is b:
        B = A
    else:
        B = fftn(b)
        B[0, 0] = 0.0
        B = B/std(b)
    Nx, Ny = a.shape
    G = ifftn(A*conj(B))/Nx/Ny
    if a.dtype.kind == 'f':
        G = real(G)
    return G

def speckle_contrast(img):
    """Calculate the contrast, standard deviation over the mean."""
    return std(img)/mean(img)

def autocorr(a):
    """Circular autocorrelation using fourier transform."""
    return corr(a, a)


def linescan(img, start, stop, npoints, method='cubic'):
    """
    Extract line scan from an image.

    Points outside the image indices are set to the mean of the image.
    
    Arguments:
    img - the image being interpolated on
    start - tuple specifying the start indices
    stop - tuple specifying the end indices
    npoints - the number of points in the linescan
    method - the kind of interpolation {'linear', 'nearest', 'cubic'}
    """
    nx, ny = img.shape
    x = linspace(start[0], stop[0], npoints)
    y = linspace(start[1], stop[1], npoints)

    x_grid = arange(nx)
    y_grid = arange(ny)
    x_grid, y_grid = meshgrid(x_grid, y_grid)
    points = zip(x_grid.flatten(), y_grid.flatten())
    z = interpolate.griddata(points, img.flatten(), (x, y), method=method)
    return x, y, z


def register(img0, img1, upsample=1, pre_calculated=False):
    """
    Efficiently register two real images to a given fraction of a pixel.

    The upsampling determines the precision of the registration; for example,
    and upsampling of 10 will register the images to within 1/10th of a pixel.

    If you are registering many images with a single 'reference' image, then
    it is inefficient to transform and normalize the reference image for each
    registration.  The function is usually used as such

    >>> v, y, x = register(img0, img1, upsample=10)

    but the following is equivalent

    >>> img0ft = fft2((img0 - mean(img0))/std(img0))
    >>> v, y, x = register(img0ft, img1, upsample=10, pre_calculated=True)

    The algorithm is based off of the following citation:
        Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
        "Efficient subpixel image registration algorithms," Opt. Lett. 33, 
        156-158 (2008).

    Arguments
    ---------
        img0 : first image (or its fft, not fftshifted)
        img1 : second image (or its fft)
        upsample : amount of upsampling
        pre_calculated : if True, assumes that img0 is already transformed and
            standardized (subtract mean, divide by std)

    Returns
    -------
        value : absolute value of the correlation between img0 and img1 after
            being shifted back; 1 if perfect correlation 0 is no correlation.  Note
            that there are not negative correlations.  
        row_shift : shift of img1
        relative to img0 along first dimnsion
        col_shift : shift of img1 relative to img0 along second dimension
    """

    ny, nx = img0.shape

    if not pre_calculated:
        img0ft = fft2(img0)
        img0ft[0, 0] = 0
        img0ft = img0ft/std(img0)
    else:
        img0ft = img0

    img1ft = fft2(img1)
    img1ft[0, 0] = 0
    img1ft = img1ft/std(img1)

    a_bconj_img = img1ft*conj(img0ft)
    corr_img = real(ifft2(a_bconj_img))
    row_first_guess, col_first_guess = _peak_shift_from_index(corr_img)

    if upsample > 1:
        # define portion of image centered on the first guess
        side = 3.0
        height = width = ceil(side*upsample)
        top = floor((row_first_guess - side/2.0)*upsample)
        left = floor((col_first_guess - side/2.0)*upsample)

        # calculate zero-padded inverse dft on area around first guess
        upsampled_corr_img_roi = upsampled_idft2(a_bconj_img, upsample, 
                height, width, top, left)

        # removing leftover imaginary part
        upsampled_corr_img_roi = real(upsampled_corr_img_roi) 

        if PLOTTING:
            imshow(upsampled_corr_img_roi)
            figure()
            imshow(img0)
            show()

        # use upsampled idft to find max in terms of the original images pixels
        roi_ind = argmax(abs(upsampled_corr_img_roi))
        roi_row, roi_col = unravel_index(roi_ind, upsampled_corr_img_roi.shape)
        row_final_guess = float(top  + roi_row)/upsample
        col_final_guess = float(left + roi_col)/upsample
        value = upsampled_corr_img_roi[roi_row, roi_col]
    else:
        row_final_guess = row_first_guess
        col_final_guess = col_first_guess
        value = corr_img[row_final_guess, col_final_guess]
    value = value/nx/ny

    return value, row_final_guess, col_final_guess

def _peak_shift_from_index(img):
    """
    Determine the image shift from the correlation.

    Because the correlation is taken in the fourier domain, it is a
    "circular correlation," and thus a negative shift appears to wrap
    around the other side of the correlation image; this function accounts
    for this.  
    
    Note that if you have shifts greater than half the image size it is
    ambiguous because of the circular nature of the correlation.  To simplify
    the code, all shifts are assumed to be less than half the image
    size.

    Also note that the original images were assumed to be real.
    """
    ny, nx = img.shape
    row, col = unravel_index(abs(argmax(img)), (ny, nx))
    if row > floor(ny/2):
        row = row - ny
    if col > floor(nx/2):
        col = col - nx
    if row < -floor(ny/2):
        row = row + ny
    if col < -floor(nx/2):
        col = col + nx
    return row, col


def upsampled_idft2(a, upsample, height, width, top, left):
    """
    Calculate a portion of the upsampled inverse DFT of a using a matrix.
    
    Height, width, top and left specify the portion of the IDFT that will be
    taken in terms of upsampled pixels

    The returned array is normalized so that it has the same values as
    ift(input) --- of course there will be slight variations due to
    interpolatation.

    NOTE: this function doesn't "properly" handle the nyquist frequency, so there
    may be a small error but shouldn't matter for reasnobly large matrices.

    See G. Strang, Introduction to Linear Algebra, Section 10.3 for some
    explanation of the fourier matrices.
    """

    rows_a, cols_a = a.shape

    # generate matrix that does the column inverse dft
    wc = np.fft.fftfreq(rows_a)/upsample
    wr = arange(top, top + height)
    w = outer(wr, wc)
    kernc = exp(2j*pi*w)

    # generate matrix that does the row inverse dft
    wc = np.fft.fftfreq(cols_a)/upsample
    wr = arange(left, left + width)
    w = outer(wc, wr)
    kernr = exp(2j*pi*w)

    norm = rows_a*cols_a

    return dot(kernc, a, kernr)/norm

def zpadf(A, zeros):
    """
    Zeropad in the frequency domain.

    Arguments
    ---------
    A : array to be padded (the zero-frequency should be at [0, 0])
    zeros : either an int indicating the amount of padding along each dimension, or
        a tuple indicating the number of zeros to be padded along each dimension
    """

    halfs = [int(h/2) for h in A.shape]
    new_shape = A.shape + array(zeros)
    dim = len(new_shape)

    out = np.zeros(new_shape, dtype=A.dtype)

    # iterate over the corners of the n-dim array
    for vert in itertools.product((1, 0), repeat=dim):
        indices = []

        # create proper slice for each dimension
        for h, front, odd in zip(halfs, vert, isodd(A.shape)):
            if front:
                if odd:
                    ind = slice(0, h + 1)
                else:
                    ind = slice(0, h)
            else:
                ind = slice(-h, None)
            indices.append(ind)
        
        # grab corner data from original and place in zeropadded
        out[indices] = A[indices]

    # even-length dimensions require care because the old nyquest frequency had
    # only a single element to keep the transform balanced, the nyquist
    # frequency needs to be split in two
    whole_array = [slice(None) for s in new_shape]
    for dim, h, s, sn in zip(range(len(halfs)), halfs, A.shape, new_shape):
        if iseven(s) and sn > s:
            old_nyquist_top = copy.copy(whole_array)
            old_nyquist_top[dim] = -h
            old_nyquist_bot = copy.copy(whole_array)
            old_nyquist_bot[dim] = h

            out[old_nyquist_top] = out[old_nyquist_top]/2.0
            out[old_nyquist_bot] = out[old_nyquist_top]
    return out

def _linphase(N, shift):
    N = double(N)
    k = linspace(0, 1, N, endpoint=False)
    linphase = empty(N, dtype="complex")
    linphase[k < 0.5] = exp(-2j*pi*shift*k[k < 0.5])
    linphase[k > 0.5] = exp(-2j*pi*shift*(k[k > 0.5] - 1.0))
    linphase[k == 0.5] = cos(2*pi*shift/2.0)
    return linphase

def circshift(a, shift, transformed=False):
    """
    Circular subpixel shift a 1D array.
    
    Algorithm uses the Fourier-shift theorem with minimal-slope interpolation.

    Arguments
    _________
    a : complex ndarray
        Array to be circularly shifted.
    shift : int, or list of shifts if multidimensional
        Number of pixels to be shifted along each dimension (can be fractional)
    transformed : boolean
        Has the array already been transformed. Default is False.
    """
    if type(shift) in [int, float]:
        shift = [shift]

    if not transformed:
        aft = fftn(a)
    else:
        aft = a

    shape = a.shape

    dim = len(shape)
    linphases = []
    for s, N in zip(shift, shape):
        linphases.append(_linphase(N, s))
    linphase = outer(*linphases)

    ashifted = ifftn(aft*linphase)
    if not transformed and not a.dtype.kind == 'c':
        ashifted = real(ashifted)
    return ashifted


def interp_max(img, x=None, y=None, precision=10):
    """
    Find the maximum value in an image using cubic interpolation.

    Assumes that the maximum value is near the maximum pixel.
    
    Arguments:
    X - x grid positions
    Y - y grid positions
    precision - increase in grid spacing from interpolation

    Returns:
    max - the value of the maximum
    x - the x position of the maximum
    y - the y position of the maximum
    """
    nx, ny = img.shape
    if x == None:
        x = np.arange(nx)
    if y == None:
        y = np.arange(ny)

    # find max of current image
    xmax, ymax = np.unravel_index(argmax(img), img.shape)

    # create subimg centered around the maximum
    xe, ye = closest_in_grid(nx, ny, xmax + 5, ymax + 5)
    xs, ys = closest_in_grid(nx, ny, xmax - 5, ymax - 5)
    xe = x[xe]
    ye = y[ye]
    xs = x[xs]
    ys = y[ys]

    # +1 to stay on original grid
    xx = linspace(xs, xe, precision*(xe - xs) + 1) 
    yy = linspace(ys, ye, precision*(ye - ys) + 1)
    XX, YY = meshgrid(xx, yy)

    X, Y = meshgrid(x, y)
    points = zip(X.flatten(), Y.flatten())

    sum_img = img[xs:xe, ys:ye]
    values = img.flatten()
    ZZ = interpolate.griddata(points, values, 
            (XX.flatten(), YY.flatten()), method='cubic')
    imax = argmax(ZZ)
    xmax, ymax = unravel_index(imax, XX.shape)
    return XX[xmax, ymax], YY[xmax, ymax], ZZ[imax]


## GENERAL SIGNAL PROCESSING

# TODO: test this
def findpeaks(X, threshold=None, smooth=1, width=1):
    """ 
    Find peaks in an array.

    optional parameters:
    smooth - apply a moving average of this length, default=1
    width - the approximate width of the peak
    """
    N = len(X)
    if not smooth == 1:
        X = convolve(X.astype(float), ones(smooth), mode='same')/smooth

    peaks = []
    indices = []
    for i in range(width, N - width):
        fd1 = X[i] - X[i+1]
        bd1 = X[i] - X[i-1]
        fdw = X[i] - X[i+width]
        bdw = X[i] - X[i-width]
        if fdw > 0 and bdw > 0 and fd1 > 0 and bd1 > 0:

            if threshold:
                if fd1 < threshold or bd1 < threshold:
                    break

            peaks.append(X[i])
            indices.append(i)

    return array(indices)


def findvalleys(X, *args, **kwargs):
    """Find the valleys in an array.  Same options as findpeaks."""
    return findpeaks(-X, *args, **kwargs);


def fwhm(x, y):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.
http://motherboard.vice.com/blog/solar-powered-trash-cans-saved-philadelphia-almost-a-million-bucks
    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """
    half_max = amax(y)/2.0
    s = splrep(x, y - half_max)
    roots = sproot(s)
    
    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])


## VISUALIZATION

def next_available_name(folder, fname, zeros=4):
    full = path.join(folder, fname)
    root, ext = path.splitext(full)
    match = re.findall(r'\d+', root)
    if match:
        num = int(match[-1]) + 1
    else:
        num = 0
    fname = root + str(num).zfill(zeros) + ext

    return fname

def savenextfig(folder, fname, zeros=4, *args, **kwargs):
    """
    Save figure using next available numbered name.

    If no number is present in the name, then 0 is used.
    
    examples:
        hello.png --> hello0.png
        already_five.png --> already_five6.png

    """
    fname = next_available_name(folder, fname, zeros)
    savefig(fname, *args, **kwargs)


def interact(func, x, y, adjustable):
    """ Interactively plot a functions' values. 

    Arguments:
    func -- the function that generates the plotted values
    x -- a string, indicating the variable to be plotted on the x-axis
         if None, the y-values will be plotted against their indicies
         if it is a numpy array, it will be used for plotting
    y -- a string, indicating the variable to be plotted on the y-axis
    adjustable -- a dict whose key's are the names of variables being
         passed to func, and value can be either a value, or a tuple.
         If the value is a single value, it will be staticly passed to 
         func on every plot update.
         If the value is a tuple, a slider will be generated that will allow 
         you to vary this function through out these values.  The initial 
         value will be in the middle of the range.
    """

    from matplotlib.widgets import Slider, Button, RadioButtons
    
    # initialize arguments
    args = {}
    for k, v in adjustable.items():
        if not isinstance(v, tuple):
            del adjustable[k] # remove static variables
            args[k] = v

    # get vars from sliders, run func, then plot
    def run_and_plot(event):
        for sl in sliders:
            args[sl.label.get_text()] = sl.val

        ret = func(**args)
        if not y:
            yval = ret
        else: # if isinstance(y, (str, int))
            yval = ret[y]

        if isinstance(x, (str, int)):
            xval = ret[x]
        else:
            xval = x
        pax.plot(xval, yval, lw=2, axes=pax)
        draw()
    
    # inialize the plot
    pax = subplot(111)
    subplots_adjust(bottom=0.25)
    
    bot = 0.1
    axs = []
    sliders = []
    for var, ends in adjustable.items():
        axs.append(axes([0.15, bot, 0.75, 0.03]))
        bot += 0.05
        sliders.append(Slider(axs[-1], var, ends[0], ends[1], valinit=mean(ends)))

    resetax = axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Plot', hovercolor='0.975')
    button.on_clicked(run_and_plot)

    run_and_plot(None) # make the first plot
    show()


def scatter3(X, Y, Z):
    """Create a 3D scatter plot."""
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X, Y, Z)
    return fig
    

def plot_fwhm(x, y, k=10):
    y_max = amax(y)
    s = splrep(x, y - y_max/2.0)
    r1, r2 = sproot(s)
    xx = linspace(amin(x), amax(x), 200)
    yy = splev(xx, s) + y_max/2.0

    f = figure()
    plot(x, y, 'ko')
    plot(xx, yy, 'r-')
    axvspan(r1, r2, facecolor='k', alpha=0.5)
    return f



## CONVENIENCE FUNCTIONS 

def closest_in_grid(gx, gy, x, y):
    """Given grid size and a point, return the closest point in the grid."""

    x = max(min(gx - 1, x), 0)
    y = max(min(gy - 1, y), 0)
    return x, y


# TODO: optimize this
def findf(x):
    """Return the indice of the first true value in x."""

    i = 0
    for val in x:
        if val:
            return i
        i += 1
    return false


def meshgridn(*arrs):
    """A multi-dimensional version of meshgrid."""

    arrs = map(asarray, arrs)
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s
    
    first_dtype = arrs[0].dtype
    for arr in arrs[1:]:
        if not arr.dtype == first_dtype:
            raise Exception("All array datatypes must be the same.")

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = arr.reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)
    return ans


def dot(*arrays):
    """Inner product of many arrays."""
    return reduce(np.dot, arrays)

def outer(*arrays):
    """Tensor product of many arrays."""
    return reduce(np.multiply.outer, arrays)

@vectorize
def viseven(el):
    return 1 - el % 2

def iseven(el):
    if type(el) in [int, float]:
        return 1 - el % 2
    else:
        return viseven(el)

@vectorize
def visodd(el):
    return el % 2

def isodd(el):
    if type(el) in [int, float]:
        return el % 2
    else:
        return visodd(el)

def eachimg(folder, regexp=r'^$'):
    """Iterator for images matching regexp in folder.  Also returns the number of images."""

    if type(regexp) == str:
        regexp = re.compile(regexp)
    imgnames = [path.join(folder, img) for img in os.listdir(folder) if regexp.match(img)]
    def imgs():
        for img in sorted(imgnames):
            yield imread(img)
    return imgs(), len(imgnames)

## TESTING

class NumericTestCase(unittest.TestCase):
    """Test case with assertions for numeric computing."""

    def assertArraysClose(self, a, b, rtol=1e-05, atol=1e-08):
        """Check that the two arrays are almost equal, see np.allclose."""

        if not a.shape == b.shape:
            msg = "The two arrays don't even have the same shape!\n"
            msg += "The first array has {} shape.\n".format(a.shape)
            msg += "The second array has {} shape.\n".format(b.shape)
            raise self.failureException(msg)

        arrays_close = np.allclose(a, b, rtol=rtol, atol=atol)

        if not arrays_close:
            Ea = sum(abs(a)**2)
            Eb = sum(abs(b)**2)
            Ed = sum(abs(a - b)**2)
            mean_angle_diff = mean(abs(angle(a) - angle(b)))*180/pi
            msg = "The two arrays are not equal.\n"
            msg += "The first 10 elements of a are:\n{}\n".format(a.flatten()[:20])
            msg += "The first 10 elements of b are:\n{}\n".format(b.flatten()[:20])
            msg += "The energy of a is %s.\n" % (str(Ea),)
            msg += "The energy of b is %s.\n" % (str(Eb),)
            msg += "The energy of a - b is %s.\n" % (str(Ed),)
            msg += "The mean angle difference is {} deg.\n\n".format(mean_angle_diff)
            raise self.failureException(msg)

    def assertAnglesAlmostEqual(self, ai1, ai2, rtol=1e-05, atol=1e-08):
        """Two angles are equal within n*2*pi."""

        a1 = ai1 % pi
        a2 = ai2 % pi

        close_same = np.allclose(a1, a2, rtol=rtol, atol=atol)

        # the angles still could be close, but fall on the opposite sides of pi
        close_oppo = False
        if a1 < pi/2 and a2 > pi/2:
            close_oppo = np.allclose(a1, a2 - pi)
        if a1 > pi/2 and a2 < pi/2:
            close_oppo = np.allclose(a1 - pi, a2)

        if close_same == close_oppo:
            msg = "Angles {} and {} are not almost equal.".format(ai1, ai2)
            msg += "After unwrapping we have {} and {}.".format(a1, a2)
            raise self.failureException(msg)
