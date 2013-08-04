import unittest

# I am slowly transitioning to using proper namespacing
from pylab import *
import pylab as pl
import scipy
import numpy as np

import mymath as mm
from mymath import NumericTestCase

PLOTTING = False

class TestFwhm(NumericTestCase):

    def setUp(self):
        self.x = arange(10)

    def test_multiple_peaks(self):
        y = array([1, 10, 6, 3, 5, 10, 7, 2, 1, 1])
        self.assertRaises(mm.MultiplePeaks, mm.fwhm, self.x, y)

    def test_flat(self):
        y = ones(self.x.shape)
        self.assertRaises(mm.NoPeaksFound, mm.fwhm, self.x, y)

    def test_gaussian(self):
        sigma = rand()
        x = linspace(-2*sigma, 2*sigma, 100)
        y = exp(-(x/sigma)**2/2)
        fwhm = mm.fwhm(x, y)
        gaus_fwhm = 2*sqrt(2*log(2))*sigma
        self.assertAlmostEqual(fwhm, gaus_fwhm, places=3)

class TestCorr(NumericTestCase):

    def test_autocorr(self):
        A = rand(10, 15)
        corr0 = mm.autocorr(A)
        self.assertAlmostEqual(corr0[0, 0], 1.0)

    def test_normalization(self):
        """All values of the correlation should be between -1 and 1."""
        A = rand(10, 15)
        B = rand(10, 15)
        C = mm.corr(A, B)
        self.assertTrue((abs(C) <= 1.000000001).all())

class TestZpadf(NumericTestCase):

    def test_1d(self):
        a = array([1.0, 1.0])
        za = array([1.0, 0.5, 0, 0.5])
        self.assertArraysClose(za, mm.zpadf(a, 2))
        self.assertArraysClose(za, mm.zpadf(a, (2,)))


        a = array([1.0, 2, 3])
        za = array([1, 2, 0, 0, 3])
        self.assertArraysClose(za, mm.zpadf(a, 2))

        a = array([1.0, 2, 3, 4])
        za = array([1, 2, 1.5, 0, 1.5, 4])
        self.assertArraysClose(za, mm.zpadf(a, 2))

        a = array([1.0, 2, 3, 4])
        za = array([1, 2, 1.5, 0, 0, 1.5, 4])
        self.assertArraysClose(za, mm.zpadf(a, 3))

    def test_2d(self):
        a = array([
            [1.0, 2], 
            [3, 8]
        ])
        za = array([
            [1,   1, 0, 1], 
            [1.5, 2, 0, 2],
            [0,   0, 0, 0],
            [1.5, 2, 0, 2]
        ])
        self.assertArraysClose(za, mm.zpadf(a, 2))
        self.assertArraysClose(za, mm.zpadf(a, (2, 2)))

        za = array([
            [1,   2], 
            [1.5, 4],
            [0,   0],
            [1.5, 4]
        ])
        self.assertArraysClose(za, mm.zpadf(a, (2, 0)))

        za = array([
            [1,   1, 0, 1], 
            [3,   4, 0, 4],
        ])
        self.assertArraysClose(za, mm.zpadf(a, (0, 2)))


class TestInterpMax(NumericTestCase):

    def test_closest_in_grid(self):
        x, y = mm.closest_in_grid(2, 3, 0, 0)
        self.assertEqual((x, y), (0, 0))
        x, y = mm.closest_in_grid(2, 3, -1, 0)
        self.assertEqual((x, y), (0, 0))
        x, y = mm.closest_in_grid(2, 3, -1, -4)
        self.assertEqual((x, y), (0, 0))

        x, y = mm.closest_in_grid(2, 3, 2, 3)
        self.assertEqual((x, y), (1, 2))
        x, y = mm.closest_in_grid(2, 3, 1, 30)
        self.assertEqual((x, y), (1, 2))
        x, y = mm.closest_in_grid(2, 3, 100, 0)
        self.assertEqual((x, y), (1, 0))

    def test_easy(self):
        img = array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        x = [1, 2, 3]
        y = [1, 2, 3]
        xx, yy, zz = mm.interp_max(img, x=x, y=y)
        self.assertAlmostEqual(xx, 2)
        self.assertAlmostEqual(yy, 2)
        self.assertAlmostEqual(zz, 1)

    def test_easy_using_indices(self):
        img = array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        xx, yy, zz = mm.interp_max(img)
        self.assertAlmostEqual(xx, 1)
        self.assertAlmostEqual(yy, 1)
        self.assertAlmostEqual(zz, 1)

    def test_surface(self):
        precision = 200
        # restrict range of zero to middle or 1x1 grid
        xm = rand()*0.8 + 0.1
        ym = rand()*0.8 + 0.1
        x = linspace(0, 1, 10)
        y = linspace(0, 1, 10)
        X, Y = meshgrid(x, y)
        def func(x, y):
            return -3*(x - xm)**2 - (y - ym)**2
        Z = reshape(array(map(func, X, Y)), (len(x), len(y)))
        xx, yy, zz = mm.interp_max(Z, x, y, precision=precision)
        expected_places = 1
        self.assertAlmostEqual(xx, xm, places=expected_places)
        self.assertAlmostEqual(yy, ym, places=expected_places)

class TestRegister(NumericTestCase):

    def setUp(self):
        img0 = scipy.misc.lena()
        ny, nx = img0.shape
        row_shift = randint(-ny/2, ny/2)
        col_shift = randint(-nx/2, nx/2)
        img1 = mm.circshift(img0, (row_shift, col_shift))

        self.img0 = img0
        self.img1 = img1
        self.nx = nx
        self.ny = ny
        self.col_shift = col_shift
        self.row_shift = row_shift

    def test_upsample_1(self):
        """Test algorithm without subpixel registration."""

        img0 = scipy.misc.lena()
        ny, nx = img0.shape
        row_shift = randint(-ny/2, ny/2)
        col_shift = randint(-nx/2, nx/2)
        row_shift = 10
        col_shift = 27
        img1 = mm.circshift(img0, (row_shift, col_shift))
        img1 = abs(img1)

        if PLOTTING:
            subplot(211)
            imshow(img0)
            title('original')
            subplot(212)
            imshow(img1)
            title('shifted by {}x{}'.format(row_shift, col_shift))
            show()

        val, row, col = mm.register(img0, img1)
        self.assertEqual(row, row_shift)
        self.assertEqual(col, col_shift)

    def test_upsample_1000(self):
        img0 = scipy.misc.lena()
        ny, nx = img0.shape
        row_shift = (rand() - 0.5)*ny
        col_shift = (rand() - 0.5)*nx
        img1 = mm.circshift(img0, (row_shift, col_shift))

        val, row, col = mm.register(img0, img1, upsample=105)
        self.assertAlmostEqual(row, row_shift, places=2)
        self.assertAlmostEqual(col, col_shift, places=2)

    def test_normalization(self):
        img0 = zeros([30, 30])
        img0[0, 0] = 2.0
        img0[0, 1] = 4.0
        ny, nx = img0.shape
        row_shift = 3
        col_shift = 5
        img1 = zeros([30, 30])
        img1[3, 3] = 2.0
        img1[3, 4] = 4.0
        
        v, y, x = mm.register(img0, img1)
        self.assertAlmostEqual(v, 1, places=8)

        v, y, x = mm.register(img0, 0.5*img1)
        self.assertAlmostEqual(v, 1, places=8)

        v, y, x = mm.register(img0, img1, upsample=4)
        self.assertAlmostEqual(v, 1, places=8)

        v, y, x = mm.register(img0, 0.5*img1, upsample=4)
        self.assertAlmostEqual(v, 1, places=8)


class TestDFTUpsample(NumericTestCase):

    def remove_nyquist(self, a, aft):
        ny, nx = a.shape
        if mm.iseven(ny):
            aft[ny/2,:] = 0
        if mm.iseven(nx):
            aft[:,nx/2] = 0
        a = real(ifft2(aft))
        return a, aft

    def setUp(self):

        ## pick random odd upsampling
        upsample = randint(1, 16)
        if mm.iseven(upsample):
            half = int((upsample - 1)/2.0)
        else:
            half = int(upsample/2.0) - 1

        # setup random area to calculate idft on
        nx = randint(10, 100)
        ny = randint(10, 100)
        height = randint(4, ny*upsample)
        width = randint(4, nx*upsample)
        try:
            top = randint(0, ny*upsample - height)
        except:
            top = 0
        try:
            left = randint(0, nx*upsample - width)
        except:
            left = 0

        # generate a random image
        aa = rand(ny, nx)
        aa[:-5, :-5] += 1
        aa[:-2, :] += 2
        a = fft2(aa)

        aa, a = self.remove_nyquist(aa, a)

        ## attach data to test case
        self.aa = aa
        self.a = a
        self.nx = nx
        self.ny = ny
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.upsample = upsample

    def test_equivalence(self):
        """
        The dft_upsample function should be equivalent to:
        1. Embedding the array "a" in an array that is "upsample" times larger
           in each dimension. ifftshift to bring the center of the image to
           (0, 0).
        2. Take the IFFT of the larger array
        3. Extract an [height, width] region of the result. Starting with the 
           [top, left] element.
        """
        aa = self.aa
        a = self.a
        nx = self.nx
        ny = self.ny
        height = self.height
        width = self.width
        top = self.top
        left = self.left
        upsample = self.upsample

        ## calculate the slow way
        extra_zeros = ((upsample - 1)*ny, (upsample - 1)*nx)
        padded = mm.zpadf(a, extra_zeros)
        b_slow_big = ifft2(padded)
        b_slow = b_slow_big[
            top:top + height,
            left:left + width,
        ]

        # calculate the fast way
        b_fast = mm.upsampled_idft2(a, upsample, height, width, top, left)
        b_fast_big = mm.upsampled_idft2(a, upsample, upsample*ny, upsample*nx, 0, 0)

        if PLOTTING:
            subplot(411)
            imshow(abs(b_fast_big))
            subplot(412)
            imshow(abs(b_slow_big))
            subplot(413)
            imshow(abs(b_fast))
            subplot(414)
            imshow(abs(b_slow))
            title('{}x{} starting at {}x{}'.format(height, width, top, left))
            figure()
            imshow(aa)
            show()

        if PLOTTING:
            subplot(221)
            imshow(abs(b_slow))
            title('slow abs')
            subplot(222)
            imshow(abs(b_fast))
            title('fast abs')
            subplot(223)
            imshow(unwrap(angle(b_slow)))
            title('slow angle')
            subplot(224)
            imshow(unwrap(angle(b_fast)))
            title('fast angle')
            show()

        # are they the same (within a multiplier)
        b_slow = b_slow/mean(abs(b_slow))
        b_fast = b_fast/mean(abs(b_fast))
        self.assertArraysClose(b_slow, b_fast, rtol=1e-2)

    def test_normalization(self):
        """The function should be properly normalized."""
        a = ones([10, 10])
        aa = fft2(a)
        b = mm.upsampled_idft2(aa, 3, 30, 30, 0, 0)
        self.assertAlmostEqual(amax(a), amax(abs(b)))

    def test_full_array(self):
        nx = self.nx
        ny = self.ny
        upsample = self.upsample

        aa = rand(ny, nx)
        aa[:-5, :-5] += 1
        aa[:-2, :] += 2
        a = fft2(aa)
        aa, a = self.remove_nyquist(aa, a)

        ## calculate the slow way
        extra_zeros = ((upsample - 1)*ny, (upsample - 1)*nx)
        padded = mm.zpadf(a, extra_zeros)
        b_slow_big = ifft2(padded)

        # calculate the fast way
        b_fast_big = mm.upsampled_idft2(a, upsample, upsample*ny, upsample*nx, 0, 0)

        b_slow_big = b_slow_big/mean(abs(b_slow_big))
        b_fast_big = b_fast_big/mean(abs(b_fast_big))

        if PLOTTING:
            subplot(131)
            imshow(abs(b_fast_big - b_slow_big))
            colorbar()
            subplot(132)
            imshow(angle(b_fast_big))
            colorbar()
            subplot(133)
            imshow(angle(b_slow_big))
            colorbar()
            show()

        self.assertArraysClose(abs(b_fast_big), abs(b_slow_big), rtol=1e-2)

    @unittest.expectedFailure
    def test_known2x2(self):
        # expected to fail because upsampled_idft2 doesn't handle the nyquist
        # component properly
        A = array([[1, 4], [0, 2]])
        AA = fft2(A)
        known_out = array([[4, 2.5], [3, 1.75]])
        out = mm.upsampled_idft2(AA, 2, 2, 2, 0, 2)
        self.assertArraysClose(known_out, out)

        # code comparing the ideal fourier matrix approach with proper
        # zero-padding, to the idft_upsampled
        AAA = zpadf(AA, (2, 2))
        F = array([[1, 1, 1, 1],[1, 1j, -1, -1j],[1,-1,1,-1],[1,-1j,-1,1j]])/4.0
        out2 = mm.dot(F, AAA, F)*4.0
        out = mm.upsampled_idft2(AA, 2, 4, 4, 0, 0)

    def test_simple(self):
        """
        Fourier transforms can be accomplished using matrices.

        You need two of them (one for each dimension).

        This test is a simple sanity check.
        """

        a = fft2(array([[0, 2, 2, 0],
                        [0, 2, 2, 2],
                        [0, 0, 0, 2],
                        [10, 0, 0, 2]]))
        ny, nx = a.shape

        F = array([[1, 1, 1, 1],
                   [1, 1j, -1, -1j],
                   [1, -1, 1, -1],
                   [1, -1j, -1, 1j]])
        self.assertArraysClose(ifft2(a), mm.dot(F, a, F)/nx/ny)

class TestCircshift(NumericTestCase):

    def test_single_dim_integer_shift(self):
        a = arange(6)
        shift = 1
        a_s = around(mm.circshift(a, shift))
        self.assertArraysClose(a_s, array([5, 0, 1, 2, 3, 4]))
    
    def test_allones(self):
        N = randint(2, 30)
        shift = 2*N*rand()
        x = ones([N])
        xx = mm.circshift(x, shift)
        self.assertArraysClose(x, xx)

    def test_2d_integer_shift(self):
        shape = array([4, 5])
        x = zeros(shape)
        x[0, 0] = 1
        shift = []
        for max_shift in shape:
            shift.append(randint(max_shift))
        xx = mm.circshift(x, shift)
        i = unravel_index(argmax(xx), x.shape)
        self.assertListEqual(list(shift), list(i))

    def test_multiple_dimensional_integer_shift(self):
        shape = array([2, 2, 3, 2])
        x = zeros(shape)
        x[0, 0, 0, 0] = 1
        shift = []
        for max_shift in shape:
            shift.append(randint(max_shift))
        xx = mm.circshift(x, shift)
        i = unravel_index(argmax(xx), x.shape)
        self.assertListEqual(list(shift), list(i))

class TestMeshgrid(NumericTestCase):
    
    def test_3(self):
        x = arange(2)
        y = arange(3)
        z = arange(4)
        X, Y, Z = mm.meshgridn(x, y, z)
        
        X_correct = array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        ])
        Y_correct = array([
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
        ])
        Z_correct = array([
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        ])
        self.assertArraysClose(X, X_correct)
        self.assertArraysClose(Y, Y_correct)
        self.assertArraysClose(Z, Z_correct)



if __name__ == '__main__':
    unittest.main()
