import unittest

from pylab import *
import numpy as np

__all__ = [NumericTestCase]


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
