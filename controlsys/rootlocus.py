from pylab import *

__all__ = ['plotrlocus']

def plotrlocus(gn, gd, hn=None, hd=None, K=None):
    """
    Plot the root locus of a system with the total transfer function

    T = K*G/(1 + K*G*H)

    where G and H are rational transfer functions described by polynomials in
    their numerator and denomenator, or gn and gd for G and optionally hn and
    hd for H.  If H is not specified, it is assumed to be unity.

    The gain range of the plot may be specified by K, but by default goes from
    0 to 100.
    """

    if not K:
        K = linspace(1e-15, 100, 4000)
    if not hn:
        hn = array([1])
    if not hd:
        hd = array([1])

    tn1 = convolve(hd, gd)
    tn2 = convolve(gn, hn)

    tnlen = max(len(tn1), len(tn2))
    tn1 = hstack([zeros(tnlen - len(tn1)), tn1])
    tn2 = hstack([zeros(tnlen - len(tn2)), tn2])

    nK = 10000
    poles = empty([nK, tnlen - 1], dtype="complex128")

    for i, KK in enumerate(linspace(0.1, 100, nK)):
        r = roots(tn1 + KK*tn2)
        poles[i, :] = r

    colors = 'rgbky'
    for p, c in zip(poles.T, colors):
        plot(real(p), imag(p), c + '-')
        plot(real(p[0]), imag(p[0]), c + 'x')
        plot(real(p[-1]), imag(p[-1]), c + 'o')

    grid('on')


if __name__ == "__main__":
    ## Problem 2 diagram e
    gn = array([1])
    gd = array([1, -9, 28, -36, 16])

    plotrlocus(gn, gd)
    show()

    ## Problem 12 a
    gn = [1, 0, 1]
    gd = [1, 4, 1, -6]

    plotrlocus(gn, gd)
    show()

    ## Problem 12 b
    gn = [1, -2, 2]
    gd = [1, 3, 2, 0]
    plotrlocus(gn, gd)
    show()

    ## Midterm 2 Example

    # TODO figure out why this isnt' working
    gn = [1]
    gd = [1, 5, 6]
    hn = [1, -4, 8]
    hd = [1, 2, 5]

    plotrlocus(gn, gd, hn, hd)
    show()


    from sympy import symbols, simplify, expand

    s, K = symbols('s K')
    den = (s + 2)*(s + 3)*(s**2 + 2*s + 5) + K*(s**2 - 4*s + 8)
    expand(den)
    print(den)

