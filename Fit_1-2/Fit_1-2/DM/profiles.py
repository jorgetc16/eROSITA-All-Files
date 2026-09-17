# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter profiles."""
import abc
import html
import numpy as np
import astropy.units as u
from gammapy.modeling import Parameter, Parameters
from gammapy.utils.integrate import trapz_loglog

__all__ = [
    "BurkertProfile",
    "DMProfile",
    "EinastoProfile",
    "IsothermalProfile",
    "MooreProfile",
    "NFWProfile",
    "ZhaoProfile",
]

#MT:All code taken from gammapy. Added few functions.
class DMProfile(abc.ABC):
    """DMProfile model base class."""

    LOCAL_DENSITY = 0.3 * u.GeV / (u.cm**3)
    """Local dark matter density as given in reference 2"""
    DISTANCE_GC = 8.33 * u.kpc
    """Distance to the Galactic Center as given in reference 2"""


    def __call__(self, radius):
        """Call evaluate method of derived classes."""
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate(radius, **kwargs)



    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"


    def scale_to_local_density(self):
        """Scale to local density."""
        scale = (self.LOCAL_DENSITY / self(self.DISTANCE_GC)).to_value("")
        self.parameters["rho_s"].value *= scale



    def _eval_substitution(self, radius, separation, squared):
        """Density at given radius together with the substitution part."""
        exponent = 2 if squared else 1
        return (
            self(radius) ** exponent
            * radius
            / np.sqrt(radius**2 - (self.DISTANCE_GC * np.sin(separation)) ** 2)
        )


    def integral(self, rmin, rmax, separation, ndecade, squared=True):
        r"""Integrate dark matter profile numerically.

        .. math::
            F(r_{min}, r_{max}) = \int_{r_{min}}^{r_{max}}\rho(r)^\gamma dr \\
            \gamma = 2 \text{for annihilation} \\
            \gamma = 1 \text{for decay}

        Parameters
        ----------
        rmin, rmax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        separation : `~numpy.ndarray`
            Separation angle in radians.
        ndecade : int, optional
            Number of grid points per decade used for the integration.
            Default is 10000.
        squared : bool, optional
            Square the profile before integration.
            Default is True.
        """
        integral = self.integrate_spectrum_separation(
            self._eval_substitution, rmin, rmax, separation, ndecade, squared
        )
        inegral_unit = u.Unit("GeV2 cm-5") if squared else u.Unit("GeV cm-2")
        return integral.to(inegral_unit)


    def integrate_spectrum_separation(
        self, func, xmin, xmax, separation, ndecade, squared=True
    ):
        """Squared dark matter profile integral.

        Parameters
        ----------
        xmin, xmax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        separation : `~numpy.ndarray`
            Separation angle in radians.
        ndecade : int
            Number of grid points per decade used for the integration.
        squared : bool
            Square the profile before integration.
            Default is True.
        """
        unit = xmin.unit
        xmin = xmin.value
        xmax = xmax.to_value(unit)
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = np.int32((logmax - logmin) * ndecade)
        x = np.logspace(logmin, logmax, n) * unit
        y = func(x, separation, squared)
        val = trapz_loglog(y, x)
        return val.sum()
    
    #MT: substitution for integration along s,theta
    def _eval_substitution_los(self, slos, separation, squared):
        """Density at given radius computed from slos and separation"""
        exponent = 2 if squared else 1
        radius=np.sqrt(self.DISTANCE_GC**2 + slos**2 -2*self.DISTANCE_GC*slos*np.cos(separation)  )
        # for small angles
        #radius=np.sqrt((self.DISTANCE_GC - slos)**2 + self.DISTANCE_GC*slos*separation*separation   )
        return (
            self(radius) ** exponent
        )

    #MT integration along los s for a given theta (separation in rad)
    def integral_los(self, smin, smax, separation, ndecade, squared=True):
        r"""Integrate dark matter profile numerically.

        .. math::
            F(r_{min}, r_{max}) = \int_{r_{min}}^{r_{max}}\rho(r)^\gamma dr \\
            \gamma = 2 \text{for annihilation} \\
            \gamma = 1 \text{for decay}

        Parameters
        ----------
        rmin, rmax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        separation : `~numpy.ndarray`
            Separation angle in radians.
        ndecade : int, optional
            Number of grid points per decade used for the integration.
            Default is 10000.
        squared : bool, optional
            Square the profile before integration.
            Default is True.
        """
        integral = self.integrate_spectrum_separation(
            self._eval_substitution_los, smin, smax, separation, ndecade, squared
        )
        inegral_unit = u.Unit("GeV2 cm-5") if squared else u.Unit("GeV cm-2")
        return integral.to(inegral_unit)



class NFWProfile(DMProfile):
    r"""NFW Profile.

    .. math::
        \rho(r) = \rho_s \frac{r_s}{r}\left(1 + \frac{r}{r_s}\right)^{-2}

    Parameters
    ----------
    r_s : `~astropy.units.Quantity`
        Scale radius, :math:`r_s`.
    rho_s : `~astropy.units.Quantity`
        Characteristic density, :math:`\rho_s`.

    References
    ----------
    * `1997ApJ...490..493 <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N>`_
    * `2011JCAP...03..051C <https://ui.adsabs.harvard.edu/abs/2011JCAP...03..051C>`_
    """

    DEFAULT_SCALE_RADIUS = 24.42 * u.kpc
    """Default scale radius as given in reference 2"""

    def __init__(self, r_s=None, rho_s=1 * u.Unit("GeV / cm3")):
        r_s = self.DEFAULT_SCALE_RADIUS if r_s is None else r_s
        self.parameters = Parameters(
            [Parameter("r_s", u.Quantity(r_s)), Parameter("rho_s", u.Quantity(rho_s))]
        )


    @staticmethod
    def evaluate(radius, r_s, rho_s):
        """Evaluate the profile."""
        rr = radius / r_s
        return rho_s / (rr * (1 + rr) ** 2)

