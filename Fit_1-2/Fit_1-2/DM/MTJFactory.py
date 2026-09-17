# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import html
import numpy as np
import astropy.units as u

__all__ = ["MTJFactory"]


# MT: Copied from original source with some modifications
# added few functions to integrate along los s instead of radius r used by gammapy code
# The two methods should agree
class MTJFactory:
    """Compute J-Factor or D-Factor maps.

    J-Factors are computed for annihilation and D-Factors for decay.
    Set the argument `annihilation` to `False` to compute D-Factors.
    The assumed dark matter profiles will be centered on the center of the map.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry.
    profile : `~gammapy.astro.darkmatter.profiles.DMProfile`
        Dark matter profile.
    distance : `~astropy.units.Quantity`
        Distance to convert angular scale of the map.
    annihilation: bool, optional
        Decay or annihilation. Default is True.
    """

    def __init__(self, geom, profile, distance, annihilation=True):
        self.geom = geom
        self.profile = profile
        self.distance = distance
        self.annihilation = annihilation

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def MTcompute_differential_jfactor_separation(self, separation, ndecade=1e4):
        rmin = u.Quantity(
            value=np.tan(separation) * self.distance, unit=self.distance.unit
        )
        rmax = self.distance
        #print(separation)
        val = [
            (
                2
                * self.profile.integral(
                    _.value * u.kpc,
                    rmax,
                    np.arctan(_.value / self.distance.value),
                    ndecade,
                    self.annihilation,
                )
                + self.profile.integral(
                    self.distance,
                    4 * rmax,
                    np.arctan(_.value / self.distance.value),
                    ndecade,
                    self.annihilation,
                )
            )
            for _ in rmin.ravel()
        ]
        integral_unit = u.Unit("GeV2 cm-5") if self.annihilation else u.Unit("GeV cm-2")
        jfact = u.Quantity(val).to(integral_unit).reshape(rmin.shape)
        # MT no units shown now!
        return jfact.value

    
    def MTcompute_differential_jfactor(self, ntheta=10, ndecade=1e4):
        r"""Compute differential J-Factor.

        .. math::
            \frac{\mathrm d J_\text{ann}}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d l \rho(l)^2

        .. math::
            \frac{\mathrm d J_\text{decay}}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d l \rho(l)
        """
        # MT: First compute in a table for various angles and then interpolate
        separation = self.geom.separation(self.geom.center_skydir).rad
        #print(separation)
        sepmin=np.amin(separation)
        sepmax=np.amax(separation)
        print("thetamin deg ",sepmin*180/np.pi," thetamax ",sepmax*180/np.pi)
        x = np.logspace(np.log10(sepmin), np.log10(sepmax), ntheta)
        y = np.log10(self.MTcompute_differential_jfactor_separation(x,ndecade))
        val =  np.interp(separation, x, y)
        val = 10**val
        #print(x)
        #print(y)
        #print(val)
        return val

    def MTcompute_jfactor(self, ntheta=10, ndecade=1e4):
        r"""Compute astrophysical J-Factor.

        .. math::
            J(\Delta\Omega) =
           \int_{\Delta\Omega} \mathrm d \Omega^{\prime}
           \frac{\mathrm d J}{\mathrm d \Omega^{\prime}}
        """
        #print(self.geom.to_image().solid_angle())
        diff_jfact = self.MTcompute_differential_jfactor(ntheta, ndecade)
        return diff_jfact * self.geom.to_image().solid_angle().value

    # To be improved: manually put reasonable values for integration
    # compute integration along los for array of separation
    def MTcompute_differential_jfactor_separation_los(self, separation, ndecade=1e4):
        smin = u.Quantity(
            value=0.1, unit=self.distance.unit
        )
        smax = 2*self.distance
        s1 = self.distance - 0.5* u.kpc
        s2 = self.distance + 0.5* u.kpc
        #print(smin)
        #print(smax)
        #print(separation)
        val = [
            (
                self.profile.integral_los(
                    smin,
                    s1,
                    _,
                    ndecade,
                    self.annihilation,
                )
                + self.profile.integral_los(
                    s1,
                    s2,
                    _,
                    ndecade,
                    self.annihilation,
                )
                + self.profile.integral_los(
                    s2,
                    smax,
                    _,
                    ndecade,
                    self.annihilation,
                )
            )
            for _ in separation.ravel()
        ]
        integral_unit = u.Unit("GeV2 cm-5") if self.annihilation else u.Unit("GeV cm-2")
        jfact = u.Quantity(val).to(integral_unit).reshape(separation.shape)
        # MT no units shown now!
        return jfact.value
    
    #MT compute jfactors along los s for all pixels in geom
    def MTcompute_differential_jfactor_los(self, ntheta=10, ndecade=1e4):
        r"""Compute differential J-Factor.

        .. math::
            \frac{\mathrm d J_\text{ann}}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d l \rho(l)^2

        .. math::
            \frac{\mathrm d J_\text{decay}}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d l \rho(l)
        """
        # MT: First compute in a table for various angles and then interpolate
        separation = self.geom.separation(self.geom.center_skydir).rad
        #print(separation)
        sepmin=np.amin(separation)
        sepmax=np.amax(separation)
        print("thetamin deg ",sepmin*180/np.pi," thetamax ",sepmax*180/np.pi)
        x = np.logspace(np.log10(sepmin), np.log10(sepmax), ntheta)
        y = np.log10(self.MTcompute_differential_jfactor_separation_los(x,ndecade))
        val =  np.interp(separation, x, y)
        val = 10**val
        #print(x)
        #print(y)
        #print(val)
        return val


    #MT compute D factor map, i.e. D integrated in the pixel for all pixels in geom using integration along los s
    def MTcompute_jfactor_los(self, ntheta=10, ndecade=1e4):
        r"""Compute astrophysical J-Factor.

        .. math::
            J(\Delta\Omega) =
           \int_{\Delta\Omega} \mathrm d \Omega^{\prime}
           \frac{\mathrm d J}{\mathrm d \Omega^{\prime}}
        """
        #print(self.geom.to_image().solid_angle())
        diff_jfact = self.MTcompute_differential_jfactor_los(ntheta, ndecade)
        return diff_jfact * self.geom.to_image().solid_angle().value
