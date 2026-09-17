import warnings

warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all="ignore")

from threeML import *



# Let's generate some data with y = Powerlaw(x)

gen_function = Powerlaw()


# Generate a dataset using the power law, and a
# constant 30% error

x = np.logspace(0, 2, 50)

xyl_generator = XYLike.from_function(
    "sim_data", function=gen_function, x=x, yerr=0.3 * gen_function(x)
)

y = xyl_generator.y
y_err = xyl_generator.yerr

fit_function = Powerlaw()

xyl = XYLike("data", x, y, y_err)

results = xyl.fit(fit_function)

fig = xyl.plot(x_scale="log", y_scale="log")

plt.show()

gof, all_results, all_like_values = xyl.goodness_of_fit()

print("The null-hypothesis probability from simulations is %.2f" % gof["data"])