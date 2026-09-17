import numpy as np 
import matplotlib.pyplot as plt

x = [0.0003345935229143244, 0.0007208598928265988, 0.0015530455597582893, 0.0033459352291432443, 0.007208598928265988]

y = [111.20941405247271, 71.60994453798679, 31.517941564040314, 1.8606448722271693, 235.45449917362942 ]

plt.plot(y, x, 'o-')

y_value = 2.71
x_value =  np.interp(y_value, y, x)
print (f"Interpolated x for y={y_value}: {x_value}")
plt.plot(y_value, x_value, 'r*')  # Plot the interpolated point in

plt.show()