from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.pyplot as plt
style.use('fivethirtyeight')
xs=np.array([1,2,3,4,5,6], dtype=np.float64)
ys=np.array([5,4,6,5,6,7], dtype=np.float64)
#best-slope
def best_fit_slope(xs,ys):
    m=( (mean(xs)*mean(ys)) - mean(xs*ys) ) / ((mean(xs)*mean(xs))-mean(xs*xs))
    return m

m=best_fit_slope(xs,ys) 
print('Best Fit Slope:',m)
# best intercept
def best_fit_intercept(xs,ys,m):
    b= (mean(ys)- (m*mean(xs)))
    return b

b=best_fit_intercept(xs,ys,m)
print('Best Fit Intercept:',b)

regression_line=[(m*x)+b for x in xs]

#Defining Squared Error
def squared_error(ys_line, ys_orig):
    return sum((ys_line-ys_orig)**2)

#Defining coefficient of determination
def coeeficient_of_determination(ys_line, ys_orig):
    y_mean_line=[mean(ys_orig) for y in ys_orig]
    squared_error_reg=squared_error(ys_line,ys_orig)
    squared_error_mean=squared_error(y_mean_line,ys_orig)
    return 1-(squared_error_reg/squared_error_mean)
r=coeeficient_of_determination(regression_line, ys)
print('Coeeficient of Determination:', r)
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()