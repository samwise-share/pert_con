import numpy as np
import scipy.optimize as optimize
from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd

#  https://www.codeproject.com/Articles/56371/Finding-Probability-Distribution-Parameters-from-P

def pert_parameters(x1, p1, x2, p2, x3, p3):
    """Find parameters for a pert random variable X
    so that P(X > x1) = p1 and P(X > x2) = p2, and P(X > x3) = p3."""

    def square(x):
        return x*x

    def objective(v):
        (a, b, c) = v
        loc = a
        scale = c-a
        A = (4*b + c - 5*a) / (c - a)
        B = (5*c - a - 4*b) / (c - a)
        # add errors from all three CDF points using current estimate of a, b, c
        temp = square(beta.cdf(x1, A, B, loc, scale) - p1)
        temp += square(beta.cdf(x2, A, B, loc, scale) - p2)
        temp += square(beta.cdf(x3, A, B, loc, scale) - p3)
        return temp

    # arbitrary initial guess of (3, 3, 3) for parameters
    xopt = optimize.fmin(objective, (x1, x2, x3), xtol=1e-6, ftol=1e-6)
    return (xopt[0], xopt[1], xopt[2])


df = pd.read_excel('excel_input_file.xlsx', usecols=[0, 1, 2])
dataarray = df.to_numpy()


# %%
a_list = []
b_list = []
c_list = []
nrows = np.shape(dataarray)[0]
for index in range(nrows):
    x1, x2, x3 = dataarray[index]
    (a, b, c) = pert_parameters(x1, 0.1, x2, 0.5, x3, 0.9)
    a_list.append(a)
    b_list.append(b)
    c_list.append(c)
df['min'] = a_list
df['mode'] = b_list
df['max'] = c_list

write_pdf_points = True
if write_pdf_points:
    npoints = 10
    pdf_points = np.zeros((nrows,npoints))
    
    for index in range(nrows):
        a = a_list[index]
        b = b_list[index]
        c = c_list[index]
        loc = a
        scale = c-a
        A = (4*b + c- 5*a) / (c - a)
        B = (5*c - a - 4*b) / (c - a)
        x = np.linspace(beta.ppf(0, A, B, loc, scale),
                        beta.ppf(1, A, B, loc, scale), npoints)
        pdf_points[index, :] = beta.pdf(x, A, B, loc, scale)
    for col in range(npoints):
        df[f'point{col}'] = list(pdf_points[:, col])

print(df)
df.to_excel("excel_output_file.xlsx", index=False)    
