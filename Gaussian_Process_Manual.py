import numpy as np
import matplotlib.pyplot as plt
import scipy

x = np.arange(0, 2, 0.02)
y = np.sin(2*np.pi*x)

#sig_f is standard deviation off mean of function
#sig_l is correlation length
def noise_kernel(val1, val2, sig_f = 1):
    delta = np.zeros(shape = (len(val1), len(val2)))
    index_val1 = np.arange(len(val1))
    index_val2 = np.arange(len(val2))
    #To my understanding, "delta" is supposed to be an identity matrix? Everything I've read says that when i == j, it's supposed to be 1
    #I haven't had much use for it yet, so I haven't focused too much on it just yet, but if this is incorrect I'll correct it
    for i in index_val1:
        for j in index_val2:
            if i == j :
                delta[i, j] = 1
            else:
                delta[i, j] = 0
    return sig_f**2 * delta

#Squared exponentiated kernel
def sq_exp_kernel(vala, valb, sig_f1 = 1, sig_l1 = 1):
    kernel = np.zeros(shape = (len(vala), len(valb))) #This is just the matrix that our values get saved to
    index_vala = np.arange(len(vala)) #these lists are necessary so each of the values of kernel[vala, valb] can be called and calculated properly
    index_valb = np.arange(len(valb))
    for k in index_vala:
        for l in index_valb:
            kernel[k, l] = np.exp(-(vala[k]-valb[l])**2 / (2 * sig_l1**2)) #This function seems to bring up a deprecation for the one of the functions in numpy
                                                                           #It'll need to be fixed eventually, but right now I'm just trying to get the code working
    return sig_f1**2 * kernel

#Defining the actual gaussian process
def GP(x1, y1, x2, kernel_func):
    K = kernel_func(x1, x1) #kernel of observations
    #print(K.shape)
    K_s = kernel_func(x1, x2) #
    #print(K_s.shape)
    solvit = scipy.linalg.solve(K, K_s, assume_a = 'pos').T #scipy user guide says this solves a * x = b, so I guess it returns a matrix x?
                                                            #Tried using K @ K_s, but I keep getting a matmul error that says:
                                                            #"Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)"
    #print(solvit.shape)
    mu2 = solvit @ y1 #mean value
    #print(mu2.shape)
    K_ss = kernel_func(x2, x2)
    #print(K_ss.shape)
    K_2 = K_ss - (solvit @ K_s) #This is supposed to be our covariance matrix
    #print(K_2.shape)
    return mu2, K_2

#Sampling random data using x values listed above
#num_samples = 100
num_functions = 5
#x_test = np.expand_dims(np.linspace(-4, 4, num_samples), 1) #This was here for testing purposes to see if my results vaguely looked like those
                                                             #in the example
#epsilon = sq_exp_kernel(x, x)
#print(epsilon)
#y_epsilon = np.random.multivariate_normal(mean = np.zeros(num_samples), cov = epsilon, size = num_functions)
#print(y_epsilon)
#plt.figure(figsize = (12, 8))
#for p in range(num_functions):
#    plt.plot(x, y_epsilon[p], linestyle = "-", label = ["GPR function", p], marker = "o", markersize = 3)
#plt.title("Random GPR Fit using Domain [0,2] and Squared Exponentiated Kernel")

#Generating training/testing data within domain (0, 2) like in the sine function in line 6
x1 = np.random.uniform(0, 2, size = (10, 1)) #Training points; If domain is restricted too much or number of training points is increased too much,
                                             #the chance of getting a singular matrix is increased, which makes scipy.linalg.solve in the GP
                                             #above unable to run. Not sure why the matrix ends up being singular, it doesn't appear to be a problem in the
                                             #example I'm following?
y1 = np.sin(2*np.pi*x1).flatten() #function connecting training points
x2 = np.linspace(-0.5, 2.5, 100).reshape(-1, 1) #testing points
mu2, K_2 = GP(x1, y1, x2, sq_exp_kernel)
y2 = np.random.multivariate_normal(mean = mu2, cov = K_2, size = num_functions)
sigma = np.sqrt(np.diag(K_2)) #This has a very strong chance of returning a lot of nans. Also something I need to investigate

#Plotting the function with the mean; more often than not the mean doesn't show up because of all the nan values
plt.figure(figsize = (12, 8))
plt.title("True Sine Function vs. GPR Fit")
plt.plot(x, y, color = "r", linestyle = "--", label = "True Sine; y = sin(x)")
plt.fill_between(x2.flat, mu2-2*sigma, mu2+2*sigma, color = "blue", alpha = 0.15, label = "2 sigma_12")
plt.plot(x2, mu2, label = "mean")
plt.plot(x1, y1, 'ko', label = "x1, y1")
plt.legend()
plt.xlabel("x")
plt.ylabel("y = f(x)")
plt.xlim(0, 2)
plt.ylim(-1.5, 1.5)

#I think the issue here is that the plots fit TOO perfectly over one another, you can't differentiate them, which tells me something is wrong elsewhere
plt.figure(figsize = (12, 8))
for q in range(num_functions):
    plt.plot(x2, y2[q].T, '-', label = ("GPR", q))
plt.title("GPR Fit to Sin(x) Function")
plt.legend()
plt.xlim(0, 2)
plt.ylim(-1.5, 1.5)
plt.show()


