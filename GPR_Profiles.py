import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

# Everything from 10-41 should automate the rest of the script
# You can change the type, type_abrev, and type_title depending on which file you're loading in
# This code was primarily created for use on DIII-D discharge #162940, so some areas are hyper-tuned to work with that
    # discharge
type = "electron_temperature"
type_abrev = "$T_e$"
type_title = "Electron Temperature"
data = np.loadtxt('', delimiter = '\t', skiprows = 2)
psi = data[:,0]
psi_err = data[:,2]
e = data[:,1]
e_err = data[:,3]
# Defining the interval for our testing points
seed = np.random.seed(600)
x_test_lower = 0
x_test_upper = 1
x_test_size = 750
x_test = np.random.uniform(x_test_lower, x_test_upper, x_test_size)
x_test.sort()
sample_test = "random_" + str(x_test_lower) + "_" + str(x_test_upper) + "_" + str(x_test_size)
bins = 50

if type_abrev == "$T_e$":
    sig_1 = 0.06
    sig_2 = 1
    var_1 = 1
    var_2 = 1
    mus = 0
    mu = 0
if type_abrev == "$n_e$":
    sig_1 = 0.05
    sig_2 = 1
    var_1 = 1
    var_2 = 1
    mus = 0
    mu = 0

ny = 20 # Number of posterior fits to plot
domain_x = (0.85, 1.0)

def binavg(x, y, y_err, bins):
    bin_edge = np.linspace(min(x), max(x), bins)
    x_anchor = []
    x_train = [x[0]]
    y_train = [y[0]]
    y_train_err = [y_err[0]]
    for i in range(len(bin_edge)):
        x_avg = []
        y_avg = []
        y_err_avg = []
        for j in range(len(x)):
            if bin_edge[i] < x[j] < bin_edge[i+1]:
                x_avg.append(x[j])
                y_avg.append(y[j])
                y_err_avg.append(y_err[j])
        # Some of the bins being defined do not have any data in them, so those bins are excluded from the sampling
        if len(x_avg) > 0:
            x_train.append(min(x_avg) + (max(x_avg) - min(x_avg)) / 2)
            y_train.append(np.sum(y_avg) / len(y_avg))
            y_train_err.append(np.sum(y_err_avg) / len(y_err_avg))
    for i in range(len(x_train)):
        if 0.78 < x_train[i] < 0.82:
            x_anchor.append(x_train[i])
    x_anchor = min(x_anchor)
    return x_train, y_train, y_train_err, x_anchor

# Sampling our training data for use in the GP
x_train, y_train, y_train_err, x_anchor = binavg(psi, e, e_err, bins)
plt.figure()
plt.plot(psi, e, 'ko', label = "Raw data")
plt.errorbar(psi, e, yerr = e_err, xerr = psi_err, color = "k", fmt = '', ls = 'none')
plt.plot(x_train, y_train, 'ro', label = "Training Data (binned approach)")
plt.errorbar(x_train, y_train, yerr = y_train_err, xerr = None, color = 'r', fmt = '', ls = 'none')
plt.legend()
plt.title("Binned Training Data for " + type_title)
plt.xlabel("$\psi$")
plt.ylabel(type_abrev)

# Squared exponential (RBF) kernel
# Prior variances (var_1 & var_2) are not used in this experiment, but there is a chance they could be useful in the
    # future, and they are a part of the RBF kernel, so they are still included
def sq_exp_kernel(vala, valb, sig_1, sig_2, var_1, var_2):
    kernel = np.zeros(shape = (len(vala), len(valb))) #This is the matrix that our values get saved to
    index_vala = range(len(vala))
    index_valb = range(len(valb))
    for k in index_vala:
        for l in index_valb:
            # The kernel is conditional due to the different densities of the raw data, the core is less
                # dense than the pedestal and the edge, so the same correlation length won't work for both of them
            if x_anchor <= vala[k] <= 1.3 and x_anchor <= valb[l] <= 1.3:
                kernel[k, l] = (var_1**2) * np.exp(-(vala[k]-valb[l])**2 / (2 * sig_1**2))
            else:
                kernel[k, l] = (var_2**2) * np.exp(-(vala[k]-valb[l])**2 / (2 * sig_2**2))
    return kernel

# This function is not used for the purposes of this experiment, but it is still included
def logprob(x, y, err, val1, val2, val3, val4):
    err_sq = []
    for i in err:
        err_sq.append(i**2)
    sig1_val = []
    sig2_val = []
    var1_val = []
    var2_val = []
    loglikelihood = []
    for i in val1:
        for j in val2:
            for k in val3:
                for l in val4:
                    K = sq_exp_kernel(x, x, i, j, k, l) + np.diag(err_sq)
                    log_prob = np.dot(y, scipy.linalg.solve(K, y)) + np.log(abs(np.linalg.det(K))) + len(y) * np.log(2 * np.pi)
                    loglikelihood.append(log_prob)
                    sig1_val.append(i)
                    sig2_val.append(j)
                    var1_val.append(k)
                    var2_val.append(l)
    sig1 = 1
    sig2 = 1
    var1 = 1
    var2 = 1
    for i in np.arange(len(loglikelihood)):
        if loglikelihood[i] == min(loglikelihood):
            sig1 = sig1_val[i]
            sig2 = sig2_val[i]
            var1 = var1_val[i]
            var2 = var2_val[i]
    return sig1, sig2, var1, var2, loglikelihood

# Defining the actual gaussian process
def GP(x1, y1, x2, y_err, sig_1, sig_2, var_1, var_2, kernel_func):
    # The logarithmic probability function is not necessarily for this experiment, but it is still included for
        # potential use

    #length_train_1 = np.arange(0.01, 1, 0.01)  # Changing these values too much to have more of a step or search for
                                                  # more values can lead to poorly conditioned kernels, which makes it
                                                  # difficult for the log probability function to find an
                                                  # appropriate value for our hyperparameters
    #length_train_2 = np.arange(0.01, 1, 0.01)
    #var_train_1 = np.arange(0.01, 1, 0.01)
    #var_train_2 = np.arange(0.01, 1, 0.01)
    #sig_1, sig_2, var_1, var_2, loglikelihood = logprob(x1, y1, y_err, length_train_1, length_train_2, var_train_1, var_train_2)
    #print(sig_1, sig_2, var_1, var_2)

    y_err_sq = []
    for err in y_err:
        y_err_sq.append(err**2)

    # This next part just plots the log probs, it's not entirely necessary

    #plt.figure()
    #ax = plt.axes(projection = '3d')
    #ax.scatter3D(sig_1_vals, sig_2_vals, loglikelihood, label = "log prob")
    #ax.scatter3D(sig_1, sig_2, min(loglikelihood), 'ro', label = "$\sigma_{1|2}$")
    #plt.title("log probability")
    #plt.legend()
    #ax.set_xlabel("sigma 1")
    #ax.set_ylabel("sigma 2")
    #ax.set_zlabel("log likelihood")

    #plt.figure()
    #log_prob_sig_1 = []

    # Calculating these again for 2D plotting purposes, this part can be commented out to save time
    # This does help to visualize the log probability functions to see how well or poorly the hyperparameter values
        # are being calculated, however

    #for l in length_train_1:
    #    K = kernel_func(x1, x1, l, sig_2) + np.diag(y_err_sq) * noise_funcn(x1, x1)
    #    log_prob_sig_1.append(np.dot(y1, scipy.linalg.solve(K, y1)) + np.log(abs(np.linalg.det(K))) + len(y1) * np.log(2 * np.pi))

    #log_prob_sig_2 = []
    #for k in length_train_2:
    #    K = kernel_func(x1, x1, sig_1, k) + np.diag(y_err_sq) * noise_funcn(x1, x1)
    #    log_prob_sig_2.append(np.dot(y1, scipy.linalg.solve(K, y1)) + np.log(abs(np.linalg.det(K))) + len(y1) * np.log(2 * np.pi))

    #plt.plot(length_train_1, log_prob_sig_1, "r", label = "log prob, $\sigma_{1}$")
    #plt.plot(sig_1, min(log_prob_sig_1), 'ro', label = "$\sigma_{1}$")
    #plt.plot(length_train_2, log_prob_sig_2, 'g', label = "log prob, $\sigma_{2}$")
    #plt.plot(sig_2, min(log_prob_sig_2), 'go', label = "$\sigma_{2}$")
    #plt.title("$\sigma_{1}$, $\sigma_{2}$, " + type_title)
    #plt.xlabel("$\sigma_n$")
    #plt.ylabel("log prob")
    #plt.legend()
    #plt.savefig("logprob_" + type + "_core_" + str(sample_core) + "_ped_" + str(sample_ped) + "_edge_" + str(sample_edge)
    #            + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
    #plt.show()

    #print(sig_1)
    var_1 = var_1
    var_2 = var_2
    sig_1 = sig_1
    sig_2 = sig_2
    K = kernel_func(x1, x1, sig_1, sig_2, var_1, var_2) + np.diag(y_err_sq) #kernel of observations
    K_s = kernel_func(x1, x2, sig_1, sig_2, var_1, var_2)
    K_0s = kernel_func(x2, x1, sig_1, sig_2, var_1, var_2)
    K_ss = kernel_func(x2, x2, sig_1, sig_2, var_1, var_2)

    mu_post = np.dot(K_0s, np.dot(scipy.linalg.inv(K), y1)) # Posterior Mean
                                                            # There are two additional terms, the mean of our testing points,
                                                                # which is added to the whole term, and the mean of our
                                                                # observations, which is subtracted from y1, that are
                                                                # present in the original function, but they are
                                                                # considered to be 0 for this experiment

    K_post = K_ss - np.dot(K_0s, np.dot(scipy.linalg.inv(K), K_s)) # Posterior Covariance Matrix
    return mu_post, K_post

# Defines the derivative operator for use in correcting the jump that occurs at the anchor point
def D(x, y):
    D1 = np.zeros(shape=(len(x), len(y)))
    i_val = range(len(x))
    j_val = range(len(y))
    for i in i_val:
        for j in j_val:
            if i == 0 and j == 0:
                D1[i, j] = 1
                # These are the values to define the proper right-hand side derivative, although they make the matrix
                    # uninvertible, which is necessary for this experiment
                #D[i, j] = -1 / (x[i+1]-x[i])
                #D[i, j+1] = 1 / (x[i+1]-x[i])
            if i == max(i_val) and j == max(j_val):
                D1[i, j] = 1 / (x[i] - x[i - 1])  # normally +1
                D1[i, j - 1] = -1 / (x[i] - x[i - 1])  # normally -1
            if 0 < i < max(i_val) and 0 < j < max(j_val) and i == j:
                D1[i, j] = 0  # normally 0
                D1[i, j - 1] = -1 / (2 * (x[i + 1] - x[i - 1]))  # normally -1
                D1[i, j + 1] = 1 / (2 * (x[i + 1] - x[i - 1]))  # normally +1
    # This portion scans for the jump in the derivative that occurs due to the conditional kernel in order to correct it
    # The primary issue with this is that it is hyper-tuned specifically to this experiment, which takes away some of
        # the automation that we want out of a GPR model. Methods to ensure the automation of this section are still
        # being investigated
    D_sol = np.dot(D1, y)
    D_sol_fix = []
    x_fix = []
    if type_abrev == "$T_e$":
        for i in range(len(x)):
            if x_anchor - 0.02 < x[i] < x_anchor + 0.015:
                x_fix.append(i)
                if D_sol[i] < -0.14 or D_sol[i] > -0.14:
                    D_sol_fix.append(D_sol[x_fix[0]])
                else:
                    D_sol_fix.append(D_sol[i])
            else:
                D_sol_fix.append(D_sol[i])
    if type_abrev == "$n_e$":
        for i in range(len(x)):
            if x_anchor - 0.02 < x[i] <= 0.92:
                x_fix.append(i)
                if D_sol[i] > -0.2 or D_sol[i] < -0.2:
                    D_sol_fix.append(-0.2)
                else:
                    D_sol_fix.append(D_sol[i])
            else:
                D_sol_fix.append(D_sol[i])
    return D1, D_sol, D_sol_fix

# x is sample points, y is interpolation points
# A defines a matrix based on lagrange-basis interpolation. This method could be useful for discretizing the GP so it
    # can be smoothed out by a derivative operator such as the one lined out above. This has yet to be applied in this
    # manner yet, and will require further testing.
def A(x, y):
    A1 = np.zeros(shape = (len(x), len(y)))
    i_val = range(len(x))
    j_val = range(len(y))
    for i in i_val:
        for j in j_val:
            if i == 0 and j == 0:
                A1[i,j] = ((x[i+1]-y[j])/(x[i+1]-x[i]))
            if i == max(i_val) and j == max(j_val):
                A1[i, j] = ((y[j]-x[i-1])/(x[i]-x[i-1]))
            if i > min(i_val) and j > min(j_val) and x[i-1] <= y[j] <= x[i]:
                A1[i,j] = ((y[j]-x[i-1])/(x[i]-x[i-1]))
            if i < max(i_val) and j < max(j_val) and x[i] <= y[j] <= x[i+1]:
                A1[i,j] = ((x[i+1]-y[j])/(x[i+1]-x[i]))
    return A1

mu_post, K_post = GP(x_train, y_train, x_test, y_train_err, sig_1, sig_2, var_1, var_2, sq_exp_kernel)
y_test = np.random.multivariate_normal(mean = mu_post, cov = K_post, size = ny)
sigma = np.sqrt(np.diag(K_post))

# Plotting figure + posterior samples + mean
plt.figure(figsize = (12, 8))

for q in range(ny):
    x_array = x_test
    y_array = y_test[q].T
    save_data = np.column_stack([x_array, y_array])
    # Commented out so the script doesn't save 20 files each time it runs
    # Also, these would save the uncorrected values, when we want to save the values from the corrected GP

    #np.savetxt("posterior_sample_" + str(q+1) + "_" + type + "_bins_" + str(bins) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + ".txt",
    #           save_data, fmt = ("%.9e", "%.9e"), delimiter = "\t", header = "Column 1: psi inference; Column 2: " + type_title + " inference")

    plt.plot(x_test, y_test[q].T, '-', alpha = 0.3, color = "g", label = "Posterior Sample" if q == 0 else None)

plt.title("Posterior Samples to " + type_title +" Profile", fontsize = 'xx-large')
plt.plot(psi, e, 'ko', label = type_abrev)
plt.errorbar(psi, e, yerr = e_err, xerr = psi_err, color = "k", fmt = '', ls = 'none')
plt.plot(x_train, y_train, 'ro', label = type_abrev + " Training")
plt.errorbar(x_train, y_train, yerr = y_train_err, xerr = None, color = 'r', fmt = '', ls = 'none')
plt.fill_between(x_test, mu_post-2*sigma, mu_post+2*sigma, color = "blue", alpha = 0.15, label = "2$\sigma$")
plt.plot(x_test, mu_post, 'b-', label = "$\mu$")
plt.axvline(x = x_anchor, color = 'r', linestyle = '--', label = "Kernel separation point")
plt.legend(fontsize = 'large')
plt.xlabel("$\psi$", fontsize = 'xx-large')
if type_abrev == "$T_e$":
    plt.ylabel(type_abrev + "$ (keV)$", fontsize = 'xx-large')
if type_abrev == "$n_e$":
    plt.ylabel(type_abrev + "$ (10^{20}/m^3)$", fontsize = 'xx-large')
#if domain_x == (0, 1.3):
#    plt.savefig("whole_posterior_fit_to_" + type + "_bins_" + str(bins) + "_ped_" + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
#if domain_x == (0, 0.85):
#    plt.savefig("core_posterior_fit_to_" + type + "_bins_" + str(bins) + "_ped_" + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
#if domain_x == (0.85, 1.0):
#    plt.savefig("ped_posterior_fit_to_" + type + "_bins_" + str(bins) + "_ped_" + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
#if domain_x == (1.0, 1.3):
#    plt.savefig("edge_posterior_fit_to_" + type + "_bins_" + str(bins) + "_ped_" + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")

Dmu2, D_sol, D_fix = D(x_test, mu_post)
mu2_corr = np.dot(scipy.linalg.inv(Dmu2), D_fix)
plt.figure(figsize = (12, 8))
plt.xlim(x_anchor-0.05, 1)
plt.ylim(-15, 2)
plt.title("Gradient of " + type_title)
plt.ylabel("d" + type_abrev + "$/d\psi$")
plt.xlabel("$\psi$")
plt.plot(x_test, D_fix, 'r', label = "Corrected Derivative")
plt.plot(x_test, D_sol, 'b', label = "Original Derivative")
plt.legend()
plt.savefig(type + "_Derivatives_Compared.png")

Csig_1 = 0.05
Csig_2 = 0.05
Cvar_1 = 1
Cvar_2 = 1
bins = 50
x_test_2 = np.linspace(0, 1, 2000)
Cx1, Cy1, Cyerr, x_anchor = binavg(x_test, mu2_corr, sigma, bins)
new_sigma = []
for i in Cyerr:
    new_sigma.append(abs(i))
Cmu2, CK2 = GP(Cx1, Cy1, x_test_2, new_sigma, Csig_1, Csig_2, Cvar_1, Cvar_2, sq_exp_kernel)
Cy_test = np.random.multivariate_normal(mean = Cmu2, cov = CK2, size = ny)
Csigma = np.sqrt(np.diag(CK2))
save_data_3 = np.column_stack([x_test_2, Cmu2])
np.savetxt("corrected_mean_posterior_sample_" + type + "_bins_" + str(bins) + "_sig_1_" + str(Csig_1)[0:4] + "_sig_2_" + str(Csig_2)[0:4] + ".txt",
               save_data_3, fmt = ("%.9e", "%.9e"), delimiter = "\t", header = "Column 1: psi inference; Column 2: " + type_title + " inference")

plt.figure(figsize = (12, 8))
for i in range(ny):
    x_array = x_test_2
    y_array = Cy_test[i].T
    save_data_4 = np.column_stack([x_array, y_array])
    # Commented out so the script doesn't save 20 files each time it runs
    #np.savetxt("posterior_sample_" + str(i+1) + "_" + type + "_bins_" + str(bins) + "_sig_1_" + str(Csig_1) + "_sig_2_" + str(Csig_2) + ".txt",
    #           save_data_4, fmt = ("%.9e", "%.9e"), delimiter = "\t", header = "Column 1: psi inference; Column 2: " + type_title + " inference")
    plt.plot(x_test_2, Cy_test[i].T, '-', alpha=0.3, color="g", label="Posterior Sample" if i == 0 else None)
plt.title("Posterior Samples to " + type_title +" Profile", fontsize = 'xx-large')
plt.plot(psi, e, 'ko', label = type_abrev)
plt.errorbar(psi, e, yerr = e_err, xerr = psi_err, color = "k", fmt = '', ls = 'none')
plt.plot(x_train, y_train, 'ro', label = type_abrev + " Training")
plt.errorbar(x_train, y_train, yerr = y_train_err, xerr = None, color = 'r', fmt = '', ls = 'none')
plt.fill_between(x_test_2, Cmu2-2*Csigma, Cmu2+2*Csigma, color = "blue", alpha = 0.15, label = "2$\sigma$")
plt.plot(x_test_2, Cmu2, 'b-', label = "Corrected mu2 w/ GP")#$D^{-1}D\mu$")
plt.xlim(domain_x[0], domain_x[1])
if type_abrev == "$T_e$":
    plt.ylim(0, 1)
    plt.ylabel(type_abrev + "$ (keV)$", fontsize = 'xx-large')
if type_abrev == "$n_e$":
    plt.ylim(0, 0.8)
    plt.ylabel(type_abrev + "$ (10^{20}/m^3)$", fontsize = 'xx-large')
plt.legend(fontsize = 'large')
plt.xlabel("$\psi$", fontsize = 'xx-large')
plt.savefig("corrected_ped_posterior_fit_to_" + type + "_bins_" + str(bins) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")

# Plotting Lagrange-basis approximation
A1 = A(x_train, x_test)
y_noise = []
for i in y_train_err:
    y_noise.append(i**2)
K_ss = sq_exp_kernel(x_test, x_test, 0.06, 0.06, 1, 1)
Aterm1 = np.dot(K_ss, A1.T)
Aterm2 = np.dot(A1, np.dot(K_ss, A1.T)) + np.diag(y_noise)
Aterm3 = np.dot(A1, K_ss)
Amu = np.dot(Aterm1, np.dot(scipy.linalg.inv(Aterm2), y_train))
AK2 = K_ss - np.dot(Aterm1, np.dot(scipy.linalg.inv(Aterm2), Aterm3))
F_test = np.random.multivariate_normal(mean = Amu, cov = AK2, size = ny)
Asigma = np.sqrt(np.diag(AK2))

plt.figure(figsize=(12,8))
for i in range(ny):
    plt.plot(x_train, np.dot(A1, F_test[i]), alpha = 0.3, color = 'g', label = "Lagrange basis Posterior sample" if i == 0 else None)
plt.plot(psi, e, 'ko', label = type_abrev + " Data")
plt.errorbar(psi, e, yerr = e_err, xerr = psi_err, color = "k", fmt = '', ls = 'none')
plt.plot(x_train, y_train, 'ro', label = type_abrev + " Training")
plt.errorbar(x_train, y_train, yerr = y_train_err, xerr = None, color = 'r', fmt = '', ls = 'none')
plt.plot(x_test_2, Cmu2, 'b-', label = "Corrected mu2 w/ GP")
plt.plot(x_train, np.dot(A1, Amu), 'r', label = "Lagrange Approximation")
plt.xlim(0.85, 1)
plt.title("Lagrange Basis Approximation to " + type_title + " Training Points", fontsize = 'xx-large')
if type_abrev == "$T_e$":
    plt.ylim(0, 1)
    plt.ylabel(type_abrev + "$ (keV)$", fontsize = 'xx-large')
if type_abrev == "$n_e$":
    plt.ylim(0, 0.8)
    plt.ylabel(type_abrev + "$ (10^{20}/m^3)$", fontsize = 'xx-large')
plt.legend()
plt.savefig("Lagrange_Basis_to_" + str(type) + "_Data.png")
print("Done")
plt.show()
