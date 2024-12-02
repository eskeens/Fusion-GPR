#This script was developed for use with the DIII-D discharge #162940, specifically to fit to the electron temperature
# and density profiles. As of 12/2/2024, it has not been tested for use with other profile types yet
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy


#Everything from 9-30 should automate the rest of the script in a few places
#You can change the type, type_abrev, and type_title depending on which file you're loading in and it should take care
# of the naming of graphs and files without you having to manually change multiple lines later on
type = "" #Fill in the name of the loaded in data here; ex. electron_density, electron_temp, used for file saving
type_abrev = "$$" #Fill in the abbreviation of your loaded data here, used for axis labeling
type_title = "" #Fill in the full name of the loaded data here, used for graph titling
data = np.loadtxt('', delimiter = '\t', skiprows = 2) #Load in whatever data you need here, and you can modify other
                                                      # portions of the script as needed
psi = data[:,0]
psi_err = data[:,2]
te = data[:,1]
te_err = data[:,3]
#Defining our test (posterior) sample
seed = np.random.seed(600)
x_test_lower = 0
x_test_upper = 1
x_test_size = 2000
x_test = np.random.uniform(x_test_lower, x_test_upper, x_test_size)
x_test.sort()
sample_test = "random_" + str(x_test_lower) + "_" + str(x_test_upper) + "_" + str(x_test_size)

ny = 20 #Number of posterior fits to plot
domain_x = (0.85, 1.0)
x_train = []
y_train = []
y_train_err = []


#The edge CAN be sampled differently than the pedestal and core if you choose
sample_core = 5
sample_ped = 20
sample_edge = 20

for t in np.arange(len(psi)):
    if 0 < psi[t] < 0.85:
        if t % sample_core == 0:
            x_train.append(psi[t])
            y_train.append(te[t])
            y_train_err.append(te_err[t])
    if 0.85 < psi[t] < 1.0:
        if t % sample_ped == 0:
            x_train.append(psi[t])
            y_train.append(te[t])
            y_train_err.append(te_err[t])
    if 1.0 < psi[t] < 1.3:
        if t % sample_edge == 0:
            x_train.append(psi[t])
            y_train.append(te[t])
            y_train_err.append(te_err[t])
#print(len(x_train))

def noise_kernel(val1, val2):
    delta = np.zeros(shape = (len(val1), len(val2)))
    index_val1 = np.arange(len(val1))
    index_val2 = np.arange(len(val2))
    #To my understanding, "delta" is supposed to be an identity matrix. Everything I've read says that when i == j,
    # it's supposed to be 1
    for i in index_val1:
        for j in index_val2:
            if i == j :
                delta[i, j] = 1
            else:
                delta[i, j] = 0
    return delta

#Squared exponential kernel
def sq_exp_kernel(vala, valb, sig_1, sig_2):
    kernel = np.zeros(shape = (len(vala), len(valb))) #This is the matrix that our values get saved to
    index_vala = range(len(vala))
    index_valb = range(len(valb))
    for k in index_vala:
        for l in index_valb:
            #The kernel is conditional due to the different densities of points in the text file, the core is less
            # dense than the pedestal and the edge, so the same correlation length won't work for both of them
            if 0.85 < vala[k] < 1.3 and 0.85 < valb[l] < 1.3:
                kernel[k, l] = np.exp(-(vala[k]-valb[l])**2 / (2 * sig_1**2)) #This function seems to bring up a
                                                                              # deprecation for the one of the
                                                                              # functions in numpy
                                                                              # It'll need to be fixed eventually
            else:
                kernel[k, l] = np.exp(-(vala[k]-valb[l])**2 / (2 * sig_2**2))
    return kernel

#Defining the actual gaussian process
def GP(x1, y1, x2, y_err, kernel_func, noise_funcn):
    y_err_sq = []
    for err in y_err:
        y_err_sq.append(err**2)

    length_train_1 = np.arange(0.01, 0.07, 0.01)#Changing these values too much to have more of a step or search for
                                                # more values can lead to poorly conditioned kernels, which seems to
                                                # make it harder for
                                                # scipy.linalg.solve(K, y1) to find an accurate solution
    length_train_2 = np.arange(0.01, 0.7, 0.01)
    loglikelihood = []
    sig_1_vals = []
    sig_2_vals = []
    for t in length_train_1:
        for u in length_train_2:
            K = kernel_func(x1, x1, t, u) + np.diag(y_err_sq) * noise_funcn(x1, x1)
            log_prob = np.dot(y1, scipy.linalg.solve(K, y1)) + np.log(abs(np.linalg.det(K))) + len(y1) * np.log(2 * np.pi)
            loglikelihood.append(log_prob)
            sig_1_vals.append(t)
            sig_2_vals.append(u)
    #print(loglikelihood)
    sig_1 = 1
    sig_2 = 1
    #A more efficient way to find sig_1 and sig_2 exists, but this method works for now
    for a in np.arange(len(loglikelihood)):
        if loglikelihood[a] == min(loglikelihood):
            sig_1 = sig_1_vals[a]
            sig_2 = sig_2_vals[a]

    #This next part just plots the log probs, it's not entirely necessary
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

    #Calculating these again for 2D plotting purposes, this part can be commented out to save time
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
    #plt.title("$\sigma_{1}$, $\sigma_{2}$")
    #plt.xlabel("$\sigma_n$")
    #plt.ylabel("log prob")
    #plt.legend()
    #plt.savefig("C:\Projects\SULI_Fall_2024\log_probs\logprob_" + type + "_core_" + str(sample_core) + "_ped_" + str(sample_ped) + "_edge_" + str(sample_edge)
                #+ "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
    #plt.show()

    print(sig_1, sig_2)

    K = kernel_func(x1, x1, sig_1, sig_2) + np.diag(y_err_sq) * noise_funcn(x1, x1) #kernel of observations
    K_s = kernel_func(x1, x2, sig_1, sig_2)
    K_0s = kernel_func(x2, x1, sig_1, sig_2)
    K_ss = kernel_func(x2, x2, sig_1, sig_2)

    mu2 = np.dot(K_0s, np.dot(scipy.linalg.inv(K), y1)) #mean considered to be 0 for testing/training points
                                                        # This is the posterior mean
    K_2 = K_ss - np.dot(K_0s, np.dot(scipy.linalg.inv(K), K_s)) #Posterior covariance matrix
    return mu2, K_2, sig_1, sig_2

mu2, K_2, sig_1, sig_2 = GP(x_train, y_train, x_test, y_train_err, sq_exp_kernel, noise_kernel)
y_test = np.random.multivariate_normal(mean = mu2, cov = K_2, size = ny)
sigma = np.sqrt(np.diag(K_2))

#plotting figure + posterior samples + mean
plt.figure(figsize = (12, 8))

for q in range(ny):
    x_array = x_test
    y_array = y_test[q].T
    save_data = np.column_stack([x_array, y_array])
    #Commented out so the script doesn't save 20 files each time it runs
    #np.savetxt("posterior_sample_" + str(q+1) + "_" + type + "_core_" + str(sample_core) + "_ped_" +str(sample_ped) + "edge_" + str(sample_edge)
    #           + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + ".txt",
    #           save_data, fmt = ("%.9e", "%.9e"), delimiter = "\t", header = "Column 1: psi inference; Column 2: " + type_title + " inference")
    plt.plot(x_test, y_test[q].T, '-', alpha = 0.3, color = "g", label = "Posterior Sample" if q == 0 else None)

save_data_2 = np.column_stack([x_test, mu2])
np.savetxt("mean_posterior_sample_" + type + "_core_" + str(sample_core) + "_ped_" +str(sample_ped) + "edge_" + str(sample_edge)
               + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + ".txt",
               save_data_2, fmt = ("%.9e", "%.9e"), delimiter = "\t", header = "Column 1: psi inference; Column 2: " + type_title + " inference")

plt.title("Posterior Samples to " + type_title +" Profile", fontsize = 'xx-large')
plt.plot(psi, te, 'ko', label = type_abrev)
plt.errorbar(psi, te, yerr = te_err, xerr = psi_err, color = "k", fmt = '', ls = 'none')
plt.plot(x_train, y_train, 'ro', label = type_abrev + " Training")
plt.errorbar(x_train, y_train, yerr = y_train_err, xerr = None, color = 'r', fmt = '', ls = 'none')
plt.fill_between(x_test, mu2-2*sigma, mu2+2*sigma, color = "blue", alpha = 0.15, label = "2$\sigma$")
plt.plot(x_test, mu2, color = 'b', label = "$\mu$")
plt.legend(fontsize = 'large')
plt.xlabel("$\psi$", fontsize = 'xx-large')
plt.xlim(domain_x[0], domain_x[1])
if type_abrev == "$T_e$":
    plt.ylim(0, 1)
    plt.ylabel(type_abrev + "$ (keV)$", fontsize = 'xx-large')
if type_abrev == "$n_e$":
    plt.ylim(0, 0.8)
    plt.ylabel(type_abrev + "$ (10^{20}/m^3)$", fontsize = 'xx-large')
if domain_x == (0, 1.3):
    plt.savefig("whole_posterior_fit_to_" + type + "_core_" + str(sample_core) + "_ped_" + str(sample_ped) + "_edge_" + str(sample_edge)
                + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
if domain_x == (0, 0.85):
    plt.savefig("core_posterior_fit_to_" + type + "_core_" + str(sample_core) + "_ped_" + str(sample_ped) + "_edge_" + str(sample_edge)
                + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
if domain_x == (0.85, 1.0):
    plt.savefig("ped_posterior_fit_to_" + type + "_core_" + str(sample_core) + "_ped_" + str(sample_ped) + "_edge_" + str(sample_edge)
                + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
if domain_x == (1.0, 1.3):
    plt.savefig("edge_posterior_fit_to_" + type + "_core_" + str(sample_core) + "_ped_" + str(sample_ped) + "_edge_" + str(sample_edge)
                + "_sample_" + str(sample_test) + "_sig_1_" + str(sig_1) + "_sig_2_" + str(sig_2) + "_plotted_" + str(ny) + ".png")
plt.show()