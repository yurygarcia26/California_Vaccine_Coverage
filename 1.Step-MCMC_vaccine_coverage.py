import numpy as np
import matplotlib.pyplot as plt
from pytwalk import pytwalk
import pandas as pd
from scipy import integrate
import scipy.stats as ss
from scipy.stats import norm,truncnorm, gamma, beta, poisson
import pickle
from datetime import date, timedelta
import matplotlib.dates as mdates
import datetime as dt

PopulationCounty  = pd.read_excel('CountiesPopulation.xlsx')
County            = np.array(PopulationCounty["county"])
Population        = np.array(PopulationCounty["Population"])
NUM               = len(County)
vec_means         = np.zeros([NUM,2])

for k in range(NUM):

    N = Population[k]
    County_name = County[k]
    n = 30 # number of data to fitting
    data = pd.read_excel('CaliforniaVaccine.xlsx')
    data = data[data['county']==County_name] # filtrar el nombre del condado
    dates = data['administered_date'][-n:]
    A_data = np.array(data['cumulative_at_least_one_dose'])[-n:]
    U_data = np.array(data['cumulative_fully_vaccinated'])[-n:]
    V_data = A_data - U_data
    W_data = N - A_data

    t = np.linspace(0,n-1,n)
    U0 = U_data[0]
    A0 = A_data[0]
    W0 = N - A0
    V0 = V_data[0]
    X0 = [W0, V0, U0, A0]  # intial condition
    
    alp_l1 = 3
    bet_l1 = 10
    alp_l2 = 3
    bet_l2 = 10
    
    def F(p):
        l1 = p[0]
        l2 = p[1]
        C = (l1 * W0 / (l1 - l2)) + V0
        U = np.exp(-l1 * t) * W0 * l2 / (l1 - l2) - C * np.exp(-l2 * t) + V0 + W0 + U0
        A = - W0 * np.exp(-l1 * t) + W0 + A0
        return np.array([U, A])
    
    def Fc(t, p, X0):
        l1 = p[0]
        l2 = p[1]
        W0, V0, U0, A0 = X0
        C = (l1 * W0 / (l1 - l2)) + V0
        W = W0 * np.exp(-l1 * t)
        V = -l1 * W0 * np.exp(-l1 * t) / (l1 - l2) + C * np.exp(-l2 * t)
        U = np.exp(-l1 * t) * W0 * l2 / (l1 - l2) - C * np.exp(-l2 * t) + V0 + W0 + U0
        A = - W0 * np.exp(-l1 * t) + W0 + A0
        return np.array([W, V, U, A])
    
    def LogLikelihood(p):
        mu_U, mu_A = F(p)
        log_likelihood_U = np.sum(ss.poisson.logpmf(U_data, mu=mu_U))
        log_likelihood_A = np.sum(ss.poisson.logpmf(A_data, mu=mu_A))
        log_likelihood = log_likelihood_U + log_likelihood_A
        return log_likelihood
    
    def LogPrior(p):
        p_l1 = gamma.logpdf(p[0], alp_l1, scale=1 / bet_l1)
        p_l2 = gamma.logpdf(p[1], alp_l2, scale=1 / bet_l2)
        return p_l1 + p_l2
    
    def Energy(p):
        """ - logarithm of the posterior distribution (could it be proportional) """
        return -(LogLikelihood(p) + LogPrior(p))
    
    def Supp(p):
        """ Check if theta is in the support of the posterior distribution"""
        #rt = (theta[-1] > 0) & (theta[-2] > 0) & (theta[-2] <1)
        rt = all(p>0)
        return rt
    
    def SimInit():
        """ Function to simulate initial values for the gamma distribution """
        p_l1 = gamma.rvs(alp_l1, scale=1 / bet_l1)
        p_l2 = gamma.rvs(alp_l2, scale=1 / bet_l2)
        return np.array([p_l1, p_l2])
    
    def Run_twalk(T):
        d = 2
        start = int(.3 * T)  # Burning
        twalk = pytwalk(n=d, U=Energy, Supp=Supp)     # Open the t-walk object
        twalk.Run(T=T, x0=SimInit(), xp0=SimInit())   # Run the t-walk with two initial values for theta
        hpd_index = np.argsort(twalk.Output[:, -1])   # Maximum a posteriori index
        MAP = twalk.Output[hpd_index[0], :-1]         # MAP
        Out_s = twalk.Output[start:, :-1]
        mean_post = Out_s[start:, :].mean(axis=0)
        quantiles = np.quantile(Out_s[start:, :], axis=0, q=[0.05,0.5,0.95])
        energy = twalk.Output[:, -1]
        return Out_s, energy, MAP, mean_post, quantiles
    
    
    def plot_post(index, Out_s , ax):
        Out_r = Out_s[:, index]  # Output without burning
    
        rt = ax.hist(Out_r, bins=20, density=True)  # Histogram of the simulations
        x = np.linspace(rt[1].min(), rt[1].max(), 100)
    
        if index == 0:
            title = '$l_1$'
            ax.plot(x, gamma.pdf(x, a=alp_l1, scale=1 / bet_l1), 'r-', lw=5, alpha=0.6, label='prior')
            ax.legend()
        else:
            title = '$l_2$'
            ax.plot(x, gamma.pdf(x, a=alp_l2, scale=1 / bet_l2), 'r-', lw=5, alpha=0.6, label='prior')
            ax.legend()
    
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.set_ylabel('Density')
    
    def plot_energy(ax, energy):
        ax.plot(energy)
        ax.set_ylabel('Energy')
        ax.set_xlabel('Iterations')
    
    def plot_all(Out_s,T):
        d=2
        start = int(.3 * T)
        plt.rcParams['font.size'] = 10.0
        fig, _axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        fig.subplots_adjust(hspace=0.8)
        axs = _axs.flatten()
        for i in range(d):
            plot_post(index=i,Out_s=Out_s,ax=axs[i])
        plot_energy(axs[d], energy)
        plot_energy(axs[d+1], energy[start:])
    
    T=10000
    Out_s, energy, MAP, mean_post, quantiles = Run_twalk(T=T)
    Output_MCMC = {"output": Out_s, "mean_post": mean_post, "quantiles": quantiles, "MAP": MAP} # guardar la media posterior
    Output_MCMC_file = open("Output_MCMC.pkl", "wb")
    pickle.dump(Output_MCMC, Output_MCMC_file)
    Output_MCMC_file.close()
    
    vec_means[k,]= mean_post
    
    #sol= F(Out_s.T)
    #Output_MCMC_file = open("Output_MCMC.pkl", "rb")
    #output = pickle.load(Output_MCMC_file)
    #print(output)
    
    #plot_all(Out_s,T)
    
df = pd.DataFrame(vec_means)
df.to_excel("CountyMeanPost.xlsx")