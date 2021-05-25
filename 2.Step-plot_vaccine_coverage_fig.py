import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.dates as mdates
import datetime as dt


PopulationCounty  = pd.read_excel('CountiesPopulation.xlsx')
County            = np.array(PopulationCounty["county"])
Population        = np.array(PopulationCounty["Population"])
NUM               = len(County)
Matrix  = np.zeros([NUM+1,4])
Matrix1 = np.zeros([NUM+1,4])



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

def plot_scenarios(ax, index, name):
    plt.rcParams['font.size'] = 18
    model0 = Fc(t_pred[:pred], mean_post, X0_new)
    model1 = Fc(t_pred[:pred], mean_post * 0.7, X0_new)  # reduced 30 %
    model2 = Fc(t_pred[:pred], mean_post * 0.4, X0_new)  # reduced 60 %
    model3 = Fc(t_pred[:pred], mean_post * 1.3, X0_new)  # increased 30 %
    if index == 0:
        ax.plot(days[:-pred], W_data, 'k*', markersize=3, label='Data')
        # ax.bar(days[:-pred], W_data, color='grey', width=0.5, alpha=0.4)
    elif index == 1:
        ax.plot(days[:-pred], V_data, 'k*', markersize=3, label='Data')
    elif index == 2:
        ax.plot(days[:-pred], U_data, 'k*', markersize=3, label='Data')
    else:
        ax.plot(days[:-pred], A_data, 'k*', markersize=3, label='Data')
    model_pred = Fc(t_pred, mean_post, X0)

    # if blue:  # Blue shaowed quantiles
    ax.fill_between(days, N * 0.5, N * 0.75, color='gray', alpha=0.2)
    ax.fill_between(days, N * 0.75, N, color='gray', alpha=0.4)
    # ax.fill_between(days[n-1:-1], model1_q05[index], model1_q95[index], color='blue',alpha=0.3)
    # ax.fill_between(days[n - 1:-1], model2_q05[index], model2_q95[index], color='blue',alpha=0.3)
    # ax.fill_between(days[n - 1:-1], model3_q05[index], model3_q95[index], color='blue',alpha=0.3)

    ax.plot(days[:n], model_pred[index][:n])  # U
    ax.plot(days[n - 1:-1], model0[index], label='Keeping', linewidth=2)
    ax.plot(days[n - 1:-1], model1[index], label='Reducing 30%', linewidth=2)
    ax.plot(days[n - 1:-1], model2[index], label='Reducing 60%', linewidth=2)
    ax.plot(days[n - 1:-1], model3[index], label='Increasing 30%', linewidth=2)

    # print valores de tablas

    print('Keeping' + str(name), model0[index][-1] * 1000 / N)
    print('Reducing 30%' + str(name), model1[index][-1] * 1000 / N)
    print('Reducing 60%' + str(name), model2[index][-1] * 1000 / N)
    print('Increasing 30%' + str(name), model3[index][-1] * 1000 / N)

    Matrix[k,] = [round(model0[2][-1] * 1000 / N, 0), round(model1[2][-1] * 1000 / N, 0),
                  round(model2[2][-1] * 1000 / N, 0), round(model3[2][-1] * 1000 / N, 0)]
    Matrix1[k,] = [round(model0[2][-1] * 100 / N, 0), round(model1[2][-1] * 100 / N, 0),
                   round(model2[2][-1] * 100 / N, 0), round(model3[2][-1] * 100 / N, 0)]
    # model1 = Fc(t_pred[:pred], mean_post*0.7, X0_new)  # reduced 30 %
    # model2 = Fc(t_pred[:pred], mean_post*0.4, X0_new)  # reduced 60 %
    # model3 = Fc(t_pred[:pred], mean_post*1.3, X0_new)  # increased 30 %
    # print('days', days[-1])
    # ax.annotate('figure points',  xy=(pred, N*0.5), xycoords='figure points')
    # ax.annotate('50 %', xy=(pred, N*0.5), xycoords='data', xytext=(pred, N*0.5), textcoords='offset points', arrowprops=dict(facecolor='black', shrink=0.05),
    # horizontalalignment='right', verticalalignment='bottom')

    ax.set_xlabel("Date (day.month)", fontsize=18)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.tick_params(which='major', axis='x', labelsize=16)
    plt.axhline(y=N * 0.5, xmin=0, xmax=n + pred, linestyle='--', label='50 % coverage', color='k')
    plt.axhline(y=N * 0.75, xmin=0, xmax=n + pred, linestyle='--', label='75 % coverage', color='b')
    plt.axhline(y=N, xmin=0, xmax=n + pred, linestyle='--', label='100 % coverage', color='r')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=18)
    # lt.ylabel(fontsize=16)
    ax.tick_params(which='major', axis='y', labelsize=18)
    ax.set_ylabel('Population', fontsize=20)





for k in range(NUM):

    N = Population[k]
    County_name = County[k]

    n = 30 # number of data to fitting
    data    = pd.read_excel('CaliforniaVaccine.xlsx')
    data    = data[data['county']==County_name] # filtras al condado
    dates   = data['administered_date'][-n:]
    Dates   = pd.to_datetime(dates)
    rownum  = Dates.index
    #params = pd.read_excel('CaliforniaParameters.xlsx')
    A_data = np.array(data['cumulative_at_least_one_dose'])[-n:]
    U_data = np.array(data['cumulative_fully_vaccinated'])[-n:]
    V_data = A_data - U_data
    W_data = N - A_data

    U0 = U_data[0]
    A0 = A_data[0]
    W0 = N - A0
    V0 = V_data[0]
    X0 = [W0, V0, U0, A0]  # intial condition
    
    #Output_MCMC_file = open("Output_MCMC.pkl", "rb")
    #Output_MCMC = pickle.load(Output_MCMC_file)
    #mean_post = Output_MCMC['mean_post']  # este es el que uso
    #quantiles = Output_MCMC['quantiles']
    
    Output_MCMC_file = np.array(pd.read_excel('CountyMeanPost.xlsx'))
    
    Output_MCMC  = Output_MCMC_file[k,]
    mean_post    = Output_MCMC_file[k,]
    quantiles    = Output_MCMC_file[k,]
    
    pred = 30 # los dias que quiere hacer la proyeccion
    t = np.linspace(0, n-1, n)
    t_pred = np.linspace(0, n + pred - 1, n + pred)
    init = Dates[rownum[0]]  # inicia abril 11
    
    days = mdates.drange(init, init + dt.timedelta(n+pred), dt.timedelta(days=1))
    
    #Out_s, energy, MAP, mean_post, quantiles = Run_twalk(T=T)

    baseline_model = Fc(t, mean_post, X0)
    X0_new = baseline_model[:, -1]

    plt.figure(k)
    fig, ax = plt.subplots(num=k, figsize=(12, 10))
    plot_scenarios(ax=ax, index=2, name='Fully vaccinated')
    fig.savefig("Fully_vaccinated_"+str(County_name)+".png")
    fig.tight_layout()


df = pd.DataFrame(Matrix)
df.to_excel("CountyValuesJune15.xlsx")

df1 = pd.DataFrame(Matrix1)
df1.to_excel("PerCountyValuesJune15.xlsx")
