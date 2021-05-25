# California_Vaccine_Coverage

This repository contains the code and data to study future vaccination coverage by county 
in California using a compartmental mathematical model and Bayesian analysis.

There are two files

1.Step-MCMC_vaccine_coverage.py

2.Step-plot_vaccine_coverage_fig.py

For run

1.Step-MCMC_vaccine_coverage  should be run first to estimate the parameters needed in 
the second file 2.Step-plot_vaccine_coverage_fig. The MCMC use the t-walk 
implementarion.

Data Needed:
1. CountiesPopulation.xlsx = This file provides the total population for each of the California counties.
2. CaliforniaVaccine.xlsx  = This file provides the vaccine information for California. This file was updated on May 16, 2021.
                             Data source: https://data.ca.gov/dataset/covid-19-vaccine-progress-dashboard-data.
                             
Output:
CountyMeanPost.xlsx

After run the MCMC you should run 2.Step-plot_vaccine_coverage_fig.py.

Output:
Plots for all counties
CountyValuesJune15.xlsx = Total of vaccinated people per 1000 inhabitants on June 15, 2021.
PerCountyValuesJune15.xlsx = Percentage of vaccinated people on June 15, 2021.
