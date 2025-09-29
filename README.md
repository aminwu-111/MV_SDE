# MV_SDE
This folder contains all code for the paper "Bayesian Parameter Estimation for Partially Observed McKean-Vlasov Diffusions Using Multilevel Markov Chain Monte Carlo"

**Data Files**
    ‘kuramoto_obs_T_100.txt’ contains the data for the Kuramoto model, which is required by ‘PMCMC_KURAMOTO.m’ and ‘MLPMCMC_KURAMOTO.m’
    ‘Modkuramoto_obs_T_100.txt’ contains the data for the Modified Kuramoto model, which is required by ‘PMCMC_MODKURAMOTO.m’ and ‘MLPMCMC_MODKURAMOTO.m’

    
**Main Implementation Files**

    ‘PMCMC_KURAMOTO.m’ and ‘MLPMCMC_KURAMOTO.m’ implement the PMCMC and MLPMCMC algorithms for the Kuramoto model, generating results shown in Figure 1
    ‘PMCMC_MODKURAMOTO.m’ and ‘MLPMCMC_MODKURAMOTO.m’ implement the PMCMC and MLPMCMC algorithms for the Modified Kuramoto model, generating results shown in Figure 2

For the rates of the Kuramoto model process, run the code 'pmmh_l3_KU_rate.m' on ibex(super computer in KAUST) for different levels (maybe 3-6) each with 64 runs and then calculate the mean at each level and then calculate the rates for pmmh. Similar using the 'mlpmmh_l4_KU_rate.m'(4-7) for mlpmmh rates.(Table 1)

For the rates of the Modified Kuramoto model process, run the code 'pmmh_l3_MODKU_rate.m' on ibex(super computer in KAUST) for different levels (maybe 3-6) each with 64 runs and then calculate the mean at each level and then calculate the rates for pmmh. Similar using the 'mlpmmh_l4_MODKU_rate.m'(4-7) for mlpmmh rates.(Table 2)
