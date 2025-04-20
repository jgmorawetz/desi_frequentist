import os
import argparse
import numpy as np
from cosmoprimo import * 


def get_sigma8(cosmo_params, z=0, engine='camb'):
    # Function to convert [ln10As, ns, H0, omegab, omegac, w0, wa] basis to sigma8 at redshift z=0
    # Assumes m_ncdm = 0.06, N_ncdm = 1, N_eff = 3.044
    try:
        cosmo = Cosmology(
            ln_A_s_1e10 = cosmo_params[0],
            n_s = cosmo_params[1],
            h = cosmo_params[2]/100,
            omega_b = cosmo_params[3],
            omega_c = cosmo_params[4],
            w0_fld = cosmo_params[5],
            wa_fld = cosmo_params[6],
            m_ncdm = 0.06,
            N_ncdm = 1,
            N_eff = 3.044
        )
        cosmo.set_engine(engine)
        fo_obs = cosmo.get_fourier()
        return fo_obs.sigma8_z([z], of='delta_cb')[0]
    except:
        print(cosmo_params, 'w0wa violation!')
        return 0


def main():

    # Which specific dataset/variation to run computations for
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Specify the dataset')
    parser.add_argument('--variation', type=str, required=True, help='Specify the variation')
    args = parser.parse_args()
    dataset = args.dataset
    variation = args.variation

    MCMC_folder = "/global/homes/j/jgmorawe/FrequentistExample1/FrequentistExample1/MCMC_results_paper"
    chain = np.load(os.path.join(MCMC_folder, f"{dataset}_{variation}_5000_1000_0.65_x6merged_chain.npy"))
    ln10As_values = chain[:, 0]
    ns_values = chain[:, 1]
    H0_values = chain[:, 2]
    omegab_values = chain[:, 3]
    omegac_values = chain[:, 4]
    if variation == "LCDM":
        w0_values = np.full(len(chain), -1)
        wa_values = np.full(len(chain), 0)
    elif variation == "w0waCDM":
        w0_values = chain[:, 5]
        wa_values = chain[:, 6]

    # Converts to new sigma8, Omegam basis instead
    Omegam_values = (omegab_values+omegac_values+0.00064419153)/(H0_values/100)**2
    sigma8_values = np.zeros(len(chain))
    for i in range(len(sigma8_values)):
        cosmo_params = [ln10As_values[i], ns_values[i], H0_values[i], omegab_values[i], omegac_values[i], w0_values[i], wa_values[i]]
        sigma8_values[i] = get_sigma8(cosmo_params)
    # Creates new copy of the chain and replaces ln10As, omegac columns with sigma8, Omegam basis
    chain_copy = np.copy(chain)
    chain_copy[:, 0] = sigma8_values
    chain_copy[:, 4] = Omegam_values
    np.save(os.path.join(MCMC_folder, f"{dataset}_{variation}_5000_1000_0.65_x6merged_chain_sigma8_Omegam_basis.npy"), chain_copy)
            
main()