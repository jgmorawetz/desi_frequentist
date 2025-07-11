using Pkg
Pkg.activate(".")


# Parameters needed for EFT basis change (same as Hanyu paper)
nd_dict = Dict("BGS" => 1/5723, "LRG1" => 1/5082, "LRG2" => 1/5229, "LRG3" => 1/9574, "ELG2" => 1/10692, "QSO" => 1/47377)
fsat_dict = Dict("BGS" => 0.15, "LRG1" => 0.15, "LRG2" => 0.15, "LRG3" => 0.15, "ELG2" => 0.10, "QSO" => 0.03)
sigv_dict = Dict("BGS" => 5.06, "LRG1" => 6.20, "LRG2" => 6.20, "LRG3" => 6.20, "ELG2" => 3.11, "QSO" => 5.68)

# Emulator range for cosmological parameters (and penalizations for ns10 and BBN when necessary)
cosmo_ranges_FS_BAO = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 100], "ωb" => [0.02, 0.025], "ωc" => [0.08, 0.18], "w0" => [-3, 1], "wa" => [-3, 2])
cosmo_ranges_CMB = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 100], "ωb" => [0.02, 0.025], "ωc" => [0.08, 0.18], "w0" => [-3, 1], "wa" => [-3, 2],
                        "τ" => [0.01, 0.2], "yₚ" => [0.95, 1.05], "Mb_D5" => [-5, 5], "Mb_PP" => [-20, -18], "Mb_U3" => [-20, 20])
cosmo_priors = Dict("ns" => [0.9649, 0.042], "ωb" => [0.02218, 0.00055])

# Search ranges for the EFT parameters when no priors are applied (for profile likelihoods)
eft_ranges = Dict("b1p_BGS" => [0, 6], "b1p_LRG1" => [0, 6], "b1p_LRG2" => [0, 6], "b1p_LRG3" => [0, 6], "b1p_ELG2" => [0, 6], "b1p_QSO" => [0, 6],
                  "b2p_BGS" => [-200, 200], "b2p_LRG1" => [-200, 200], "b2p_LRG2" => [-200, 200], "b2p_LRG3" => [-200, 200], "b2p_ELG2" => [-200, 200], "b2p_QSO" => [-200, 200],
                  "bsp_BGS" => [-200, 200], "bsp_LRG1" => [-200, 200], "bsp_LRG2" => [-200, 200], "bsp_LRG3" => [-200, 200], "bsp_ELG2" => [-200, 200], "bsp_QSO" => [-200, 200],
                  "alpha0p_BGS" => [-500, 500], "alpha0p_LRG1" => [-500, 500], "alpha0p_LRG2" => [-500, 500], "alpha0p_LRG3" => [-500, 500], "alpha0p_ELG2" => [-500, 500], "alpha0p_QSO" => [-500, 500],
                  "alpha2p_BGS" => [-500, 500], "alpha2p_LRG1" => [-500, 500], "alpha2p_LRG2" => [-500, 500], "alpha2p_LRG3" => [-500, 500], "alpha2p_ELG2" => [-500, 500], "alpha2p_QSO" => [-500, 500],
                  "st0p_BGS" => [-100, 100], "st0p_LRG1" => [-100, 100], "st0p_LRG2" => [-100, 100], "st0p_LRG3" => [-100, 100], "st0p_ELG2" => [-100, 100], "st0p_QSO" => [-100, 100],
                  "st2p_BGS" => [-200, 200], "st2p_LRG1" => [-200, 200], "st2p_LRG2" => [-200, 200], "st2p_LRG3" => [-200, 200], "st2p_ELG2" => [-200, 200], "st2p_QSO" => [-200, 200])

# Priors for the EFT parameters (for the MAP and chains, not for profile likelihoods)
eft_priors = Dict("b1p_BGS" => [0, 3], "b1p_LRG1" => [0, 3], "b1p_LRG2" => [0, 3], "b1p_LRG3" => [0, 3], "b1p_ELG2" => [0, 3], "b1p_QSO" => [0, 3],
                  "b2p_BGS" => [0, 5], "b2p_LRG1" => [0, 5], "b2p_LRG2" => [0, 5], "b2p_LRG3" => [0, 5], "b2p_ELG2" => [0, 5], "b2p_QSO" => [0, 5],
                  "bsp_BGS" => [0, 5], "bsp_LRG1" => [0, 5], "bsp_LRG2" => [0, 5], "bsp_LRG3" => [0, 5], "bsp_ELG2" => [0, 5], "bsp_QSO" => [0, 5],
                  "alpha0p_BGS" => [0, 12.5], "alpha0p_LRG1" => [0, 12.5], "alpha0p_LRG2" => [0, 12.5], "alpha0p_LRG3" => [0, 12.5], "alpha0p_ELG2" => [0, 12.5], "alpha0p_QSO" => [0, 12.5],
                  "alpha2p_BGS" => [0, 12.5], "alpha2p_LRG1" => [0, 12.5], "alpha2p_LRG2" => [0, 12.5], "alpha2p_LRG3" => [0, 12.5], "alpha2p_ELG2" => [0, 12.5], "alpha2p_QSO" => [0, 12.5],
                  "st0p_BGS" => [0, 2], "st0p_LRG1" => [0, 2], "st0p_LRG2" => [0, 2], "st0p_LRG3" => [0, 2], "st0p_ELG2" => [0, 2], "st0p_QSO" => [0, 2],
                  "st2p_BGS" => [0, 5], "st2p_LRG1" => [0, 5], "st2p_LRG2" => [0, 5], "st2p_LRG3" => [0, 5], "st2p_ELG2" => [0, 5], "st2p_QSO" => [0, 5])