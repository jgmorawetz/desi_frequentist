using Pkg
Pkg.activate(".")


# Parameters needed for EFT physical to Eulerian basis change (same as Hanyu paper)
nd_dict = Dict("BGS" => 1/5723, "LRG1" => 1/5082, "LRG2" => 1/5229, "LRG3" => 1/9574, "ELG2" => 1/10692, "QSO" => 1/47377)
fsat_dict = Dict("BGS" => 0.15, "LRG1" => 0.15, "LRG2" => 0.15, "LRG3" => 0.15, "ELG2" => 0.10, "QSO" => 0.03)
sigv_dict = Dict("BGS" => 5.06, "LRG1" => 6.20, "LRG2" => 6.20, "LRG3" => 6.20, "ELG2" => 3.11, "QSO" => 5.68)

# Emulator ranges for FS/BAO and CMB (and penalizations for ns and BBN when necessary)
cosmo_ranges = Dict("ln10Aₛ" => [2.0, 3.5], "nₛ" => [0.8, 1.1], "h" => [0.5, 0.9], "ωb" => [0.02, 0.025], "ωc" => [0.08, 0.18], "w0" => [-3, 0.5], "wa" => [-3, 2], "τ" => [0.01, 0.2], "yₚ" => [0.95, 1.05],
                    "M_D5" => [-5, 5], "M_PP" => [-20, -18], "M_U3" => [-20, 20])
cosmo_priors = Dict("nₛ" => [0.9649, 0.042], "ωb" => [0.02218, 0.00055], "τ" => [0.0506, 0.0086], "yₚ" => [1.0, 0.0025])
eft_ranges = Dict("b1p" => [0, 3], "b2p" => [-20, 20], "bsp" => [-20, 20], "alpha0p" => [-1000, 1000], "alpha2p" => [-1000, 1000], "st0p" => [-10, 10], "st2p" => [-20, 20])
eft_priors = Dict("b1p" => [0, 3], "b2p" => [0, 5], "bsp" => [0, 5], "alpha0p" => [0, 12.5], "alpha2p" => [0, 12.5], "st0p" => [0, 2], "st2p" => [0, 5])