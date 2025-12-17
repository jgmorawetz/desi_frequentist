using Pkg
Pkg.activate(".")
using DelimitedFiles
using PlanckLite
using SNIaLikelihoods
using LinearAlgebra


# Folder path to DESI data
desi_dir = "/home/jgmorawe/projects/rrg-wperciva/jgmorawe/frequentist_project/DESI/"

# Dictionaries with information about each tracer (labels/indices, effective redshifts, redshift ranges, etc.)
tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
redshift_labels = ["z0.1-0.4", "z0.4-0.6", "z0.6-0.8", "z0.8-1.1", "z1.1-1.6", "z0.8-2.1"]
redshift_labels = Dict(zip(tracers, redshift_labels))
redshift_eff = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
redshift_eff = Dict(zip(tracers, redshift_eff))
redshift_indices = [1, 2, 3, 4, 6, 7] # missing ELG1 (index 5)
redshift_indices = Dict(zip(tracers, redshift_indices))

# File paths for the power spectra/BAO vectors, k vectors, window matrices, inverse covariance matrices for each tracer
pk_paths = Dict(tracer => desi_dir * "pk_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
baopost_paths = Dict(tracer => desi_dir * "bao-post_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
kin_paths = Dict(tracer => desi_dir * "kin_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
wmat_paths = Dict(tracer => desi_dir * "wmatrix_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers) 
invcov_pk_paths = Dict(tracer => desi_dir * "invcov_pk_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers) 
invcov_pk_baopost_paths = Dict(tracer => desi_dir * "invcov_pk_bao-post_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)

# Dictionaries with the power spectra/BAO vectors, k vectors, window matrices, inverse covariance matrices for each tracer
pk_dict = Dict(tracer => vec(readdlm(pk_paths[tracer], ' ')) for tracer in tracers)
baopost_dict = Dict(tracer => vec(readdlm(baopost_paths[tracer], ' ')) for tracer in tracers)
pk_baopost_dict = Dict(tracer => vcat(pk_dict[tracer], baopost_dict[tracer]) for tracer in tracers)
kin_dict = Dict(tracer => vec(readdlm(kin_paths[tracer], ' ')) for tracer in tracers)
wmat_dict = Dict(tracer => readdlm(wmat_paths[tracer], ' ') for tracer in tracers)
invcov_pk_dict = Dict(tracer => readdlm(invcov_pk_paths[tracer], ' ') for tracer in tracers)
invcov_pk_baopost_dict = Dict(tracer => readdlm(invcov_pk_baopost_paths[tracer], ' ') for tracer in tracers)
cov_pk_dict = Dict(tracer => inv(invcov_pk_dict[tracer]) for tracer in tracers) # inverts the inverse covariance matrices
cov_pk_baopost_dict = Dict(tracer => inv(invcov_pk_baopost_dict[tracer]) for tracer in tracers)
cov_size = Dict(tracer => size(cov_pk_baopost_dict[tracer])[1] for tracer in tracers) # isolates the BAO only covariance
cov_baopost_dict = Dict(tracer => (cov_pk_baopost_dict[tracer])[cov_size[tracer]-1:cov_size[tracer], cov_size[tracer]-1:cov_size[tracer]] for tracer in tracers)
cov_baopost_dict["BGS"] = cov_pk_baopost_dict["BGS"][cov_size["BGS"]:cov_size["BGS"], cov_size["BGS"]:cov_size["BGS"]] # BGS and QSO only have one element
cov_baopost_dict["QSO"] = cov_pk_baopost_dict["QSO"][cov_size["QSO"]:cov_size["QSO"], cov_size["QSO"]:cov_size["QSO"]]

# Applies whitening transformation to improve efficiency in the code
Γ_FS_dict = Dict(tracer => sqrt(cov_pk_dict[tracer]) for tracer in tracers)
iΓ_FS_dict = Dict(tracer => inv(Γ_FS_dict[tracer]) for tracer in tracers)
D_FS_dict = Dict(tracer => iΓ_FS_dict[tracer] * pk_dict[tracer] for tracer in tracers)
Γ_BAO_dict = Dict(tracer => sqrt(cov_baopost_dict[tracer]) for tracer in tracers)
iΓ_BAO_dict = Dict(tracer => inv(Γ_BAO_dict[tracer]) for tracer in tracers)
D_BAO_dict = Dict(tracer => iΓ_BAO_dict[tracer] * baopost_dict[tracer] for tracer in tracers)
Γ_FS_BAO_dict = Dict(tracer => sqrt(cov_pk_baopost_dict[tracer]) for tracer in tracers)
iΓ_FS_BAO_dict = Dict(tracer => inv(Γ_FS_BAO_dict[tracer]) for tracer in tracers)
D_FS_BAO_dict = Dict(tracer => iΓ_FS_BAO_dict[tracer] * pk_baopost_dict[tracer] for tracer in tracers)

# Adds in Lya as standalone BAO (not associated FS)
cov_Lya = inv([3.294630008635918330e+03 1.295814779172360204e+03; 1.295814779172360204e+03 2.235282696116466013e+03])
Γ_Lya = sqrt(cov_Lya)
iΓ_Lya = inv(Γ_Lya)
D_Lya = iΓ_Lya * [9.891736201237542048e-01, 1.013384755757678057e+00]

# Reads in CMB vector/covariance, and then applies whitening transformation
Γ_CMB = sqrt(PlanckLite.cov)
iΓ_CMB = inv(Γ_CMB)
D_CMB = iΓ_CMB * PlanckLite.data

# Reads in supernovae vectors/covariances, and then applies whitening transformation
DY5SN = DESY5SN_info()
z_DY5SN = DY5SN.data.zHD
cov_DY5SN = DY5SN.covariance
Γ_DY5SN = sqrt(cov_DY5SN)
iΓ_DY5SN = inv(Γ_DY5SN)
D_DY5SN = iΓ_DY5SN * DY5SN.obs_flatdata
PPSN = PantheonPlusSN_info()
z_PPSN = PPSN.data.zHD
cov_PPSN = PPSN.covariance
Γ_PPSN = sqrt(cov_PPSN)
iΓ_PPSN = inv(Γ_PPSN)
D_PPSN = iΓ_PPSN * PPSN.obs_flatdata
U3SN = Union3SN_info()
z_U3SN = U3SN.data.zcmb
cov_U3SN = U3SN.covariance
Γ_U3SN = sqrt(cov_U3SN)
iΓ_U3SN = inv(Γ_U3SN)
D_U3SN = iΓ_U3SN * U3SN.obs_flatdata