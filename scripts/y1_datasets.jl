using Pkg
Pkg.activate(".")
using DelimitedFiles
using PlanckLite
using SNIaLikelihoods


# Folder path
desi_dir = "/global/homes/j/jgmorawe/DESI/"

# All available DESI tracers and their labels/indices, effective redshifts, redshift ranges, etc.
tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
redshift_labels = ["z0.1-0.4", "z0.4-0.6", "z0.6-0.8", "z0.8-1.1", "z1.1-1.6", "z0.8-2.1"]
redshift_eff = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
redshift_indices = [1, 2, 3, 4, 6, 7]
redshift_labels = Dict(zip(tracers, redshift_labels))
redshift_eff = Dict(zip(tracers, redshift_eff))
redshift_indices = Dict(zip(tracers, redshift_indices))

# By default uses all tracers
tracer_list = "BGS,LRG1,LRG2,LRG3,ELG2,QSO"
tracer_vector = Vector{String}(split(tracer_list, ","))

# File paths for the power spectra and BAO vectors/covariances, k vectors, window matrices, etc. for each of the tracers
pk_paths = Dict(tracer => desi_dir * "pk_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
baopost_paths = Dict(tracer => desi_dir * "bao-post_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
kin_paths = Dict(tracer => desi_dir * "kin_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
wmat_paths = Dict(tracer => desi_dir * "wmatrix_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
invcov_pk_paths = Dict(tracer => desi_dir * "invcov_pk_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)
invcov_pk_baopost_paths = Dict(tracer => desi_dir * "invcov_pk_bao-post_" * tracer * "_" * redshift_labels[tracer] * ".txt" for tracer in tracers)

# Dictionaries for the power spectra and BAO vectors/covariances, k vectors, window matrices, etc. for each of the tracers
pk_dict = Dict(tracer => vec(readdlm(pk_paths[tracer], ' ')) for tracer in tracers)
baopost_dict = Dict(tracer => vec(readdlm(baopost_paths[tracer], ' ')) for tracer in tracers)
pk_baopost_dict = Dict(tracer => vcat(pk_dict[tracer], baopost_dict[tracer]) for tracer in tracers)
kin_dict = Dict(tracer => vec(readdlm(kin_paths[tracer], ' ')) for tracer in tracers)
wmat_dict = Dict(tracer => readdlm(wmat_paths[tracer], ' ') for tracer in tracers)
invcov_pk_dict = Dict(tracer => readdlm(invcov_pk_paths[tracer], ' ') for tracer in tracers)
invcov_pk_baopost_dict = Dict(tracer => readdlm(invcov_pk_baopost_paths[tracer], ' ') for tracer in tracers)
cov_pk_dict = Dict(tracer => inv(invcov_pk_dict[tracer]) for tracer in tracers)
cov_pk_baopost_dict = Dict(tracer => inv(invcov_pk_baopost_dict[tracer]) for tracer in tracers)
# Isolates the BAO only covariance
cov_size = Dict(tracer => size(cov_pk_baopost_dict[tracer])[1] for tracer in tracers)
cov_baopost_dict = Dict(tracer => (cov_pk_baopost_dict[tracer])[cov_size[tracer]-1:cov_size[tracer], cov_size[tracer]-1:cov_size[tracer]] for tracer in tracers)
cov_baopost_dict["BGS"] = cov_pk_baopost_dict["BGS"][cov_size["BGS"]:cov_size["BGS"], cov_size["BGS"]:cov_size["BGS"]] # last two only have one element
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

# Lyman alpha is stand alone BAO without FS
cov_Lya = inv([3.294630008635918330e+03 1.295814779172360204e+03; 1.295814779172360204e+03 2.235282696116466013e+03])
Γ_Lya = sqrt(cov_Lya)
iΓ_Lya = inv(Γ_Lya)
D_Lya = iΓ_Lya * [9.891736201237542048e-01, 1.013384755757678057e+00]

# Reads in CMB vector/covariance, and then applies whitening transformation
Γ_CMB = sqrt(PlanckLite.cov)
iΓ_CMB = inv(Γ_CMB)
D_CMB = iΓ_CMB * PlanckLite.data

# Reads in supernovae vectors/covariances, and then applies whitening transformation
DESY5SN = DESY5SN_info()
z_DESY5SN = DESY5SN.data.zHD
cov_DESY5SN = DESY5SN.covariance
data_DESY5SN = DESY5SN.obs_flatdata
PantheonPlusSN = PantheonPlusSN_info()
z_PantheonPlusSN = PantheonPlusSN.data.zHD
cov_PantheonPlusSN = PantheonPlusSN.covariance
data_PantheonPlusSN = PantheonPlusSN.obs_flatdata
Union3SN = Union3SN_info()
z_Union3SN = Union3SN.data.zhel
cov_Union3SN = Union3SN.covariance
data_Union3SN = Union3SN.obs_flatdata

Γ_DESY5SN = sqrt(cov_DESY5SN)
iΓ_DESY5SN = inv(Γ_DESY5SN)
D_DESY5SN = iΓ_DESY5SN * data_DESY5SN
Γ_PantheonPlusSN = sqrt(cov_PantheonPlusSN)
iΓ_PantheonPlusSN = inv(Γ_PantheonPlusSN)
D_PantheonPlusSN = iΓ_PantheonPlusSN * data_PantheonPlusSN
Γ_Union3SN = sqrt(cov_Union3SN)
iΓ_Union3SN = inv(Γ_Union3SN)
D_Union3SN = iΓ_Union3SN * data_Union3SN