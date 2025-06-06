# Import statements
using Pkg
Pkg.activate(".")
using ArgParse
using DelimitedFiles
using PlanckLite
using SNIaLikelihoods
using Effort
using Capse
using Turing
using Optim
using LinearAlgebra
using Statistics
using NPZ
using DelimitedFiles
using SharedArrays
using DataInterpolations
using InvertedIndices


# Specifies the number of runs, parameter details and the dataset/variation
config = ArgParseSettings()
@add_arg_table config begin
    "--n_runs"
    help="Specify number of runs"
    arg_type=Int64
    required=true
    "--param_label"
    help="Specify profile parameter"
    arg_type=String
    required=true
    "--param_lower"
    help="Specify the lower bound of profile parameter"
    arg_type=Float64
    required=true
    "--param_upper"
    help="Specify the upper bound of profile parameter"
    arg_type=Float64
    required=true
    "--n_profile"
    help="Specify the total number of values for the profile parameter"
    arg_type=Int64
    required=true
    "--param_index"
    help="Specify the index of the given fixed value of the parameter"
    arg_type=Int64
    required=true
    "--dataset"
    help="Specify dataset"
    arg_type=String
    required=true
    "--variation"
    help="Specify variation"
    arg_type=String
    required=true
    "--chains_path"
    help="Specify the path to the file containing the MCMC chains (for preconditioning/initial guess purposes)"
    arg_type=String
    required=true
    "--MLE_path"
    help="Specify the path to the file containing the MLE best fits (for initial guess)"
end
parsed_args = parse_args(config)
n_runs = parsed_args["n_runs"]
param_label = parsed_args["param_label"]
param_lower = parsed_args["param_lower"]
param_upper = parsed_args["param_upper"]
n_profile = parsed_args["n_profile"]
param_index = parsed_args["param_index"]
dataset = parsed_args["dataset"]
variation = parsed_args["variation"]
chains_path = parsed_args["chains_path"]
MLE_path = parsed_args["MLE_path"]
all_params = LinRange(param_lower, param_upper, n_profile)
param_value = all_params[param_index]


# Relevant folders and file paths
home_dir = "/global/homes/j/jgmorawe/FrequentistExample1/FrequentistExample1/"
save_dir = home_dir * "/profile_likelihood_results_paper/"
desi_data_dir = home_dir * "/DESI_data/DESI/"
FS_emu_dir = home_dir * "/FS_emulator/batch_trained_velocileptors_james_effort_wcdm_20000/"
BAO_emu_dir = home_dir * "/BAO_emulator_ln10As_version/"
BAO_emu_dir_alternate = home_dir * "/BAO_emulator_sigma8_version/"
CMB_emu_dir = home_dir * "/CMB_emulator/"
# Tracers to use for chains
tracer_list = "BGS,LRG1,LRG2,LRG3,ELG2,QSO"
tracer_vector = Vector{String}(split(tracer_list, ","))

# Assembles dictionaries with all the relevant quantities for the different tracers
tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
redshift_labels = ["z0.1-0.4", "z0.4-0.6", "z0.6-0.8", "z0.8-1.1", "z1.1-1.6", "z0.8-2.1"]
redshift_eff = vec(readdlm(desi_data_dir * "zeff_bao-post.txt", ' '))
redshift_indices = [1, 2, 3, 4, 6, 7]
zrange_all = Dict(zip(tracers, redshift_labels))
zeff_all = Dict(zip(tracers, redshift_eff))
zindex_all = Dict(zip(tracers, redshift_indices))

pk_paths = Dict(tracer => desi_data_dir * "pk_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
baopost_paths = Dict(tracer => desi_data_dir * "bao-post_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
kin_paths = Dict(tracer => desi_data_dir * "kin_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
wmat_paths = Dict(tracer => desi_data_dir * "wmatrix_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
invcov_pk_paths = Dict(tracer => desi_data_dir * "invcov_pk_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
invcov_pk_baopost_paths = Dict(tracer => desi_data_dir * "invcov_pk_bao-post_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)

pk_all = Dict(tracer => vec(readdlm(pk_paths[tracer], ' ')) for tracer in tracers)
baopost_all = Dict(tracer => vec(readdlm(baopost_paths[tracer], ' ')) for tracer in tracers)
pk_baopost_all = Dict(tracer => vcat(pk_all[tracer], baopost_all[tracer]) for tracer in tracers)
kin_all = Dict(tracer => vec(readdlm(kin_paths[tracer], ' ')) for tracer in tracers)
wmat_all = Dict(tracer => readdlm(wmat_paths[tracer], ' ') for tracer in tracers)
invcov_pk_all = Dict(tracer => readdlm(invcov_pk_paths[tracer], ' ') for tracer in tracers)
invcov_pk_baopost_all = Dict(tracer => readdlm(invcov_pk_baopost_paths[tracer], ' ') for tracer in tracers)
cov_pk_all = Dict(tracer => inv(invcov_pk_all[tracer]) for tracer in tracers)
cov_pk_baopost_all = Dict(tracer => inv(invcov_pk_baopost_all[tracer]) for tracer in tracers)
cov_size = Dict(tracer => size(cov_pk_baopost_all[tracer])[1] for tracer in tracers)
# Isolates the BAO only covariance (adjusts since only one entry for BGS and QSO)
cov_baopost_all = Dict(tracer => (cov_pk_baopost_all[tracer])[cov_size[tracer]-1:cov_size[tracer], cov_size[tracer]-1:cov_size[tracer]] for tracer in tracers) 
cov_baopost_all["BGS"] = cov_pk_baopost_all["BGS"][cov_size["BGS"]:cov_size["BGS"], cov_size["BGS"]:cov_size["BGS"]]
cov_baopost_all["QSO"] = cov_pk_baopost_all["QSO"][cov_size["QSO"]:cov_size["QSO"], cov_size["QSO"]:cov_size["QSO"]] 

# Reparameterizes the data vectors (for efficiency in the code)
Γ_FS_all = Dict(tracer => sqrt(cov_pk_all[tracer]) for tracer in tracers)
iΓ_FS_all = Dict(tracer => inv(Γ_FS_all[tracer]) for tracer in tracers)
D_FS_all = Dict(tracer => iΓ_FS_all[tracer] * pk_all[tracer] for tracer in tracers)
Γ_BAO_all = Dict(tracer => sqrt(cov_baopost_all[tracer]) for tracer in tracers)
iΓ_BAO_all = Dict(tracer => inv(Γ_BAO_all[tracer]) for tracer in tracers)
D_BAO_all = Dict(tracer => iΓ_BAO_all[tracer] * baopost_all[tracer] for tracer in tracers)
Γ_FS_BAO_all = Dict(tracer => sqrt(cov_pk_baopost_all[tracer]) for tracer in tracers)
iΓ_FS_BAO_all = Dict(tracer => inv(Γ_FS_BAO_all[tracer]) for tracer in tracers)
D_FS_BAO_all = Dict(tracer => iΓ_FS_BAO_all[tracer] * pk_baopost_all[tracer] for tracer in tracers)

# Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
Lya_data = [9.891736201237542048e-01, 1.013384755757678057e+00]
Lya_cov = inv([3.294630008635918330e+03 1.295814779172360204e+03; 1.295814779172360204e+03 2.235282696116466013e+03])
Γ_Lya = sqrt(Lya_cov)
iΓ_Lya = inv(Γ_Lya)
D_Lya = iΓ_Lya * Lya_data

# Reads in the Planck CMB data
Γ_CMB = sqrt(PlanckLite.cov)
iΓ_CMB = inv(Γ_CMB)
D_CMB = iΓ_CMB * PlanckLite.data

# Reads in the supernovae data
DESY5SN = DESY5SN_info()
z_DESY5SN = DESY5SN.data.zHD
cov_DESY5SN = DESY5SN.covariance
data_DESY5SN = DESY5SN.obs_flatdata
Γ_DESY5SN = sqrt(cov_DESY5SN)
iΓ_DESY5SN = inv(Γ_DESY5SN)
D_DESY5SN = iΓ_DESY5SN * data_DESY5SN

PantheonPlusSN = PantheonPlusSN_info()
z_PantheonPlusSN = PantheonPlusSN.data.zHD
cov_PantheonPlusSN = PantheonPlusSN.covariance
data_PantheonPlusSN = PantheonPlusSN.obs_flatdata
Γ_PantheonPlusSN = sqrt(cov_PantheonPlusSN)
iΓ_PantheonPlusSN = inv(Γ_PantheonPlusSN)
D_PantheonPlusSN = iΓ_PantheonPlusSN * data_PantheonPlusSN

Union3SN = Union3SN_info()
z_Union3SN = Union3SN.data.zhel
cov_Union3SN = Union3SN.covariance
data_Union3SN = Union3SN.obs_flatdata
Γ_Union3SN = sqrt(cov_Union3SN)
iΓ_Union3SN = inv(Γ_Union3SN)
D_Union3SN = iΓ_Union3SN * data_Union3SN

# Reads in the full-shape and BAO emulators
mono_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/0/" for tracer in tracers)
quad_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/2/" for tracer in tracers)
hexa_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/4/" for tracer in tracers)
FS_emus = Dict(tracer => [Effort.load_multipole_noise_emulator(mono_paths[tracer]),
                          Effort.load_multipole_noise_emulator(quad_paths[tracer]),
                          Effort.load_multipole_noise_emulator(hexa_paths[tracer])] for tracer in tracers)
BAO_emu = Effort.load_BAO_emulator(BAO_emu_dir)
BAO_emu_alternate = Effort.load_BAO_emulator(BAO_emu_dir_alternate)
# Reads in the emulators associated with the Plancklite CMB data 
TT_emu = Capse.load_emulator(CMB_emu_dir * "/TT/")
TE_emu = Capse.load_emulator(CMB_emu_dir * "/TE/")
EE_emu = Capse.load_emulator(CMB_emu_dir * "/EE/")
CMB_emus = [TT_emu, TE_emu, EE_emu]

# Additional parameters needed for EFT basis change (same as Hanyu paper)
nd_all = Dict("BGS" => 1/5723, "LRG1" => 1/5082, "LRG2" => 1/5229, "LRG3" => 1/9574, "ELG2" => 1/10692, "QSO" => 1/47377)
fsat_all = Dict("BGS" => 0.15, "LRG1" => 0.15, "LRG2" => 0.15, "LRG3" => 0.15, "ELG2" => 0.10, "QSO" => 0.03)
sigv_all = Dict("BGS" => 5.06, "LRG1" => 6.20, "LRG2" => 6.20, "LRG3" => 6.20, "ELG2" => 3.11, "QSO" => 5.68)

# Emulator range for cosmological parameters (and priors for ns10 and BBN when necessary)
cosmo_ranges_FS_BAO = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25], "w0" => [-2, 0.5], "wa" => [-3, 1.64])
cosmo_ranges_CMB = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "w0" => [-2, 0.5], "wa" => [-3, 1.64],
                        "τ" => [0.02, 0.12], "yₚ" => [0.95, 1.05], "Mb_D5" => [-5, 5], "Mb_PP" => [-20, -18], "Mb_U3" => [-20, 20])
cosmo_priors = Dict("ns" => [0.9649, 0.042], "ωb" => [0.02218, 0.00055])
# Uniform prior ranges for EFT parameters for each tracer (exploration range for EFT parameters)
eft_ranges = Dict("b1p_BGS" => [0, 6], "b1p_LRG1" => [0, 6], "b1p_LRG2" => [0, 6], "b1p_LRG3" => [0, 6], "b1p_ELG2" => [0, 6], "b1p_QSO" => [0, 6],
                  "b2p_BGS" => [-200, 200], "b2p_LRG1" => [-200, 200], "b2p_LRG2" => [-200, 200], "b2p_LRG3" => [-200, 200], "b2p_ELG2" => [-200, 200], "b2p_QSO" => [-200, 200],
                  "bsp_BGS" => [-200, 200], "bsp_LRG1" => [-200, 200], "bsp_LRG2" => [-200, 200], "bsp_LRG3" => [-200, 200], "bsp_ELG2" => [-200, 200], "bsp_QSO" => [-200, 200],
                  "alpha0p_BGS" => [-500, 500], "alpha0p_LRG1" => [-500, 500], "alpha0p_LRG2" => [-500, 500], "alpha0p_LRG3" => [-500, 500], "alpha0p_ELG2" => [-500, 500], "alpha0p_QSO" => [-500, 500],
                  "alpha2p_BGS" => [-500, 500], "alpha2p_LRG1" => [-500, 500], "alpha2p_LRG2" => [-500, 500], "alpha2p_LRG3" => [-500, 500], "alpha2p_ELG2" => [-500, 500], "alpha2p_QSO" => [-500, 500],
                  "st0p_BGS" => [-80, 80], "st0p_LRG1" => [-80, 80], "st0p_LRG2" => [-80, 80], "st0p_LRG3" => [-80, 80], "st0p_ELG2" => [-80, 80], "st0p_QSO" => [-80, 80],
                  "st2p_BGS" => [-200, 200], "st2p_LRG1" => [-200, 200], "st2p_LRG2" => [-200, 200], "st2p_LRG3" => [-200, 200], "st2p_ELG2" => [-200, 200], "st2p_QSO" => [-200, 200])


function theory_FS(theta_FS, emu_FS_components, kin)
    """Constructs theory vector for full-shape."""
    cosmo_params = theta_FS[1:7] # ln10As, ns, H0, ωb, ωc, w0, wa
    eft_params = theta_FS[8:18] # b1, b2, b3, bs, alpha0, alpha2, alpha4, alpha6, st0, st2, st4
    mono_emu = emu_FS_components[1]
    quad_emu = emu_FS_components[2]
    hexa_emu = emu_FS_components[3]
    pk0 = Effort.get_Pℓ(cosmo_params, eft_params, mono_emu)
    pk2 = Effort.get_Pℓ(cosmo_params, eft_params, quad_emu)
    pk4 = Effort.get_Pℓ(cosmo_params, eft_params, hexa_emu)
    return vcat(Effort._akima_spline(pk0, mono_emu.Pℓ.P11.kgrid, kin),
                Effort._akima_spline(pk2, quad_emu.Pℓ.P11.kgrid, kin),
                Effort._akima_spline(pk4, hexa_emu.Pℓ.P11.kgrid, kin))
end
                
function theory_BAO(theta_BAO, emu_BAO, zeff, tracer)
    """Constructs theory vector for post-recon BAO."""
    # theta_BAO is vector [ln10As, ns, H0, ωb, ωc, w0, wa]
    ln10As_fid, ns_fid, H0_fid, ωb_fid, ωc_fid, w0_fid, wa_fid = 3.044, 0.9649, 67.36, 0.02237, 0.1200, -1, 0 # fiducial planck 2018 cosmology
    theta_BAO_fid = [ln10As_fid, ns_fid, H0_fid, ωb_fid, ωc_fid, w0_fid, wa_fid]
    h_fid = theta_BAO_fid[3]/100; Ωcb_fid = (theta_BAO_fid[4]+theta_BAO_fid[5])/h_fid^2
    h_true = theta_BAO[3]/100; Ωcb_true = (theta_BAO[4]+theta_BAO[5])/h_true^2; w0_true=theta_BAO[6]; wa_true=theta_BAO[7]
    mν_fixed = 0.06
    # Computes H(z) and D_A(z) for fid and model cosmologies
    H_fid = h_fid*Effort._E_z(zeff, Ωcb_fid, h_fid; mν=mν_fixed, w0=w0_fid, wa=wa_fid)
    H_true = h_true*Effort._E_z(zeff, Ωcb_true, h_true; mν=mν_fixed, w0=w0_true, wa=wa_true)
    DA_fid = Effort._r_z(zeff, Ωcb_fid, h_fid; mν=mν_fixed, w0=w0_fid, wa=wa_fid)
    DA_true = Effort._r_z(zeff, Ωcb_true, h_true; mν=mν_fixed, w0=w0_true, wa=wa_true)
    # Computes rs_drag from emulator
    rsdrag_fid = Effort.get_BAO(theta_BAO_fid, emu_BAO)[1] # rs_drag is first entry
    rsdrag_true = Effort.get_BAO(theta_BAO, emu_BAO)[1]
    # Converts to alpha par and perp (or iso) components
    alpha_par = (H_fid*rsdrag_fid)/(H_true*rsdrag_true)
    alpha_perp = (DA_true*rsdrag_fid)/(DA_fid*rsdrag_true)
    alpha_iso = (alpha_par*alpha_perp^2)^(1/3)
    # Returns either [alpha_par, alpha_perp] or [alpha_iso] depending on the tracer
    if tracer in ["LRG1", "LRG2", "LRG3", "ELG2", "Lya"]
        return [alpha_par, alpha_perp]
    elseif tracer in ["BGS", "QSO"]
        return [alpha_iso]
    end
end
               
function theory_CMB(theta_CMB, emu_CMB_components)
    """Constructs theory vector for CMB."""
    # theta CMB: [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    emu_TT = emu_CMB_components[1]
    emu_TE = emu_CMB_components[2]
    emu_EE = emu_CMB_components[3]
    lsTT = 2:2508
    lsTE = 2:1996
    facTT=lsTT.*(lsTT.+1)./(2*π)
    facTE=lsTE.*(lsTE.+1)./(2*π)
    return PlanckLite.bin_Cℓ(Capse.get_Cℓ(theta_CMB, emu_TT)[1:2507]./facTT,
                             Capse.get_Cℓ(theta_CMB, emu_TE)[1:1995]./facTE,
                             Capse.get_Cℓ(theta_CMB, emu_EE)[1:1995]./facTE)
end
                
function theory_SN(theta_SN, Mb, z_SN, SN_type)
    """Constructs theory vector for supernovae."""
    # theta_SN: [ln10As, ns, H0, ωb, ωc, w0, wa]
    h = theta_SN[3]/100; Ωcb = (theta_SN[4]+theta_SN[5])/h^2; w0 = theta_SN[6]; wa = theta_SN[7]
    mν_fixed = 0.06
    z_interp = Array(LinRange(0, 2.5, 50)) # uses interpolation to not have to calculate for all supernovae redshifts
    DL_interp = Effort._r_z.(z_interp, Ωcb, h; mν=mν_fixed, w0=w0, wa=wa)
    DL_SN = DataInterpolations.QuadraticSpline(DL_interp, z_interp).(z_SN) .* (1 .+ z_SN)
    if SN_type == "DESY5SN"
        return 5 .* log10.(DL_SN) .+ 25 .+ Mb
    elseif SN_type == "PantheonPlusSN"
        return 5 .* log10.(DL_SN) .+ 25 .+ Mb
    elseif SN_type == "Union3SN"
        return 5 .* log10.(100 .* DL_SN .* h) .+ 25 .+ Mb
    end
end
        

@model function model_FS(D_FS_all)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_FS_BAO["ln10As"][1], cosmo_ranges_FS_BAO["ln10As"][2])
    ns ~ Truncated(Normal(cosmo_priors["ns"][1], cosmo_priors["ns"][2]), cosmo_ranges_FS_BAO["ns"][1], cosmo_ranges_FS_BAO["ns"][2])               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    ωc ~ Uniform(cosmo_ranges_FS_BAO["ωc"][1], cosmo_ranges_FS_BAO["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS theory vector given parameters
        prediction_FS = iΓ_FS_all[tracer]*(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]))
        D_FS_all[tracer] ~ MvNormal(prediction_FS, I)
    end
end

@model function model_FS_BAO(D_FS_BAO_all, D_Lya)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_FS_BAO["ln10As"][1], cosmo_ranges_FS_BAO["ln10As"][2])
    ns ~ Truncated(Normal(cosmo_priors["ns"][1], cosmo_priors["ns"][2]), cosmo_ranges_FS_BAO["ns"][1], cosmo_ranges_FS_BAO["ns"][2])               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    ωc ~ Uniform(cosmo_ranges_FS_BAO["ωc"][1], cosmo_ranges_FS_BAO["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
end

@model function model_FS_BAO_CMB(D_FS_BAO_all, D_Lya, D_CMB)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_CMB["ln10As"][1], cosmo_ranges_CMB["ln10As"][2])
    ns ~ Uniform(cosmo_ranges_CMB["ns"][1], cosmo_ranges_CMB["ns"][2])         
    H0 ~ Uniform(cosmo_ranges_CMB["H0"][1], cosmo_ranges_CMB["H0"][2])
    ωb ~ Uniform(cosmo_ranges_CMB["ωb"][1], cosmo_ranges_CMB["ωb"][2])           
    ωc ~ Uniform(cosmo_ranges_CMB["ωc"][1], cosmo_ranges_CMB["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_CMB["w0"][1], cosmo_ranges_CMB["w0"][2])
    wa ~ Uniform(cosmo_ranges_CMB["wa"][1], cosmo_ranges_CMB["wa"][2])
    # Parameters for CMB contribution
    τ ~ Truncated(Normal(0.0506, 0.0086), cosmo_ranges_CMB["τ"][1], cosmo_ranges_CMB["τ"][2])
    mν = 0.06
    yₚ ~ Truncated(Normal(1.0, 0.0025), cosmo_ranges_CMB["yₚ"][1], cosmo_ranges_CMB["yₚ"][2])
    cosmo_params_FS_BAO = [ln10As, ns, H0, ωb, ωc, w0, wa]
    cosmo_params_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params_FS_BAO, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params_FS_BAO, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
end

@model function model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_SN, D_SN, z_SN, SN_type)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_CMB["ln10As"][1], cosmo_ranges_CMB["ln10As"][2])
    ns ~ Uniform(cosmo_ranges_CMB["ns"][1], cosmo_ranges_CMB["ns"][2])         
    H0 ~ Uniform(cosmo_ranges_CMB["H0"][1], cosmo_ranges_CMB["H0"][2])
    ωb ~ Uniform(cosmo_ranges_CMB["ωb"][1], cosmo_ranges_CMB["ωb"][2])           
    ωc ~ Uniform(cosmo_ranges_CMB["ωc"][1], cosmo_ranges_CMB["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_CMB["w0"][1], cosmo_ranges_CMB["w0"][2])
    wa ~ Uniform(cosmo_ranges_CMB["wa"][1], cosmo_ranges_CMB["wa"][2])
    # Parameters for CMB contribution
    τ ~ Truncated(Normal(0.0506, 0.0086), cosmo_ranges_CMB["τ"][1], cosmo_ranges_CMB["τ"][2])
    mν = 0.06
    yₚ ~ Truncated(Normal(1.0, 0.0025), cosmo_ranges_CMB["yₚ"][1], cosmo_ranges_CMB["yₚ"][2])
    # Parameters for SN contribution
    if SN_type == "DESY5SN"
        Mb ~ Uniform(-5, 5)
    elseif SN_type == "PantheonPlusSN"
        Mb ~ Uniform(-20, -18)
    elseif SN_type == "Union3SN"
        Mb ~ Uniform(-20, 20)
    end
    cosmo_params_FS_BAO = [ln10As, ns, H0, ωb, ωc, w0, wa]
    cosmo_params_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params_FS_BAO, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params_FS_BAO, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
    # Adds SN contribution
    prediction_SN = iΓ_SN * theory_SN(cosmo_params_FS_BAO, Mb, z_SN, SN_type)
    D_SN ~ MvNormal(prediction_SN, I)
end


# Creates separate versions of the functions to handle Omega_m (since in different basis)

@model function model_FS_Omegam(D_FS_all)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_FS_BAO["ln10As"][1], cosmo_ranges_FS_BAO["ln10As"][2])
    ns ~ Truncated(Normal(cosmo_priors["ns"][1], cosmo_priors["ns"][2]), cosmo_ranges_FS_BAO["ns"][1], cosmo_ranges_FS_BAO["ns"][2])               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    Om ~ Uniform(0.01, 1) # doesn't matter the range used here since fixing based on profile values anyway
    ωc = Om*(H0/100)^2 - ωb - 0.00064419153
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS theory vector given parameters
        prediction_FS = iΓ_FS_all[tracer]*(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]))
        D_FS_all[tracer] ~ MvNormal(prediction_FS, I)
    end
end

@model function model_FS_BAO_Omegam(D_FS_BAO_all, D_Lya)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_FS_BAO["ln10As"][1], cosmo_ranges_FS_BAO["ln10As"][2])
    ns ~ Truncated(Normal(cosmo_priors["ns"][1], cosmo_priors["ns"][2]), cosmo_ranges_FS_BAO["ns"][1], cosmo_ranges_FS_BAO["ns"][2])               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    Om ~ Uniform(0.01, 1) # doesn't matter the range used here since fixing based on profile values anyway
    ωc = Om*(H0/100)^2 - ωb - 0.00064419153
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
end

@model function model_FS_BAO_CMB_Omegam(D_FS_BAO_all, D_Lya, D_CMB)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_CMB["ln10As"][1], cosmo_ranges_CMB["ln10As"][2])
    ns ~ Uniform(cosmo_ranges_CMB["ns"][1], cosmo_ranges_CMB["ns"][2])         
    H0 ~ Uniform(cosmo_ranges_CMB["H0"][1], cosmo_ranges_CMB["H0"][2])
    ωb ~ Uniform(cosmo_ranges_CMB["ωb"][1], cosmo_ranges_CMB["ωb"][2])           
    Om ~ Uniform(0.01, 1) # doesn't matter the range used here since fixing based on profile values anyway
    ωc = Om*(H0/100)^2 - ωb - 0.00064419153
    w0 ~ Uniform(cosmo_ranges_CMB["w0"][1], cosmo_ranges_CMB["w0"][2])
    wa ~ Uniform(cosmo_ranges_CMB["wa"][1], cosmo_ranges_CMB["wa"][2])
    # Parameters for CMB contribution
    τ ~ Truncated(Normal(0.0506, 0.0086), cosmo_ranges_CMB["τ"][1], cosmo_ranges_CMB["τ"][2])
    mν = 0.06
    yₚ ~ Truncated(Normal(1.0, 0.0025), cosmo_ranges_CMB["yₚ"][1], cosmo_ranges_CMB["yₚ"][2])
    cosmo_params_FS_BAO = [ln10As, ns, H0, ωb, ωc, w0, wa]
    cosmo_params_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params_FS_BAO, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params_FS_BAO, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
end

@model function model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_SN, D_SN, z_SN, SN_type)
    # Draws cosmological parameters
    ln10As ~ Uniform(cosmo_ranges_CMB["ln10As"][1], cosmo_ranges_CMB["ln10As"][2])
    ns ~ Uniform(cosmo_ranges_CMB["ns"][1], cosmo_ranges_CMB["ns"][2])         
    H0 ~ Uniform(cosmo_ranges_CMB["H0"][1], cosmo_ranges_CMB["H0"][2])
    ωb ~ Uniform(cosmo_ranges_CMB["ωb"][1], cosmo_ranges_CMB["ωb"][2])           
    Om ~ Uniform(0.01, 1) # doesn't matter the range used here since fixing based on profile values anyway
    ωc = Om*(H0/100)^2 - ωb - 0.00064419153
    w0 ~ Uniform(cosmo_ranges_CMB["w0"][1], cosmo_ranges_CMB["w0"][2])
    wa ~ Uniform(cosmo_ranges_CMB["wa"][1], cosmo_ranges_CMB["wa"][2])
    # Parameters for CMB contribution
    τ ~ Truncated(Normal(0.0506, 0.0086), cosmo_ranges_CMB["τ"][1], cosmo_ranges_CMB["τ"][2])
    mν = 0.06
    yₚ ~ Truncated(Normal(1.0, 0.0025), cosmo_ranges_CMB["yₚ"][1], cosmo_ranges_CMB["yₚ"][2])
    # Parameters for SN contribution
    if SN_type == "DESY5SN"
        Mb ~ Uniform(-5, 5)
    elseif SN_type == "PantheonPlusSN"
        Mb ~ Uniform(-20, -18)
    elseif SN_type == "Union3SN"
        Mb ~ Uniform(-20, 20)
    end
    cosmo_params_FS_BAO = [ln10As, ns, H0, ωb, ωc, w0, wa]
    cosmo_params_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params_FS_BAO, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params_FS_BAO, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
    # Adds SN contribution
    prediction_SN = iΓ_SN * theory_SN(cosmo_params_FS_BAO, Mb, z_SN, SN_type)
    D_SN ~ MvNormal(prediction_SN, I)
end


# Creates separate versions of the functions to handle sigma8 (since in different basis)

@model function model_FS_sigma8(D_FS_all)
    # Draws cosmological parameters
    sigma8 ~ Uniform(0.01, 2) # doesn't matter the range used here since fixing based on profile values anyway
    ns ~ Truncated(Normal(cosmo_priors["ns"][1], cosmo_priors["ns"][2]), cosmo_ranges_FS_BAO["ns"][1], cosmo_ranges_FS_BAO["ns"][2])               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    ωc ~ Uniform(cosmo_ranges_FS_BAO["ωc"][1], cosmo_ranges_FS_BAO["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    ln10As = Effort.get_BAO([sigma8, ns, H0, ωb, ωc, w0, wa], BAO_emu_alternate)[1]
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS theory vector given parameters
        prediction_FS = iΓ_FS_all[tracer]*(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]))
        D_FS_all[tracer] ~ MvNormal(prediction_FS, I)
    end
end

@model function model_FS_BAO_sigma8(D_FS_BAO_all, D_Lya)
    # Draws cosmological parameters
    sigma8 ~ Uniform(0.01, 2) # doesn't matter the range used here since fixing based on profile values anyway
    ns ~ Truncated(Normal(cosmo_priors["ns"][1], cosmo_priors["ns"][2]), cosmo_ranges_FS_BAO["ns"][1], cosmo_ranges_FS_BAO["ns"][2])               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    ωc ~ Uniform(cosmo_ranges_FS_BAO["ωc"][1], cosmo_ranges_FS_BAO["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    ln10As = Effort.get_BAO([sigma8, ns, H0, ωb, ωc, w0, wa], BAO_emu_alternate)[1]
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
end

@model function model_FS_BAO_CMB_sigma8(D_FS_BAO_all, D_Lya, D_CMB)
    # Draws cosmological parameters
    sigma8 ~ Uniform(0.01, 2) # doesn't matter the range used here since fixing based on profile values anyway
    ns ~ Uniform(cosmo_ranges_CMB["ns"][1], cosmo_ranges_CMB["ns"][2])         
    H0 ~ Uniform(cosmo_ranges_CMB["H0"][1], cosmo_ranges_CMB["H0"][2])
    ωb ~ Uniform(cosmo_ranges_CMB["ωb"][1], cosmo_ranges_CMB["ωb"][2])           
    ωc ~ Uniform(cosmo_ranges_CMB["ωc"][1], cosmo_ranges_CMB["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_CMB["w0"][1], cosmo_ranges_CMB["w0"][2])
    wa ~ Uniform(cosmo_ranges_CMB["wa"][1], cosmo_ranges_CMB["wa"][2])
    ln10As = Effort.get_BAO([sigma8, ns, H0, ωb, ωc, w0, wa], BAO_emu_alternate)[1]
    # Parameters for CMB contribution
    τ ~ Truncated(Normal(0.0506, 0.0086), cosmo_ranges_CMB["τ"][1], cosmo_ranges_CMB["τ"][2])
    mν = 0.06
    yₚ ~ Truncated(Normal(1.0, 0.0025), cosmo_ranges_CMB["yₚ"][1], cosmo_ranges_CMB["yₚ"][2])
    cosmo_params_FS_BAO = [ln10As, ns, H0, ωb, ωc, w0, wa]
    cosmo_params_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params_FS_BAO, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params_FS_BAO, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
end

@model function model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_SN, D_SN, z_SN, SN_type)
    # Draws cosmological parameters
    sigma8 ~ Uniform(0.01, 2) # doesn't matter the range used here since fixing based on profile values anyway
    ns ~ Uniform(cosmo_ranges_CMB["ns"][1], cosmo_ranges_CMB["ns"][2])         
    H0 ~ Uniform(cosmo_ranges_CMB["H0"][1], cosmo_ranges_CMB["H0"][2])
    ωb ~ Uniform(cosmo_ranges_CMB["ωb"][1], cosmo_ranges_CMB["ωb"][2])           
    ωc ~ Uniform(cosmo_ranges_CMB["ωc"][1], cosmo_ranges_CMB["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_CMB["w0"][1], cosmo_ranges_CMB["w0"][2])
    wa ~ Uniform(cosmo_ranges_CMB["wa"][1], cosmo_ranges_CMB["wa"][2])
    ln10As = Effort.get_BAO([sigma8, ns, H0, ωb, ωc, w0, wa], BAO_emu_alternate)[1]
    # Parameters for CMB contribution
    τ ~ Truncated(Normal(0.0506, 0.0086), cosmo_ranges_CMB["τ"][1], cosmo_ranges_CMB["τ"][2])
    mν = 0.06
    yₚ ~ Truncated(Normal(1.0, 0.0025), cosmo_ranges_CMB["yₚ"][1], cosmo_ranges_CMB["yₚ"][2])
    # Parameters for SN contribution
    if SN_type == "DESY5SN"
        Mb ~ Uniform(-5, 5)
    elseif SN_type == "PantheonPlusSN"
        Mb ~ Uniform(-20, -18)
    elseif SN_type == "Union3SN"
        Mb ~ Uniform(-20, 20)
    end
    cosmo_params_FS_BAO = [ln10As, ns, H0, ωb, ωc, w0, wa]
    cosmo_params_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params_FS_BAO, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p_BGS"][1], eft_ranges["b1p_BGS"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p_BGS"][1], eft_ranges["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp_BGS"][1], eft_ranges["bsp_BGS"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p_BGS"][1], eft_ranges["alpha0p_BGS"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p_BGS"][1], eft_ranges["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p_BGS"][1], eft_ranges["st0p_BGS"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p_BGS"][1], eft_ranges["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p_LRG1"][1], eft_ranges["b1p_LRG1"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p_LRG1"][1], eft_ranges["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp_LRG1"][1], eft_ranges["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p_LRG1"][1], eft_ranges["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p_LRG1"][1], eft_ranges["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p_LRG1"][1], eft_ranges["st0p_LRG1"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p_LRG1"][1], eft_ranges["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p_LRG2"][1], eft_ranges["b1p_LRG2"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p_LRG2"][1], eft_ranges["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp_LRG2"][1], eft_ranges["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p_LRG2"][1], eft_ranges["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p_LRG2"][1], eft_ranges["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p_LRG2"][1], eft_ranges["st0p_LRG2"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p_LRG2"][1], eft_ranges["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p_LRG3"][1], eft_ranges["b1p_LRG3"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p_LRG3"][1], eft_ranges["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp_LRG3"][1], eft_ranges["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p_LRG3"][1], eft_ranges["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p_LRG3"][1], eft_ranges["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p_LRG3"][1], eft_ranges["st0p_LRG3"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p_LRG3"][1], eft_ranges["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p_ELG2"][1], eft_ranges["b1p_ELG2"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p_ELG2"][1], eft_ranges["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp_ELG2"][1], eft_ranges["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p_ELG2"][1], eft_ranges["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p_ELG2"][1], eft_ranges["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p_ELG2"][1], eft_ranges["st0p_ELG2"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p_ELG2"][1], eft_ranges["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p_QSO"][1], eft_ranges["b1p_QSO"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p_QSO"][1], eft_ranges["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp_QSO"][1], eft_ranges["bsp_QSO"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p_QSO"][1], eft_ranges["alpha0p_QSO"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p_QSO"][1], eft_ranges["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p_QSO"][1], eft_ranges["st0p_QSO"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p_QSO"][1], eft_ranges["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_all[tracer]); st2e = st2p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^2; st4e = st4p/(nd_all[tracer])*(fsat_all[tracer])*(sigv_all[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_all[tracer]*vcat(wmat_all[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_all[tracer]),
                                                       theory_BAO(cosmo_params_FS_BAO, BAO_emu, zeff_all[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
    # Adds SN contribution
    prediction_SN = iΓ_SN * theory_SN(cosmo_params_FS_BAO, Mb, z_SN, SN_type)
    D_SN ~ MvNormal(prediction_SN, I)
end


# Retrieves the models given the parameter and cosmological model of interest
if variation == "LCDM"
    if param_label == "sigma8"
        FS_model = model_FS_sigma8(D_FS_all) | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_model = model_FS_BAO_sigma8(D_FS_BAO_all, D_Lya) | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_model = model_FS_BAO_CMB_sigma8(D_FS_BAO_all, D_Lya, D_CMB) | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (sigma8=param_value, w0=-1, wa=0)
    elseif param_label == "H0"
        FS_model = model_FS(D_FS_all) | (H0=param_value, w0=-1, wa=0)
        FS_BAO_model = model_FS_BAO(D_FS_BAO_all, D_Lya) | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_model = model_FS_BAO_CMB(D_FS_BAO_all, D_Lya, D_CMB) | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (H0=param_value, w0=-1, wa=0)
    elseif param_label == "Omegam"
        FS_model = model_FS_Omegam(D_FS_all) | (Om=param_value, w0=-1, wa=0)
        FS_BAO_model = model_FS_BAO_Omegam(D_FS_BAO_all, D_Lya) | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_model = model_FS_BAO_CMB_Omegam(D_FS_BAO_all, D_Lya, D_CMB) | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (Om=param_value, w0=-1, wa=0)
    end
elseif variation == "w0waCDM"
    if param_label == "sigma8"
        FS_model = model_FS_sigma8(D_FS_all) | (sigma8=param_value,)
        FS_BAO_model = model_FS_BAO_sigma8(D_FS_BAO_all, D_Lya) | (sigma8=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_sigma8(D_FS_BAO_all, D_Lya, D_CMB) | (sigma8=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (sigma8=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (sigma8=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_sigma8(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (sigma8=param_value,)
    elseif param_label == "H0"
        FS_model = model_FS(D_FS_all) | (H0=param_value,)
        FS_BAO_model = model_FS_BAO(D_FS_BAO_all, D_Lya) | (H0=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB(D_FS_BAO_all, D_Lya, D_CMB) | (H0=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (H0=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (H0=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (H0=param_value,)
    elseif param_label == "Omegam"
        FS_model = model_FS_Omegam(D_FS_all) | (Om=param_value,)
        FS_BAO_model = model_FS_BAO_Omegam(D_FS_BAO_all, D_Lya) | (Om=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_Omegam(D_FS_BAO_all, D_Lya, D_CMB) | (Om=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (Om=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (Om=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_Omegam(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (Om=param_value,)
    elseif param_label == "w0"
        FS_model = model_FS(D_FS_all) | (w0=param_value,)
        FS_BAO_model = model_FS_BAO(D_FS_BAO_all, D_Lya) | (w0=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB(D_FS_BAO_all, D_Lya, D_CMB) | (w0=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (w0=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (w0=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (w0=param_value,)
    elseif param_label == "wa"
        FS_model = model_FS(D_FS_all) | (wa=param_value,)
        FS_BAO_model = model_FS_BAO(D_FS_BAO_all, D_Lya) | (wa=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB(D_FS_BAO_all, D_Lya, D_CMB) | (wa=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (wa=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (wa=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN(D_FS_BAO_all, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (wa=param_value,)
    end
end


# Retrieves the specific model given the desired dataset
if dataset == "FS"
    fit_model = FS_model
    if variation == "LCDM"
        n_fit_params = 4 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        n_fit_params = 6 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "FS+BAO"
    fit_model = FS_BAO_model
    if variation == "LCDM"
        n_fit_params = 4 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        n_fit_params = 6 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "FS+BAO+CMB"
    fit_model = FS_BAO_CMB_model
    if variation == "LCDM"
        n_fit_params = 6 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ"]
    elseif variation == "w0waCDM"
        n_fit_params = 8 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ"]
    end
elseif dataset == "FS+BAO+CMB+DESY5SN"
    fit_model = FS_BAO_CMB_DESY5SN_model
    if variation == "LCDM"
        n_fit_params = 7 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_D5"]
    elseif variation == "w0waCDM"
        n_fit_params = 9 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_D5"]
    end
elseif dataset == "FS+BAO+CMB+PantheonPlusSN"
    fit_model = FS_BAO_CMB_PantheonPlusSN_model
    if variation == "LCDM"
        n_fit_params = 7 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_PP"]
    elseif variation == "w0waCDM"
        n_fit_params = 9 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_PP"]
    end
elseif dataset == "FS+BAO+CMB+Union3SN"
    fit_model = FS_BAO_CMB_Union3SN_model
    if variation == "LCDM"
        n_fit_params = 7 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_U3"]
    elseif variation == "w0waCDM"
        n_fit_params = 9 + 7*size(tracer_vector)[1]; cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_U3"]
    end
end

# Removes the one parameter being profiled from cosmo_fit_labels
if param_label == "sigma8"
    cosmo_fit_labels = [cosmo_fit_label for cosmo_fit_label in cosmo_fit_labels if cosmo_fit_label != "ln10As"]
elseif param_label == "H0"
    cosmo_fit_labels = [cosmo_fit_label for cosmo_fit_label in cosmo_fit_labels if cosmo_fit_label != "H0"]
elseif param_label == "Omegam"
    cosmo_fit_labels = [cosmo_fit_label for cosmo_fit_label in cosmo_fit_labels if cosmo_fit_label != "ωc"]
elseif param_label == "w0"
    cosmo_fit_labels = [cosmo_fit_label for cosmo_fit_label in cosmo_fit_labels if cosmo_fit_label != "w0"]
elseif param_label == "wa"
    cosmo_fit_labels = [cosmo_fit_label for cosmo_fit_label in cosmo_fit_labels if cosmo_fit_label != "wa"]
end

eft_fit_labels = []
if dataset in ["FS", "FS+BAO", "FS+BAO+CMB", "FS+BAO+CMB+DESY5SN", "FS+BAO+CMB+PantheonPlusSN", "FS+BAO+CMB+Union3SN"]
    for tracer in tracer_vector
        append!(eft_fit_labels, ["b1p_$(tracer)", "b2p_$(tracer)", "bsp_$(tracer)", "alpha0p_$(tracer)", "alpha2p_$(tracer)", "st0p_$(tracer)", "st2p_$(tracer)"])
    end
end


# Reads in the file storing the MCMC chains and MLE estimates in order to set the preconditioning matrix and initial guess distributions
MCMC_chains = npzread(chains_path)
MLE_bestfits = npzread(MLE_path)
if param_label == "sigma8"
    MCMC_chains = MCMC_chains[:, Not(1)]
    MLE_bestfits = MLE_bestfits[Not(1)]
elseif param_label == "H0"
    MCMC_chains = MCMC_chains[:, Not(3)]
    MLE_bestfits = MLE_bestfits[Not(3)]
elseif param_label == "Omegam"
    MCMC_chains = MCMC_chains[:, Not(5)]
    MLE_bestfits = MLE_bestfits[Not(5)]
elseif param_label == "w0"
    MCMC_chains = MCMC_chains[:, Not(6)]
    MLE_bestfits = MLE_bestfits[Not(6)]
elseif param_label == "wa"
    MCMC_chains = MCMC_chains[:, Not(7)]
    MLE_bestfits = MLE_bestfits[Not(7)]
end
cov_mat = cov(MCMC_chains)
step_sizes = 3*sqrt.(diag(cov_mat)) # only goes 2x of chain width since already have access to global MLE parameter values (already very close)
precondition_mat = inv(cov_mat)
means = MLE_bestfits

ncosmo = length(cosmo_fit_labels)
cosmo_means = means[1:ncosmo]
cosmo_step_sizes = step_sizes[1:ncosmo]
eft_means = means[ncosmo+1:end]
eft_step_sizes = step_sizes[ncosmo+1:end]
if dataset in ["FS", "FS+BAO"]
    cosmo_bounds = [cosmo_ranges_FS_BAO[label] for label in cosmo_fit_labels]
elseif dataset in ["FS+BAO+CMB", "FS+BAO+CMB+DESY5SN", "FS+BAO+CMB+PantheonPlusSN", "FS+BAO+CMB+Union3SN"]
    cosmo_bounds = [cosmo_ranges_CMB[label] for label in cosmo_fit_labels]
end
eft_bounds = [eft_ranges[label] for label in eft_fit_labels]


MLE_param_estimates = SharedArray{Float64}(n_runs, n_fit_params)
MLE_likelihood_estimates = SharedArray{Float64}(n_runs)
for i in 1:n_runs
    try
        # uses the actual MLE as the first guess and then does random ones after this
        if i == 1
            init_guesses_cosmo = cosmo_means
            init_guesses_eft = eft_means
        else
            init_guesses_cosmo = [rand(Truncated(Normal(cosmo_means[cosmo_ind], cosmo_step_sizes[cosmo_ind]),
                                  cosmo_bounds[cosmo_ind][1], cosmo_bounds[cosmo_ind][2])) for cosmo_ind in 1:length(cosmo_means)]
            init_guesses_eft = [rand(Truncated(Normal(eft_means[eft_ind], eft_step_sizes[eft_ind]),
                                eft_bounds[eft_ind][1], eft_bounds[eft_ind][2])) for eft_ind in 1:length(eft_means)]
        end
        init_guesses_all = vcat(init_guesses_cosmo, init_guesses_eft)
        @time fit_result = maximum_a_posteriori(fit_model, LBFGS(m=50, P=precondition_mat); initial_params=init_guesses_all)
        MLE_likelihood_estimates[i] = fit_result.lp
        MLE_param_estimates[i, :] = fit_result.values.array
        println("minimization okay")
    catch e
        println("minimization FAILED")
        println(e)
    end
end

npzwrite(save_dir * "$(param_label)_$(param_lower)_$(param_upper)_$(n_profile)_$(param_index)_$(dataset)_$(variation)_MLE_params.npy", MLE_param_estimates)
npzwrite(save_dir * "$(param_label)_$(param_lower)_$(param_upper)_$(n_profile)_$(param_index)_$(dataset)_$(variation)_MLE_likelihood.npy", MLE_likelihood_estimates)