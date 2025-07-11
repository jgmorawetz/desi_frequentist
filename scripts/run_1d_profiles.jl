using Pkg
Pkg.activate(".")
using ArgParse
using LinearAlgebra
using Statistics
using Turing
using Optim
using NPZ
using SharedArrays
using InvertedIndices
include("likelihoods_no_EFTpriors.jl")


# Specifies dataset, cosmological model and details about MLE
config = ArgParseSettings()
@add_arg_table config begin
    "--n_runs"
    help="Specify number of independent optimizations (different starting guesses)"
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
    help="Specify the path to the file containing the MCMC chains (for preconditioning)"
    arg_type=String
    required=true
    "--MLE_path"
    help="Specify the path to the file containing the MLE best fits (for initial guess)"
    "--save_dir"
    help="Specify the path to the save directory"
    arg_type=String
    required=true
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
save_dir = parsed_args["save_dir"]
all_params = LinRange(param_lower, param_upper, n_profile)
param_value = all_params[param_index]


# Identifies the model based on dataset/variation

if variation == "LCDM"
    if param_label == "sigma8"
        FS_model = model_FS_freq_sigma8(D_FS_dict) | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_model = model_FS_BAO_freq_sigma8(D_FS_BAO_dict, D_Lya) | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB) | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (sigma8=param_value, w0=-1, wa=0)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (sigma8=param_value, w0=-1, wa=0)
    elseif param_label == "H0"
        FS_model = model_FS_freq(D_FS_dict) | (H0=param_value, w0=-1, wa=0)
        BAO_model = model_BAO_freq(D_BAO_dict, D_Lya) | (H0=param_value, w0=-1, wa=0)
        FS_BAO_model = model_FS_BAO_freq(D_FS_BAO_dict, D_Lya) | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq(D_FS_BAO_dict, D_Lya, D_CMB) | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (H0=param_value, w0=-1, wa=0)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (H0=param_value, w0=-1, wa=0)
    elseif param_label == "Omegam"
        FS_model = model_FS_freq_Omegam(D_FS_dict) | (Om=param_value, w0=-1, wa=0)
        BAO_model = model_BAO_freq_Omegam(D_BAO_dict, D_Lya) | (Om=param_value, w0=-1, wa=0)
        FS_BAO_model = model_FS_BAO_freq_Omegam(D_FS_BAO_dict, D_Lya) | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB) | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (Om=param_value, w0=-1, wa=0)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (Om=param_value, w0=-1, wa=0)
    end
elseif variation == "w0waCDM"
    if param_label == "sigma8"
        FS_model = model_FS_freq_sigma8(D_FS_dict) | (sigma8=param_value,)
        FS_BAO_model = model_FS_BAO_freq_sigma8(D_FS_BAO_dict, D_Lya) | (sigma8=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB) | (sigma8=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (sigma8=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (sigma8=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq_sigma8(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (sigma8=param_value,)
    elseif param_label == "H0"
        FS_model = model_FS_freq(D_FS_dict) | (H0=param_value,)
        BAO_model = model_BAO_freq(D_BAO_dict, D_Lya) | (H0=param_value,)
        FS_BAO_model = model_FS_BAO_freq(D_FS_BAO_dict, D_Lya) | (H0=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq(D_FS_BAO_dict, D_Lya, D_CMB) | (H0=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (H0=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (H0=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (H0=param_value,)
    elseif param_label == "Omegam"
        FS_model = model_FS_freq_Omegam(D_FS_dict) | (Om=param_value,)
        BAO_model = model_BAO_freq_Omegam(D_BAO_dict, D_Lya) | (Om=param_value,)
        FS_BAO_model = model_FS_BAO_freq_Omegam(D_FS_BAO_dict, D_Lya) | (Om=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB) | (Om=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (Om=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (Om=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq_Omegam(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (Om=param_value,)
    elseif param_label == "w0"
        FS_model = model_FS_freq(D_FS_dict) | (w0=param_value,)
        BAO_model = model_BAO_freq(D_BAO_dict, D_Lya) | (w0=param_value,)
        FS_BAO_model = model_FS_BAO_freq(D_FS_BAO_dict, D_Lya) | (w0=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq(D_FS_BAO_dict, D_Lya, D_CMB) | (w0=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (w0=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (w0=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (w0=param_value,)
    elseif param_label == "wa"
        FS_model = model_FS_freq(D_FS_dict) | (wa=param_value,)
        BAO_model = model_BAO_freq(D_BAO_dict, D_Lya) | (wa=param_value,)
        FS_BAO_model = model_FS_BAO_freq(D_FS_BAO_dict, D_Lya) | (wa=param_value,)
        FS_BAO_CMB_model = model_FS_BAO_CMB_freq(D_FS_BAO_dict, D_Lya, D_CMB) | (wa=param_value,)
        FS_BAO_CMB_DESY5SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (wa=param_value,)
        FS_BAO_CMB_Union3SN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (wa=param_value,)
        FS_BAO_CMB_PantheonPlusSN_model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (wa=param_value,)
    end
end


if dataset == "FS"
    model = FS_model
    if variation == "LCDM"
        n_fit_params = 4 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        n_fit_params = 6 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "BAO"
    model = BAO_model
    if variation == "LCDM"
        n_fit_params = 2
        cosmo_fit_labels = ["H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        n_fit_params = 4
        cosmo_fit_labels = ["H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "FS+BAO"
    model = FS_BAO_model
    if variation == "LCDM"
        n_fit_params = 4 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        n_fit_params = 6 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "FS+BAO+CMB"
    model = FS_BAO_CMB_model
    if variation == "LCDM"
        n_fit_params = 6 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ"]
    elseif variation == "w0waCDM"
        n_fit_params = 8 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ"]
    end
elseif dataset == "FS+BAO+CMB+DESY5SN"
    model = FS_BAO_CMB_DESY5SN_model
    if variation == "LCDM"
        n_fit_params = 7 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_D5"]
    elseif variation == "w0waCDM"
        n_fit_params = 9 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_D5"]
    end
elseif dataset == "FS+BAO+CMB+Union3SN"
    model = FS_BAO_CMB_Union3SN_model
    if variation == "LCDM"
        n_fit_params = 7 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_U3"]
    elseif variation == "w0waCDM"
        n_fit_params = 9 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_U3"]
    end
elseif dataset == "FS+BAO+CMB+PantheonPlusSN"
    model = FS_BAO_CMB_PantheonPlusSN_model
    if variation == "LCDM"
        n_fit_params = 7 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_PP"]
    elseif variation == "w0waCDM"
        n_fit_params = 9 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_PP"]
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
if dataset != "BAO" # only exception is BAO only which has no EFT nuisance parameters
    for tracer in tracer_vector
        append!(eft_fit_labels, ["b1p_$(tracer)", "b2p_$(tracer)", "bsp_$(tracer)", "alpha0p_$(tracer)", "alpha2p_$(tracer)", "st0p_$(tracer)", "st2p_$(tracer)"])
    end
end

# Reads in the file storing the MCMC chains in order to set the preconditioning matrix
MCMC_chains = npzread(chains_path)
MLE_bestfits = npzread(MLE_path)
if dataset != "BAO"
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
elseif dataset == "BAO"
    if param_label == "H0"
        MCMC_chains = MCMC_chains[:, Not(1)]
        MLE_bestfits = MLE_bestfits[Not(1)]
    elseif param_label == "Omegam"
        MCMC_chains = MCMC_chains[:, Not(3)]
        MLE_bestfits = MLE_bestfits[Not(3)]
    elseif param_label == "w0"
        MCMC_chains = MCMC_chains[:, Not(4)]
        MLE_bestfits = MLE_bestfits[Not(4)]
    elseif param_label == "wa"
        MCMC_chains = MCMC_chains[:, Not(5)]
        MLE_bestfits = MLE_bestfits[Not(5)]
end 
cov_mat = cov(MCMC_chains)
precondition_mat = inv(cov_mat)
step_sizes = 3*sqrt.(diag(cov_mat))
ncosmo = length(cosmo_fit_labels)
cosmo_means = MLE_bestfits[1:ncosmo]
cosmo_step_sizes = step_sizes[1:ncosmo]
if dataset != "BAO"
    eft_means = MLE_bestfits[ncosmo+1:end]
    eft_step_sizes = step_sizes[ncosmo+1:end]
    eft_bounds = [eft_ranges[label] for label in eft_fit_labels]
end
if dataset in ["FS", "BAO", "FS+BAO"]
    cosmo_bounds = [cosmo_ranges_FS_BAO[label] for label in cosmo_fit_labels]
elseif dataset in ["FS+BAO+CMB", "FS+BAO+CMB+DESY5SN", "FS+BAO+CMB+PantheonPlusSN", "FS+BAO+CMB+Union3SN"]
    cosmo_bounds = [cosmo_ranges_CMB[label] for label in cosmo_fit_labels]
end

# Performs all the minimizations
MLE_param_estimates = SharedArray{Float64}(n_runs, n_fit_params)
MLE_likelihood_estimates = SharedArray{Float64}(n_runs)
for i in 1:n_runs
    try
        if i == 1
            init_guesses_cosmo = cosmo_means
            if dataset != "BAO"
                init_guesses_eft = eft_means
            end
        else
            init_guesses_cosmo = [rand(Truncated(Normal(cosmo_means[cosmo_ind], cosmo_step_sizes[cosmo_ind]),
                                  cosmo_bounds[cosmo_ind][1], cosmo_bounds[cosmo_ind][2])) for cosmo_ind in 1:length(cosmo_means)]
            if dataset != "BAO"
                init_guesses_eft = [rand(Truncated(Normal(eft_means[eft_ind], eft_step_sizes[eft_ind]),
                                    eft_bounds[eft_ind][1], eft_bounds[eft_ind][2])) for eft_ind in 1:length(eft_means)]
            end
        end
        if dataset != "BAO"
            init_guesses_all = vcat(init_guesses_cosmo, init_guesses_eft)
        elseif dataset == "BAO"
            init_guesses_all = init_guesses_cosmo
        end
        @time fit_result = maximum_a_posteriori(model, LBFGS(m=50, P=precondition_mat); initial_params=init_guesses_all)
        MLE_likelihood_estimates[i] = fit_result.lp
        MLE_param_estimates[i, :] = fit_result.values.array
        println("okay")
    catch e
        println("FAILED!")
        println(e)
    end
end

npzwrite(save_dir * "$(param_label)_$(param_lower)_$(param_upper)_$(n_profile)_$(param_index)_$(dataset)_$(variation)_$(n_runs)_MLE_params.npy", MLE_param_estimates)
npzwrite(save_dir * "$(param_label)_$(param_lower)_$(param_upper)_$(n_profile)_$(param_index)_$(dataset)_$(variation)_$(n_runs)_MLE_likelihood.npy", MLE_likelihood_estimates)
