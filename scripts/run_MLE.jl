using Pkg
Pkg.activate(".")
using ArgParse
using LinearAlgebra
using Statistics
using Turing
using Optim
using NPZ
using SharedArrays
include("likelihoods_no_EFTpriors.jl")


# Specifies dataset, cosmological model and details about MLE
config = ArgParseSettings()
@add_arg_table config begin
    "--n_runs"
    help="Specify number of independent optimizations (different starting guesses)"
    arg_type=Int64
    required = true
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
    "--MAP_path"
    help="Specify the path to the file containing the MAP bestfits (for initial guesses)"
    arg_type=String 
    required=true
    "--save_dir"
    help="Specify the path to the save directory"
    arg_type=String
    required=true
end
parsed_args = parse_args(config)
n_runs = parsed_args["n_runs"]
dataset = parsed_args["dataset"]
variation = parsed_args["variation"]
chains_path = parsed_args["chains_path"]
MAP_path = parsed_args["MAP_path"]
save_dir = parsed_args["save_dir"]

# Identifies the model based on dataset/variation
if dataset == "FS"
    if variation == "LCDM"
        model = model_FS_freq(D_FS_dict) | (w0=-1, wa=0)
        n_fit_params = 5 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        model = model_FS_freq(D_FS_dict)
        n_fit_params = 7 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "BAO"
    if variation == "LCDM"
        model = model_BAO_freq(D_BAO_dict, D_Lya) | (w0=-1, wa=0)
        n_fit_params = 3
        cosmo_fit_labels = ["H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        model = model_BAO_freq(D_BAO_dict, D_Lya)
        n_fit_params = 5
        cosmo_fit_labels = ["H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "FS+BAO"
    if variation == "LCDM"
        model = model_FS_BAO_freq(D_FS_BAO_dict, D_Lya) | (w0=-1, wa=0)
        n_fit_params = 5 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc"]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_freq(D_FS_BAO_dict, D_Lya)
        n_fit_params = 7 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    end
elseif dataset == "FS+BAO+CMB"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_freq(D_FS_BAO_dict, D_Lya, D_CMB) | (w0=-1, wa=0)
        n_fit_params = 7 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ"]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_freq(D_FS_BAO_dict, D_Lya, D_CMB)
        n_fit_params = 9 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ"]
    end
elseif dataset == "FS+BAO+CMB+DESY5SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (w0=-1, wa=0)
        n_fit_params = 8 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_D5"]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN")
        n_fit_params = 10 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_D5"]
    end
elseif dataset == "FS+BAO+CMB+Union3SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (w0=-1, wa=0)
        n_fit_params = 8 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_U3"]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN")
        n_fit_params = 10 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_U3"]
    end
elseif dataset == "FS+BAO+CMB+PantheonPlusSN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (w0=-1, wa=0)
        n_fit_params = 8 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb_PP"]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_freq(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN")
        n_fit_params = 10 + 7*size(tracer_vector)[1]
        cosmo_fit_labels = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa", "τ", "yₚ", "Mb_PP"]
    end
end
eft_fit_labels = []
if dataset != "BAO" # only exception is BAO only which has no EFT nuisance parameters
    for tracer in tracer_vector
        append!(eft_fit_labels, ["b1p_$(tracer)", "b2p_$(tracer)", "bsp_$(tracer)", "alpha0p_$(tracer)", "alpha2p_$(tracer)", "st0p_$(tracer)", "st2p_$(tracer)"])
    end
end

# Reads in the file storing the MCMC chains in order to set the preconditioning matrix
MCMC_chains = npzread(chains_path)
cov_mat = cov(MCMC_chains)
precondition_mat = inv(cov_mat)
# Reads in the file storing MAP bestfits to use as reference point for initial guess distribution
MAP_bestfits = npzread(MAP_path)
step_sizes = 3*sqrt.(diag(cov_mat))
ncosmo = length(cosmo_fit_labels)
cosmo_means = MAP_bestfits[1:ncosmo]
cosmo_step_sizes = step_sizes[1:ncosmo]
if dataset != "BAO"
    eft_means = MAP_bestfits[ncosmo+1:end]
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

npzwrite(save_dir * "$(dataset)_$(variation)_$(n_runs)_MLE_params.npy", MLE_param_estimates)
npzwrite(save_dir * "$(dataset)_$(variation)_$(n_runs)_MLE_likelihood.npy", MLE_likelihood_estimates)
