using Pkg
Pkg.activate(".")
using ArgParse
using LinearAlgebra
using Statistics
using Turing
using Optim
using NPZ
using SharedArrays
include("likelihoods_EFTpriors.jl")


# Specifies dataset, cosmological model and details about MAP
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
save_dir = parsed_args["save_dir"]

# Identifies the model based on dataset/variation
if dataset == "FS"
    if variation == "LCDM"
        model = model_FS_bay(D_FS_dict) | (w0=-1, wa=0)
        n_fit_params = 5 + 7*size(tracer_vector)[1]
    elseif variation == "w0waCDM"
        model = model_FS_bay(D_FS_dict)
        n_fit_params = 7 + 7*size(tracer_vector)[1]
    end
elseif dataset == "BAO"
    if variation == "LCDM"
        model = model_BAO_bay(D_BAO_dict, D_Lya) | (w0=-1, wa=0)
        n_fit_params = 3
    elseif variation == "w0waCDM"
        model = model_BAO_bay(D_BAO_dict, D_Lya)
        n_fit_params = 5
    end
elseif dataset == "FS+BAO"
    if variation == "LCDM"
        model = model_FS_BAO_bay(D_FS_BAO_dict, D_Lya) | (w0=-1, wa=0)
        n_fit_params = 5 + 7*size(tracer_vector)[1]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_bay(D_FS_BAO_dict, D_Lya)
        n_fit_params = 7 + 7*size(tracer_vector)[1]
    end
elseif dataset == "FS+BAO+CMB"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_bay(D_FS_BAO_dict, D_Lya, D_CMB) | (w0=-1, wa=0)
        n_fit_params = 7 + 7*size(tracer_vector)[1]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_bay(D_FS_BAO_dict, D_Lya, D_CMB)
        n_fit_params = 9 + 7*size(tracer_vector)[1]
    end
elseif dataset == "FS+BAO+CMB+DESY5SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (w0=-1, wa=0)
        n_fit_params = 8 + 7*size(tracer_vector)[1]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN")
        n_fit_params = 10 + 7*size(tracer_vector)[1]
    end
elseif dataset == "FS+BAO+CMB+Union3SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (w0=-1, wa=0)
        n_fit_params = 8 + 7*size(tracer_vector)[1]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN")
        n_fit_params = 10 + 7*size(tracer_vector)[1]
    end
elseif dataset == "FS+BAO+CMB+PantheonPlusSN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (w0=-1, wa=0)
        n_fit_params = 8 + 7*size(tracer_vector)[1]
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN")
        n_fit_params = 10 + 7*size(tracer_vector)[1]
    end
end

# Reads in the file storing the MCMC chains in order to set the preconditioning matrix
MCMC_chains = npzread(chains_path)
cov_mat = cov(MCMC_chains)
precondition_mat = inv(cov_mat)

# Performs all the minimizations
MAP_param_estimates = SharedArray{Float64}(n_runs, n_fit_params)
MAP_posterior_estimates = SharedArray{Float64}(n_runs)
for i in 1:n_runs
    try
        @time fit_result = maximum_a_posteriori(model, LBFGS(m=50, P=precondition_mat))
        MAP_posterior_estimates[i] = fit_result.lp
        MAP_param_estimates[i, :] = fit_result.values.array
        println("okay")
    catch e
        println("FAILED!")
        println(e)
    end
end

npzwrite(save_dir * "$(dataset)_$(variation)_$(n_runs)_MAP_params.npy", MAP_param_estimates)
npzwrite(save_dir * "$(dataset)_$(variation)_$(n_runs)_MAP_posterior.npy", MAP_posterior_estimates)
