using Pkg
Pkg.activate(".")
using ArgParse
using LinearAlgebra
using Turing
using NPZ
include("model_likelihoods.jl")


# Specifies dataset, cosmological model and details about MCMC chains
config = ArgParseSettings()
@add_arg_table config begin
    "--n_step"
    help="Specify number of accepted steps (discarding burn in)"
    arg_type=Int64
    required=true
    "--n_burn"
    help="Specify the number of burn in steps"
    arg_type=Int64
    required=true
    "--acceptance"
    help="Specify the acceptance fraction"
    arg_type=Float64
    required=true
    "--chain_index"
    help="Specify the chain index (in case multiple chains are to be combined together)"
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
    "--save_dir"
    help="Specify the path to the save directory"
    arg_type=String
    required=true
end
parsed_args = parse_args(config)
n_step = parsed_args["n_step"]
n_burn = parsed_args["n_burn"]
acceptance = parsed_args["acceptance"]
chain_index = parsed_args["chain_index"]
dataset = parsed_args["dataset"]
variation = parsed_args["variation"]
save_dir = parsed_args["save_dir"]


# Uses all tracers
tracer_vector = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]


# Identifies the model based on dataset/variation
if dataset == "FS"
    if variation == "LCDM"
        model = model_FS(D_FS_dict, tracer_vector, "bay") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS(D_FS_dict, tracer_vector, "bay")
    end
elseif dataset == "FS+BAO"
    if variation == "LCDM"
        model = model_FS_BAO(D_FS_BAO_dict, D_Lya, tracer_vector, "bay") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO(D_FS_BAO_dict, D_Lya, tracer_vector, "bay")
    end
elseif dataset == "FS+BAO+CMB"
    if variation == "LCDM"
        model = model_FS_BAO_CMB(D_FS_BAO_dict, D_Lya, D_CMB, tracer_vector, "bay") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB(D_FS_BAO_dict, D_Lya, D_CMB, tracer_vector, "bay")
    end
elseif dataset == "FS+BAO+CMB+DESY5SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_DY5SN, iΓ_DY5SN, z_DY5SN, "DESY5", tracer_vector, "bay") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_DY5SN, iΓ_DY5SN, z_DY5SN, "DESY5", tracer_vector, "bay")
    end
elseif dataset == "FS+BAO+CMB+PantheonPlusSN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_PPSN, iΓ_PPSN, z_PPSN, "PantheonPlus", tracer_vector, "bay") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_PPSN, iΓ_PPSN, z_PPSN, "PantheonPlus", tracer_vector, "bay")
    end
elseif dataset == "FS+BAO+CMB+Union3SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_U3SN, iΓ_U3SN, z_U3SN, "Union3", tracer_vector, "bay") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_U3SN, iΓ_U3SN, z_U3SN, "Union3", tracer_vector, "bay")
    end
end


# Runs the chain and saves to file
chain = sample(model, NUTS(n_burn, acceptance), n_step)
chain_array = Array(chain)
npzwrite(save_dir * "$(dataset)_$(variation)_$(n_step)_$(n_burn)_$(acceptance)_$(chain_index)_chain.npy", chain_array)