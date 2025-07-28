using Pkg
Pkg.activate(".")
using ArgParse
using LinearAlgebra
using Turing
using NPZ
include("likelihoods_EFTpriors.jl")


# Specifies dataset, cosmological model and details about MCMC chains
config = ArgParseSettings()
@add_arg_table config begin
    "--n_steps"
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
n_steps = parsed_args["n_steps"]
n_burn = parsed_args["n_burn"]
acceptance = parsed_args["acceptance"]
chain_index = parsed_args["chain_index"]
dataset = parsed_args["dataset"]
variation = parsed_args["variation"]
save_dir = parsed_args["save_dir"]

# Identifies the model based on dataset/variation
if dataset == "FS"
    if variation == "LCDM"
        model = model_FS_bay(D_FS_dict) | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_bay(D_FS_dict)
    end
elseif dataset == "BAO"
    if variation == "LCDM"
        model = model_BAO_bay(D_BAO_dict, D_Lya) | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_BAO_bay(D_BAO_dict, D_Lya)
    end
elseif dataset == "FS+BAO"
    if variation == "LCDM"
        model = model_FS_BAO_bay(D_FS_BAO_dict, D_Lya) | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_bay(D_FS_BAO_dict, D_Lya)
    end
elseif dataset == "FS+BAO+CMB"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_bay(D_FS_BAO_dict, D_Lya, D_CMB) | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_bay(D_FS_BAO_dict, D_Lya, D_CMB)
    end
elseif dataset == "FS+BAO+CMB+DESY5SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_DESY5SN, D_DESY5SN, z_DESY5SN, "DESY5SN")
    end
elseif dataset == "FS+BAO+CMB+Union3SN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN")
    end
elseif dataset == "FS+BAO+CMB+PantheonPlusSN"
    if variation == "LCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_PantheonPlusSN, D_PantheonPlusSN, z_PantheonPlusSN, "PantheonPlusSN")
    end
elseif dataset == "BAO+CMB"
    if variation == "LCDM"
        model = model_BAO_CMB_bay(D_BAO_dict, D_Lya, D_CMB) | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_BAO_CMB_bay(D_BAO_dict, D_Lya, D_CMB)
    end
elseif dataset == "BAO+CMB+Union3SN"
    if variation == "LCDM"
        model = model_BAO_CMB_SN_bay(D_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN") | (w0=-1, wa=0)
    elseif variation == "w0waCDM"
        model = model_BAO_CMB_SN_bay(D_BAO_dict, D_Lya, D_CMB, iΓ_Union3SN, D_Union3SN, z_Union3SN, "Union3SN")
    end
end

# Runs the chain and saves to file
chain = sample(model, NUTS(n_burn, acceptance), n_steps)
chain_array = Array(chain)
npzwrite(save_dir * "$(dataset)_$(variation)_$(n_steps)_$(n_burn)_$(acceptance)_$(chain_index)_chain.npy", chain_array)