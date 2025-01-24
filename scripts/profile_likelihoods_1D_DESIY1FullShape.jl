
using Pkg
Pkg.activate("/global/homes/j/jgmorawe/FrequentistExample")
using ArgParse
using Distributed

# The number of parallel processes to run for each profile likelihood
n_bins = 3#########
addprocs(n_bins)
# The number of independent minimization runs to perform to reach global minimum
n_runs = 1

@everywhere begin
    using Statistics
    using Random
    using Distributions
    using SimpleChains
    using Static
    using NPZ
    using ForwardDiff
    using LinearAlgebra
    using Turing
    using Optim
    using DelimitedFiles
    using Printf
    using Effort
    using Capse
    using PlanckLite
    using SharedArrays
    using OptimizationOptimJL: ParticleSwarm
end

function main()

    # Reads in the necessary arguments to run script 
    config = ArgParseSettings()
    @add_arg_table config begin
        "--param" 
        help = "Specify the parameter" # e.g. ln10As, H0, Om, w0, wa, etc.
        arg_type = String
        required = true
        "--dataset" 
        help = "Specify the desired dataset" # e.g. FS, FS+BAO, FS+CMB, FS+BAO+CMB, FS+BAO+CMB+SN, etc.
        arg_type = String
        required = true
        "--variation"
        help = "Specify the desired variation" # e.g. LCDM, w0waCDM
        arg_type = String
        required = true
        "--tracer_list" 
        help = "Specify the tracer(s)" # e.g. BGS, LRG1, LRG2, LRG3, ELG2, QSO, BGS,LRG1,LRG2,LRG3,ELG2,QSO, etc.
        arg_type = String
        required = true
        "--lower"
        help = "Specify the lower parameter bound"
        arg_type = Float64
        required = true
        "--upper"
        help = "Specify the upper parameter bound"
        arg_type = Float64
        required = true
        "--init_values"
        help = "Specify the initial guesses for the other parameters"
        arg_type = String
        required = true
        "--desi_data_dir"
        help = "Specify the DESI data directory"
        arg_type = String
        required = true
        "--FS_emu_dir"
        help = "Specify the full-shape emulator directory"
        arg_type = String
        required = true
        "--BAO_emu_dir"
        help = "Specify the BAO emulator directory"
        arg_type = String
        required = true
        "--CMB_emu_dir"
        help = "Specify the CMB emulator directory"
        arg_type = String
        required = true
    end

    # Parses the arguments
    parsed_args = parse_args(config)
    param = parsed_args["param"]
    dataset = parsed_args["dataset"]
    variation = parsed_args["variation"]
    tracer_list = parsed_args["tracer_list"]
    tracer_vector = Vector{String}(split(tracer_list, ","))
    save_path = "/global/homes/j/jgmorawe/FrequentistExample/profile_likelihood_results/$(param)_$(dataset)_$(variation)_$(tracer_list)"
    lower = parsed_args["lower"]
    upper = parsed_args["upper"]
    init_values = parse.(Float64, split(parsed_args["init_values"], ","))
    desi_data_dir = parsed_args["desi_data_dir"]
    FS_emu_dir = parsed_args["FS_emu_dir"]
    BAO_emu_dir = parsed_args["BAO_emu_dir"]
    CMB_emu_dir = parsed_args["CMB_emu_dir"]
    println("parsed arguments")

    # Retrieves the relevant data/information associated with all of the tracers
    tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
    redshift_labels = ["z0.1-0.4", "z0.4-0.6", "z0.6-0.8", "z0.8-1.1", "z1.1-1.6", "z0.8-2.1"]
    redshift_eff = vec(readdlm(desi_data_dir * "zeff_bao-post.txt", ' '))
    redshift_indices = [1, 2, 3, 4, 6, 7]
    zrange_all = Dict(zip(tracers, redshift_labels))
    zeff_all = Dict(zip(tracers, redshift_eff))
    zindex_all = Dict(zip(tracers, redshift_indices))

    # Reads in the DESI full-shape and BAO data
    pk_paths = Dict(tracer => desi_data_dir * "pk_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
    baopost_paths = Dict(tracer => desi_data_dir * "bao-post_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
    kin_paths = Dict(tracer => desi_data_dir * "kin_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
    wmat_paths = Dict(tracer => desi_data_dir * "wmatrix_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
    invcov_pk_paths = Dict(tracer => desi_data_dir * "invcov_pk_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
    invcov_pk_baopost_paths = Dict(tracer => desi_data_dir * "invcov_pk_bao-post_" * tracer * "_" * zrange_all[tracer] * ".txt" for tracer in tracers)
    pk_all = Dict(tracer => vec(readdlm(pk_paths[tracer], ' ')) for tracer in tracers) # reads in the DESI data
    baopost_all = Dict(tracer => vec(readdlm(baopost_paths[tracer], ' ')) for tracer in tracers)
    pk_baopost_all = Dict(tracer => vcat(pk_all[tracer], baopost_all[tracer]) for tracer in tracers)
    kin_all = Dict(tracer => vec(readdlm(kin_paths[tracer], ' ')) for tracer in tracers)
    wmat_all = Dict(tracer => readdlm(wmat_paths[tracer], ' ') for tracer in tracers)
    invcov_pk_all = Dict(tracer => readdlm(invcov_pk_paths[tracer], ' ') for tracer in tracers)
    invcov_pk_baopost_all = Dict(tracer => readdlm(invcov_pk_baopost_paths[tracer], ' ') for tracer in tracers)
    cov_pk_all = Dict(tracer => inv(invcov_pk_all[tracer]) for tracer in tracers) # inverts the covariance matrices
    cov_pk_baopost_all = Dict(tracer => inv(invcov_pk_baopost_all[tracer]) for tracer in tracers)

    # Reads in the Plancklite CMB data
    CMB_cov_sqrt = sqrt(PlanckLite.cov)
    inv_CMB_cov_sqrt = inv(CMB_cov_sqrt)
    CMB_data = inv_CMB_cov_sqrt * PlanckLite.data # reparameterization to make computation more efficient

    # Reads in the emulators associated with the DESI full-shape/BAO data
    mono_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/0/" for tracer in tracers)
    quad_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/2/" for tracer in tracers)
    hexa_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/4/" for tracer in tracers)
    FS_emus = Dict(tracer => [Effort.load_multipole_noise_emulator(mono_paths[tracer]),
                              Effort.load_multipole_noise_emulator(quad_paths[tracer]),
                              Effort.load_multipole_noise_emulator(hexa_paths[tracer])] for tracer in tracers)
    BAO_emu = Effort.load_BAO_emulator(BAO_emu_dir)

    # Reads in the emulators associated with the Plancklite CMB data 
    CℓTT_emu = Capse.load_emulator(CMB_emu_dir * "/TT/")
    CℓTE_emu = Capse.load_emulator(CMB_emu_dir * "/TE/")
    CℓEE_emu = Capse.load_emulator(CMB_emu_dir * "/EE/")
    CMB_emu = [CℓTT_emu, CℓTE_emu, CℓEE_emu]
    
    # Additional parameters needed for EFT basis change
    nd_all = Dict("BGS" => 5e-4, "LRG1" => 5e-4, "LRG2" => 5e-4, "LRG3" => 3e-4, "ELG2" => 5e-4, "QSO" => 3e-5)
    fsat_all = Dict("BGS" => 0.15, "LRG1" => 0.15, "LRG2" => 0.15, "LRG3" => 0.15, "ELG2" => 0.10, "QSO" => 0.03)
    sigv_all = Dict("BGS" => 150*(10)^(1/3)*(1+0.2)^(1/2)/70, 
                    "LRG1" => 150*(10)^(1/3)*(1+0.8)^(1/2)/70, 
                    "LRG2" => 150*(10)^(1/3)*(1+0.8)^(1/2)/70, 
                    "LRG3" => 150*(10)^(1/3)*(1+0.8)^(1/2)/70, 
                    "ELG2" => 150*2.1^(1/2)/70, 
                    "QSO" => 150*(10)^(0.7/3)*(2.4)^(1/2)/70)
    println("read in data and emulators")

    # Distributes the variables among the processes 
    @everywhere n_runs = $n_runs
    @everywhere param = $param
    @everywhere dataset = $dataset
    @everywhere variation = $variation
    @everywhere tracer_vector = $tracer_vector
    @everywhere pk_all = $pk_all
    @everywhere pk_baopost_all = $pk_baopost_all
    @everywhere cov_pk_all = $cov_pk_all
    @everywhere cov_pk_baopost_all = $cov_pk_baopost_all
    @everywhere inv_CMB_cov_sqrt = $inv_CMB_cov_sqrt
    @everywhere CMB_data = $CMB_data
    @everywhere FS_emus = $FS_emus
    @everywhere BAO_emu = $BAO_emu
    #@everywhere CℓTT_emu = $(CℓTT_emu)###########################################################
    #println("ClTT_emu\n")
    #@everywhere CMB_emu = $CMB_emu
    #println("CMB_emu\n")
    @everywhere kin_all = $kin_all
    @everywhere zeff_all = $zeff_all
    @everywhere wmat_all = $wmat_all
    @everywhere init_values = $init_values
    @everywhere nd_all = $nd_all
    @everywhere fsat_all = $fsat_all
    @everywhere sigv_all = $sigv_all
    println("distribute among processes")
    # Specifies the range of parameter values and initiates arrays to store
    param_values = range(lower, stop=upper, length=n_bins)
    profile_values = SharedArray{Float64}(n_bins, n_runs)
    if dataset in ["FS", "FS+BAO"]
        if variation == "LCDM"
            ncosmo = 5
        elseif variation == "w0waCDM"
            ncosmo = 7
        end
    elseif dataset in ["FS+CMB", "FS+BAO+CMB", "FS+BAO+CMB+SN"] # extra two parameters for CMB beyond FS and BAO
        if variation == "LCDM"
            ncosmo = 7
        elseif variation == "w0waCDM"
            ncosmo = 9
        end
    end
    bestfit_values = SharedArray{Float64}(n_bins, (ncosmo-1)+7*size(tracer_vector)[1], n_runs)
    println("initiate arrays")
    # Retrieves the specific dataset based on the desired combination
    if dataset in ["FS", "FS+CMB"]
        FS_BAO_data_vecs = pk_all
        FS_BAO_cov_mats = cov_pk_all
    elseif dataset in ["FS+BAO", "FS+BAO+CMB", "FS+BAO+CMB+SN"]
        FS_BAO_data_vecs = pk_baopost_all
        FS_BAO_cov_mats = cov_pk_baopost_all
    end
    print("starting parallelization")
    # Runs the parallel processes and saves results to file
    @sync @distributed for index in 1:length(param_values)
        @time (profile_values[index, :], bestfit_values[index, :, :]) = run_worker(param_values[index], param, dataset, variation, FS_BAO_data_vecs, CMB_data, FS_BAO_cov_mats, inv_CMB_cov_sqrt, FS_emus, BAO_emu, nothing, tracer_vector, kin_all, zeff_all, wmat_all, nd_all, fsat_all, sigv_all)#CMB_emu, tracer_vector, kin_all, zeff_all, wmat_all, nd_all, fsat_all, sigv_all)
    end  
    npzwrite(string(save_path, "_param_values.npy"), param_values)
    npzwrite(string(save_path, "_profile_values.npy"), profile_values)
    npzwrite(string(save_path, "_bestfit_values.npy"), bestfit_values)                                                                          
end

@everywhere function theory_FS(theta_FS, emu_FS_components, kin)
    """
    Outputs the full-shape theory vector.
        theta_FS -> The full vector of cosmological and EFT parameters.
        emu_FS_components -> List of the three components for the emulator (monopole/quadrupole/hexadecapole).
        kin -> The input k vector.
    """
    cosmo_params = theta_FS[1:7] # 7 cosmological parameters: ln10As, ns, H0, omega_b, omega_c, w0, wa
    eft_params = theta_FS[8:18] # 11 EFT parameters: b1, b2, b3, bs, alpha0, alpha2, alpha4, alpha6, st0, st2, st4
    mono_emu = emu_FS_components[1] # emulators listed in order mono, quad, hexa
    quad_emu = emu_FS_components[2]
    hexa_emu = emu_FS_components[3]
    pk0 = Effort.get_Pℓ(cosmo_params, eft_params, mono_emu)
    pk2 = Effort.get_Pℓ(cosmo_params, eft_params, quad_emu)
    pk4 = Effort.get_Pℓ(cosmo_params, eft_params, hexa_emu)
    pks_all = vcat(Effort._akima_spline(pk0, mono_emu.Pℓ.P11.kgrid, kin),
                   Effort._akima_spline(pk2, quad_emu.Pℓ.P11.kgrid, kin),
                   Effort._akima_spline(pk4, hexa_emu.Pℓ.P11.kgrid, kin))
    return pks_all
end

@everywhere function theory_BAO(theta_BAO, emu_BAO, zeff, tracer)
    """
    Outputs the post-recon BAO theory vector (concatenated alpha parallel, alpha perpendicular)
    given the parameters, emulator and effective redshift.
        theta_BAO -> The vector of cosmological parameters.
        emu_BAO -> The BAO emulator.
        zeff -> The effective redshift for the given tracer calculation.
        tracer -> The particular tracer (since computes alpha_iso only for BGS/QSO but [alpha_par, alpha_perp] for LRG123/ELG2).
    """
    # Fiducial cosmology (Planck 2018 cosmology)
    ln10As_fid = 3.044; ns_fid = 0.9649; H0_fid = 67.36; ωb_fid = 0.02237; ωc_fid = 0.1200; w0_fid = -1; wa_fid = 0
    theta_BAO_fid = [ln10As_fid, ns_fid, H0_fid, ωb_fid, ωc_fid, w0_fid, wa_fid]
    # Converts both fiducial and true to format needed for H(z) and DA(z) calculations
    Ωcb_fid = (theta_BAO_fid[4]+theta_BAO_fid[5])/(theta_BAO_fid[3]/100)^2; h_fid = theta_BAO_fid[3]/100
    Ωcb_true = (theta_BAO[4]+theta_BAO[5])/(theta_BAO[3]/100)^2; h_true = theta_BAO[3]/100; w0_true = theta_BAO[6]; wa_true = theta_BAO[7]
    mν_fixed = 0.06
    # Computes H(z) and DA(z) associated with each
   # print(zeff, Ωcb_true, h_true, mν_fixed, w0_true, wa_true)######################################
    #print(zeff, Ωcb_fid, h_fid, mν_fixed, w0_fid, wa_fid)##########################################
    E_fid = Effort._E_z(zeff, Ωcb_fid, h_fid, mν=mν_fixed, w0=w0_fid, wa=wa_fid)
    E_true = Effort._E_z(zeff, Ωcb_true, h_true, mν=mν_fixed, w0=w0_true, wa=wa_true)
    DA_fid = Effort._d̃A_z(zeff, Ωcb_fid, h_fid, mν=mν_fixed, w0=w0_fid, wa=wa_fid)
    DA_true = Effort._d̃A_z(zeff, Ωcb_true, h_true, mν=mν_fixed, w0=w0_true, wa=wa_true)
    # Computes rsdrag from emulator
    rsdrag_fid = Effort.get_BAO(theta_BAO_fid, emu_BAO)[1] # rsdrag is first entry in this vector
    rsdrag_true = Effort.get_BAO(theta_BAO, emu_BAO)[1]
    # Converts to alpha par and perp components
    alpha_par = (E_fid * rsdrag_fid) / (E_true * rsdrag_true)
    alpha_perp = (DA_true * rsdrag_fid) / (DA_fid * rsdrag_true)
    alpha_iso = (alpha_par * alpha_perp^2)^(1/3)
    if tracer in ["LRG1", "LRG2", "LRG3", "ELG2"]
        return [alpha_par, alpha_perp]
    elseif tracer in ["BGS", "QSO"]
        return [alpha_iso]
    end
end

@everywhere function theory_CMB(theta_CMB, emu_CMB)
    """
    Outputs the CMB data vector given the parameters and emulators.
        theta_CMB -> The vector of cosmological parameters associated with the CMB specifically (ln10As, ns, H0, omega_b, omega_c, tau, mnu, w0, wa)
        emu_CMB -> List of the different components for the emulator (TT/TE/EE).
    """
    TT_emu = emu_CMB[1] # emulators in list with order TT, TE, EE
    TE_emu = emu_CMB[2]
    EE_emu = emu_CMB[3]
    lsTT = 2:2508
    lsTE = 2:1996
    facTT=lsTT.*(lsTT.+1)./(2*π)
    facTE=lsTE.*(lsTE.+1)./(2*π)
    return PlanckLite.bin_Cℓ(Capse.get_Cℓ(theta_CMB, TT_emu)[1:2507]./facTT,
                             Capse.get_Cℓ(theta_CMB, TE_emu)[1:1995]./facTE,
                             Capse.get_Cℓ(theta_CMB, EE_emu)[1:1995]./facTE)
end

@everywhere @model function model(fixed_value, param, dataset, variation, FS_BAO_data_vecs, CMB_data, FS_BAO_cov_mats, inv_CMB_cov_sqrt, emus_FS, emu_BAO, emu_CMB, tracers_used, kin_used, zeff_used, wmat_used, nd_used, fsat_used, sigv_used)
    """
    Probalistic model function for the joint fit of data vectors.
        fixed_value -> The value of the parameter being varied in profile likelihood.
        param -> The parameter being varied in the profile likelihood.
        dataset -> The dataset combination used for the fitting.
        variation -> What cosmological model to fit for (i.e. LCDM, w0waCDM).
        FS_BAO_data_vecs -> The dictionary of FS (or FS/BAO joint if desired) data vectors across the various tracers.
        CMB_data -> The data vector for the CMB.
        FS_BAO_cov_mats -> The covariance matrices (across the different tracers) associated with the FS (or FS+BAO) vectors.
        inv_CMB_cov_sqrt -> The (reparameterized) inverse covariance matrix associated with the CMB vector.
        emus_FS -> The dictionary (across the different tracers) with each element being a list of the emulators needed for different components (monopole/quadrupole/hexadecapole).
        emu_BAO -> The BAO emulator.
        emu_CMB -> A list of the components (TT/TE/EE) for the CMB emulator.
        tracers_used -> List of all the tracers used in the joint fit.
        kin_used -> A dictionary (across the different tracers) of the input k bins.
        zeff_used -> The dictionary (across the different tracers) of the effective redshifts.
        wmat_used -> The dictionary (across the different tracers) with the relevant window matrices.
        nd_used ->
        fsat_used ->
        sigv_used ->
    """
    # Ranges for each of the cosmological parameters (narrow from the emulator itself to speed up computation)
    ln10As_range = [2.7, 3.3]; ns_range = [0.8, 1.1]; H0_range = [67, 74]; ωb_range = [0.02, 0.025]; ωc_range = [0.09, 0.15]; w0_range = [-1.6, 0.4]; wa_range = [-3, 1.64]
    # emulator ranges are (2, 3.5), (0.8, 1.1), (50, 80), (0.02, 0.025), (0.09, 0.25), (-2, 0.5), (-3, 1.64)

    # Priors to apply (sometimes): ns10 and BBN
    ns_prior = [0.9649, 0.042]
    ωb_prior = [0.02218, 0.00055]
    
    # Priors to use for CMB only parameters
    mν_fixed = 0.06
    τ_prior = [0.0506, 0.0086]
    yₚ_prior = [1.0, 0.0025]
    if dataset in ["FS+CMB", "FS+BAO+CMB", "FS+BAO+CMB+SN"]
        τ ~ Normal(τ_prior[1], τ_prior[2])
        yₚ ~ Normal(yₚ_prior[1], yₚ_prior[2])
    end

    # Reduce the parameter exploration space for ns and omega_b (only relevant for FS or FS+BAO, when ns10 and BBN priors are applied)
    if dataset in ["FS", "FS+BAO"]
        ns ~ Normal(ns_prior[1], ns_prior[2])
        ωb ~ Normal(ωb_prior[1], ωb_prior[2])
    else
        ns ~ Uniform(ns_range[1], ns_range[2])
        ωb ~ Uniform(ωb_range[1], ωb_range[2])
    end

    # Sets the exploration ranges (uniform for the other cosmological parameters)
    if param == "ln10As"
        ln10As = fixed_value; H0 ~ Uniform(H0_range[1], H0_range[2]); ωc ~ Uniform(ωc_range[1], ωc_range[2])
        if variation == "LCDM"
            w0 = -1; wa = 0
        elseif variation == "w0waCDM"
            w0 ~ Uniform(w0_range[1], w0_range[2]); wa ~ Uniform(wa_range[1], wa_range[2])
        end
    elseif param == "H0"
        ln10As ~ Uniform(ln10As_range[1], ln10As_range[2]); H0 = fixed_value; ωc ~ Uniform(ωc_range[1], ωc_range[2])
        if variation == "LCDM"
            w0 = -1; wa = 0
        elseif variation == "w0waCDM"
            w0 ~ Uniform(w0_range[1], w0_range[2]); wa ~ Uniform(wa_range[1], wa_range[2])
        end
    elseif param == "Om"
        ln10As ~ Uniform(ln10As_range[1], ln10As_range[2]); H0 ~ Uniform(H0_range[1], H0_range[2]); Om = fixed_value; ωc = Om*(H0/100)^2-ωb-0.0006441915
        if variation == "LCDM"
            w0 = -1; wa = 0
        elseif variation == "w0waCDM"
            w0 ~ Uniform(w0_range[1], w0_range[2]); wa ~ Uniform(wa_range[1], wa_range[2])
        end
    elseif param == "w0"
        ln10As ~ Uniform(ln10As_range[1], ln10As_range[2]); H0 ~ Uniform(H0_range[1], H0_range[2]); ωc ~ Uniform(ωc_range[1], ωc_range[2]); w0 = fixed_value; wa ~ Uniform(wa_range[1], wa_range[2]) 
    elseif param == "wa"
        ln10As ~ Uniform(ln10As_range[1], ln10As_range[2]); H0 ~ Uniform(H0_range[1], H0_range[2]); ωc ~ Uniform(ωc_range[1], ωc_range[2]); w0 ~ Uniform(w0_range[1], w0_range[2]); wa = fixed_value
    end
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa] # makes a vector of the cosmological parameters
    #println("\n", cosmo_params, "\n")
    # Retrieves the other EFT (nuisance) parameters
    b1_range = [0, 6]; b2_range = [-15, 5]; bs_range = [-10, 15]; alpha0_range = [-100, 400]; alpha2_range = [-800, 200]; st0_range = [-80, 80]; st2_range = [-200, 200] # sampling ranges for EFT parameters
    BAO_emulator_details = Effort.get_BAO(cosmo_params, emu_BAO)
    f_values, sigma8_values = BAO_emulator_details[2:8], BAO_emulator_details[9:15] # extracts f and sigma8 values from the BAO emulator ahead of time (to avoid re-calling for each tracer)
    for tracer in tracers_used
        if tracer == "BGS"
            b1p_BGS ~ Uniform(b1_range[1], b1_range[2])
            b2p_BGS ~ Uniform(b2_range[1], b2_range[2])
            b3p_BGS = 0 
            bsp_BGS ~ Uniform(bs_range[1], bs_range[2])
            alpha0p_BGS ~ Uniform(alpha0_range[1], alpha0_range[2])
            alpha2p_BGS ~ Uniform(alpha2_range[1], alpha2_range[2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(st0_range[1], st0_range[2])
            st2p_BGS ~ Uniform(st2_range[1], st2_range[2])
            st4p_BGS = 0
            eft_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
            f, sigma8 = f_values[1], sigma8_values[1]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(b1_range[1], b1_range[2])
            b2p_LRG1 ~ Uniform(b2_range[1], b2_range[2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(bs_range[1], bs_range[2]) 
            alpha0p_LRG1 ~ Uniform(alpha0_range[1], alpha0_range[2])
            alpha2p_LRG1 ~ Uniform(alpha2_range[1], alpha2_range[2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(st0_range[1], st0_range[2])
            st2p_LRG1 ~ Uniform(st2_range[1], st2_range[2])
            st4p_LRG1 = 0
            eft_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
            f, sigma8 = f_values[2], sigma8_values[2]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(b1_range[1], b1_range[2]) 
            b2p_LRG2 ~ Uniform(b2_range[1], b2_range[2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(bs_range[1], bs_range[2]) 
            alpha0p_LRG2 ~ Uniform(alpha0_range[1], alpha0_range[2])
            alpha2p_LRG2 ~ Uniform(alpha2_range[1], alpha2_range[2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(st0_range[1], st0_range[2])
            st2p_LRG2 ~ Uniform(st2_range[1], st2_range[2])
            st4p_LRG2 = 0
            eft_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
            f, sigma8 = f_values[3], sigma8_values[3]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(b1_range[1], b1_range[2]) 
            b2p_LRG3 ~ Uniform(b2_range[1], b2_range[2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(bs_range[1], bs_range[2]) 
            alpha0p_LRG3 ~ Uniform(alpha0_range[1], alpha0_range[2])
            alpha2p_LRG3 ~ Uniform(alpha2_range[1], alpha2_range[2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(st0_range[1], st0_range[2])
            st2p_LRG3 ~ Uniform(st2_range[1], st2_range[2])
            st4p_LRG3 = 0
            eft_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
            f, sigma8 = f_values[4], sigma8_values[4]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(b1_range[1], b1_range[2])
            b2p_ELG2 ~ Uniform(b2_range[1], b2_range[2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(bs_range[1], bs_range[2]) 
            alpha0p_ELG2 ~ Uniform(alpha0_range[1], alpha0_range[2])
            alpha2p_ELG2 ~ Uniform(alpha2_range[1], alpha2_range[2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(st0_range[1], st0_range[2])
            st2p_ELG2 ~ Uniform(st2_range[1], st2_range[2])
            st4p_ELG2 = 0
            eft_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
            f, sigma8 = f_values[6], sigma8_values[6]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(b1_range[1], b1_range[2])
            b2p_QSO ~ Uniform(b2_range[1], b2_range[2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(bs_range[1], bs_range[2]) 
            alpha0p_QSO ~ Uniform(alpha0_range[1], alpha0_range[2])
            alpha2p_QSO ~ Uniform(alpha2_range[1], alpha2_range[2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(st0_range[1], st0_range[2])
            st2p_QSO ~ Uniform(st2_range[1], st2_range[2])
            st4p_QSO = 0
            eft_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]
            f, sigma8 = f_values[7], sigma8_values[7]
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_physical
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl; b3e = b3l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/nd_used[tracer]; st2e = st2p/nd_used[tracer]*fsat_used[tracer]*sigv_used[tracer]^2; st4e = st4p/nd_used[tracer]*fsat_used[tracer]*sigv_used[tracer]^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e] # collects together final EFT parameter vector
        theta_FS = vcat(cosmo_params, eft_params)
        FS_BAO_prediction = wmat_used[tracer] * theory_FS(theta_FS, emus_FS[tracer], kin_used[tracer])
        println("\n", cosmo_params, "\n")#######################################################################################################################
        if dataset in ["FS", "FS+CMB"] # does not add BAO vector unless BAO is being fit too
            nothing
        elseif dataset in ["FS+BAO", "FS+BAO+CMB", "FS+BAO+CMB+SN"]
            FS_BAO_prediction = vcat(FS_BAO_prediction, theory_BAO(cosmo_params, emu_BAO, zeff_used[tracer], tracer))
        end
        FS_BAO_data_vecs[tracer] ~ MvNormal(FS_BAO_prediction, (FS_BAO_cov_mats[tracer]+FS_BAO_cov_mats[tracer]')/2)
    end

    # Adds in the CMB fit case if needed (skips otherwise)
    if dataset in ["FS+CMB", "FS+BAO+CMB", "FS+BAO+CMB+SN"] # skips this step if CMB not being fit too
        theta_CMB = [ln10As, ns, H0, ωb, ωc, τ, mν_fixed, w0, wa]
        CMB_prediction = inv_CMB_cov_sqrt * theory_CMB(theta_CMB, emu_CMB) ./(yₚ^2)
        CMB_data ~ MvNormal(CMB_prediction, I)
    end

    # Adds in the ns10 and BBN priors if the data calls for it (only for FS and FS+BAO)
    if dataset in ["FS", "FS+BAO"]
        sigma_ωb = ωb_prior[2]
        dωb = ωb - ωb_prior[1]
        sigma_ns = ns_prior[2]
        dns = ns - ns_prior[1]
        Turing.@addlogprob! - 0.5 * dωb^2/sigma_ωb^2
        Turing.@addlogprob! - 0.5 * dns^2/sigma_ns^2
        return nothing
    else
        return nothing
    end
end


@everywhere function run_worker(fixed_value, param, dataset, variation, FS_BAO_data_vecs, CMB_data, FS_BAO_cov_mats, inv_CMB_cov_sqrt, emus_FS, emu_BAO, emu_CMB, tracers_used, kin_used, zeff_used, wmat_used, nd_used, fsat_used, sigv_used)
    """
    Runs the worker for a given parameter value in the profile likelihood.
    (Performs LBFGS minimization of chi-squared).
    """
    fit_model = model(fixed_value, param, dataset, variation, FS_BAO_data_vecs, CMB_data, FS_BAO_cov_mats, inv_CMB_cov_sqrt, emus_FS, emu_BAO, emu_CMB, tracers_used, kin_used, zeff_used, wmat_used, nd_used, fsat_used, sigv_used);
    profile_values_array = SharedArray{Float64}(n_runs)
    if dataset in ["FS", "FS+BAO"]
        if variation == "LCDM"
            ncosmo = 5
        elseif variation == "w0waCDM"
            ncosmo = 7
        end
    elseif dataset in ["FS+CMB", "FS+BAO+CMB", "FS+BAO+CMB+SN"] # extra two parameters for CMB beyond FS and BAO
        if variation == "LCDM"
            ncosmo = 7
        elseif variation == "w0waCDM"
            ncosmo = 9
        end
    end
    bestfit_values_array = SharedArray{Float64}((ncosmo-1)+7*size(tracer_vector)[1], n_runs)
    # Performs many independent runs in order to reach global minima
    for i in 1:n_runs
        fit_result = maximum_likelihood(fit_model, LBFGS(m=100)) # add init_values and preconditioning if needed!
        profile_values_array[i] = fit_result.lp
        bestfit_values_array[:, i] = fit_result.values.array
    end
    # Returns array of all best fits for each of the different runs
    return [profile_values_array, bestfit_values_array]
end

main()