
using Pkg
Pkg.activate(".")
using ArgParse
using Distributed

# The number of parallel processes to run for each profile likelihood
n_bins = 16
addprocs(n_bins)
# The number of independent minimization runs to perform to reach global minimum
n_runs = 50

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
    using SNIaLikelihoods
    using PlanckLite
    using SharedArrays
    using OptimizationOptimJL: ParticleSwarm
    using DataInterpolations
end

function main()

    # Reads in the necessary arguments to run script 
    config = ArgParseSettings()
    @add_arg_table config begin
        "--param" 
        help = "Specify the parameter" # i.e. ln10As, H0, w0, wa, etc.
        arg_type = String
        required = true
        "--dataset" 
        help = "Specify the desired dataset" # i.e. FS, FS+BAO, FS+BAO+CMB, FS+BAO+CMB+SN, etc.
        arg_type = String
        required = true
        "--variation"
        help = "Specify the desired variation" # i.e. LCDM, w0waCDM
        arg_type = String
        required = true
        "--tracer_list" 
        help = "Specify the tracer(s)" # i.e. BGS, LRG1, LRG2, LRG3, ELG2, QSO, BGS,LRG1,LRG2,LRG3,ELG2,QSO, etc.
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
        "--SN_type"
        help = "Specify SN type (if applicable)."
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
    save_path = "/global/homes/j/jgmorawe/FrequentistExample1/profile_likelihood_results/$(param)_$(dataset)_$(variation)_$(tracer_list)_higher_resolution_testing"############################################
    lower = parsed_args["lower"]
    upper = parsed_args["upper"]
    desi_data_dir = parsed_args["desi_data_dir"]
    FS_emu_dir = parsed_args["FS_emu_dir"]
    BAO_emu_dir = parsed_args["BAO_emu_dir"]
    CMB_emu_dir = parsed_args["CMB_emu_dir"]
    SN_type = parsed_args["SN_type"]
    
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
    cov_size = Dict(tracer => size(cov_pk_baopost_all[tracer])[1] for tracer in tracers) # computes dimension of covariance matrix so knows how to slice to isolate BAO component
    cov_baopost_all = Dict(tracer => (cov_pk_baopost_all[tracer])[cov_size[tracer]-1:cov_size[tracer], cov_size[tracer]-1:cov_size[tracer]] for tracer in tracers); 
    cov_baopost_all["BGS"] = cov_pk_baopost_all["BGS"][cov_size["BGS"]:cov_size["BGS"], cov_size["BGS"]:cov_size["BGS"]]
    cov_baopost_all["QSO"] = cov_pk_baopost_all["QSO"][cov_size["QSO"]:cov_size["QSO"], cov_size["QSO"]:cov_size["QSO"]] # isolates the BAO only covariance (adjusts since different for BGS and QSO)
    # Actual data vectors (reparameterized for efficiency)
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
    Lya_cov = [3.294630008635918330e+03 1.295814779172360204e+03; 1.295814779172360204e+03 2.235282696116466013e+03]
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


    
    #if dataset == "FS"
    #    Γ_LSS_all = Dict(tracer => sqrt(cov_pk_all[tracer]) for tracer in tracers) # using reparameterization to make more efficient
    #    iΓ_LSS_all = Dict(tracer => inv(Γ_LSS_all[tracer]) for tracer in tracers)
    #    D_LSS_all = Dict(tracer => iΓ_LSS_all[tracer]*pk_all[tracer] for tracer in tracers)
    #elseif dataset == "BAO"
    #    Γ_LSS_all = Dict(tracer => sqrt(cov_baopost_all[tracer]) for tracer in tracers)
    #    iΓ_LSS_all = Dict(tracer => inv(Γ_LSS_all[tracer]) for tracer in tracers)
    #    D_LSS_all = Dict(tracer => iΓ_LSS_all[tracer]*baopost_all[tracer] for tracer in tracers)
    #elseif dataset in ["FS+BAO", "FS+BAO+CMB", "FS+BAO+CMB+SN"]
    #    Γ_LSS_all = Dict(tracer => sqrt(cov_pk_baopost_all[tracer]) for tracer in tracers)
    #    iΓ_LSS_all = Dict(tracer => inv(Γ_LSS_all[tracer]) for tracer in tracers)
    #    D_LSS_all = Dict(tracer => iΓ_LSS_all[tracer]*pk_baopost_all[tracer] for tracer in tracers)
    #end

    # Reads in the Planck CMB data (also using reparameterization to make more efficient)
    #Γ_CMB = sqrt(PlanckLite.cov)
    #iΓ_CMB = inv(Γ_CMB)
    #D_CMB = iΓ_CMB * PlanckLite.data

    # Reads in the supernovae data (also using reparameterization to make more efficient)
    #DESY5SN = DESY5SN_info()
    #z_DESY5SN = DESY5SN.data.zHD
    #cov_DESY5SN = DESY5SN.covariance
    #data_DESY5SN = DESY5SN.obs_flatdata
    #Γ_DESY5SN = sqrt(cov_DESY5SN)
    #iΓ_DESY5SN = inv(Γ_DESY5SN)
    #D_DESY5SN = iΓ_DESY5SN * data_DESY5SN

    #PantheonPlusSN = PantheonPlusSN_info()
    #z_PantheonPlusSN = PantheonPlusSN.data.zHD
    #cov_PantheonPlusSN = PantheonPlusSN.covariance
    #data_PantheonPlusSN = PantheonPlusSN.obs_flatdata
    #Γ_PantheonPlusSN = sqrt(cov_PantheonPlusSN)
    #iΓ_PantheonPlusSN = inv(Γ_PantheonPlusSN)
    #D_PantheonPlusSN = iΓ_PantheonPlusSN * data_PantheonPlusSN

    #Union3SN = Union3SN_info()
    #z_Union3SN = Union3SN.data.zHD
    #cov_Union3SN = Union3SN.covariance
    #data_Union3SN = Union3SN.obs_flatdata
    #Γ_Union3SN = sqrt(cov_Union3SN)
    #iΓ_Union3SN = inv(Γ_Union3SN)
    #D_Union3SN = iΓ_Union3SN * data_Union3SN

    # Reads in the emulators associated with the DESI full-shape/BAO data
    mono_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/0/" for tracer in tracers)
    quad_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/2/" for tracer in tracers)
    hexa_paths = Dict(tracer => FS_emu_dir * string(zindex_all[tracer]) * "/4/" for tracer in tracers)
    FS_emus = Dict(tracer => [Effort.load_multipole_noise_emulator(mono_paths[tracer]),
                              Effort.load_multipole_noise_emulator(quad_paths[tracer]),
                              Effort.load_multipole_noise_emulator(hexa_paths[tracer])] for tracer in tracers)
    BAO_emu = Effort.load_BAO_emulator(BAO_emu_dir)

    # Reads in the emulators associated with the Plancklite CMB data 
    TT_emu = Capse.load_emulator(CMB_emu_dir * "/TT/")
    TE_emu = Capse.load_emulator(CMB_emu_dir * "/TE/")
    EE_emu = Capse.load_emulator(CMB_emu_dir * "/EE/")
    CMB_emus = [TT_emu, TE_emu, EE_emu]
    
    # Additional parameters needed for EFT basis change
    nd_all = Dict("BGS" => 5e-4, "LRG1" => 5e-4, "LRG2" => 5e-4, "LRG3" => 3e-4, "ELG2" => 5e-4, "QSO" => 3e-5)
    fsat_all = Dict("BGS" => 0.15, "LRG1" => 0.15, "LRG2" => 0.15, "LRG3" => 0.15, "ELG2" => 0.10, "QSO" => 0.03)
    sigv_all = Dict("BGS" => 150*(10)^(1/3)*(1+0.2)^(1/2)/70, 
                    "LRG1" => 150*(10)^(1/3)*(1+0.8)^(1/2)/70, 
                    "LRG2" => 150*(10)^(1/3)*(1+0.8)^(1/2)/70, 
                    "LRG3" => 150*(10)^(1/3)*(1+0.8)^(1/2)/70, 
                    "ELG2" => 150*2.1^(1/2)/70, 
                    "QSO" => 150*(10)^(0.7/3)*(2.4)^(1/2)/70)

    # Distributes the variables among the processes 
    @everywhere n_runs = $n_runs
    @everywhere param = $param
    @everywhere dataset = $dataset
    @everywhere variation = $variation
    @everywhere tracer_vector = $tracer_vector  
    @everywhere D_FS_all = $D_FS_all
    @everywhere iΓ_FS_all = $iΓ_FS_all
    @everywhere D_BAO_all = $D_BAO_all
    @everywhere iΓ_BAO_all = $iΓ_BAO_all
    @everywhere D_FS_BAO_all = $D_FS_BAO_all
    @everywhere iΓ_FS_BAO_all = $iΓ_FS_BAO_all
    @everywhere D_Lya = $D_Lya
    @everywhere iΓ_Lya = $iΓ_Lya
    @everywhere kin_all = $kin_all
    @everywhere zeff_all = $zeff_all
    @everywhere wmat_all = $wmat_all
    @everywhere nd_all = $nd_all
    @everywhere fsat_all = $fsat_all
    @everywhere sigv_all = $sigv_all
    @everywhere iΓ_CMB = $iΓ_CMB
    @everywhere D_CMB = $D_CMB
    @everywhere iΓ_DESY5SN = $iΓ_DESY5SN
    @everywhere D_DESY5SN = $D_DESY5SN
    #@everywhere iΓ_PantheonPlusSN = $iΓ_PantheonPlusSN
    #@everywhere D_PantheonPlusSN = $D_PantheonPlusSN
    #@everywhere iΓ_Union3SN = $iΓ_Union3SN
    #@everywhere D_Union3SN = $D_Union3SN
    @everywhere FS_emus = $FS_emus
    @everywhere BAO_emu = $BAO_emu
    #@everywhere CMB_emus = $CMB_emus
    @everywhere SN_type = $SN_type

    # Specifies the range of parameter values and initiates arrays to store
    param_values = range(lower, stop=upper, length=n_bins)
    profile_values = SharedArray{Float64}(n_bins, n_runs)
    # Specifies the number of bestfit parameters
    if dataset in ["FS", "BAO", "FS+BAO"]
        if variation == "LCDM"
            ncosmo_tot = 5
        elseif variation == "w0waCDM"
            ncosmo_tot = 7
        end
    elseif dataset == "FS+BAO+CMB"
        if variation == "LCDM"
            ncosmo_tot = 8
        elseif variation == "w0waCDM"
            ncosmo_tot = 10
        end
    elseif dataset == "FS+BAO+CMB+SN"
        if variation == "LCDM"
            ncosmo_tot = 9
        elseif variation == "w0waCDM"
            ncosmo_tot = 11
        end
    end
    if dataset == "BAO"
        bestfit_values = SharedArray{Float64}(n_bins, (ncosmo_tot-1), n_runs)
    else
        bestfit_values = SharedArray{Float64}(n_bins, (ncosmo_tot-1)+7*size(tracer_vector)[1], n_runs)
    end
    @everywhere ncosmo_tot = $ncosmo_tot

    # Runs the parallel processes and saves results to file
    @sync @distributed for index in 1:length(param_values)
        @time (profile_values[index, :], bestfit_values[index, :, :]) = run_worker(param_values[index])#param_values[index], param, variation, tracer_vector, SN_type)
    end  
    npzwrite(string(save_path, "_param_values.npy"), param_values)
    npzwrite(string(save_path, "_profile_values.npy"), profile_values)
    npzwrite(string(save_path, "_bestfit_values.npy"), bestfit_values)                                                                          
end

@everywhere function theory_FS(theta_FS, emu_FS_components, kin)
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

@everywhere function theory_BAO(theta_BAO, emu_BAO, zeff, tracer)
    # theta_BAO: [ln10As, ns, H0, ωb, ωc, w0, wa]
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

@everywhere function theory_CMB(theta_CMB, emu_CMB_components)
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

@everywhere function theory_SN(theta_SN, Mb, z_SN)
    # theta_SN: [ln10As, ns, H0, ωb, ωc, w0, wa]
    h = theta_SN[3]/100; Ωcb = (theta_SN[4]+theta_SN[5])/h^2; w0 = theta_SN[6]; wa = theta_SN[7]
    mν_fixed = 0.06 # fixes neutrino mass
    z_interp = Array(LinRange(0, 2, 50)) # uses interpolation to not have to calculate for all supernovae redshifts
    DL_interp = Effort._r_z.(z_interp, Ωcb, h; mν=mν_fixed, w0=w0, wa=wa) .* (1 .+ z_interp)
    DL_SN = DataInterpolations.QuadraticSpline(DL_interp, z_interp).(z_SN)
    return 5 .* log10.(DL_SN) .+ 25 .+ Mb
end


@everywhere @model function model_FS(D_FS_all)
    # Draws cosmological parameters
    cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.9649, 0.042], "H0" => [50, 80], "ωb" => [0.02218, 0.00055], 
                        "ωc" => [0.09, 0.25], "w0" => [-2, 0.5], "wa" => [-3, 1.64])
    ln10As ~ Uniform(cosmo_ranges["ln10As"][1], cosmo_ranges["ln10As"][2])
    ns ~ Normal(cosmo_ranges["ns"][1], cosmo_ranges["ns"][2])               # might need to adjust if using MAP
    H0 ~ Uniform(cosmo_ranges["H0"][1], cosmo_ranges["H0"][2])
    ωb ~ Normal(cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])                # might need to adjust if using MAP
    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
    if param == "Ωm" # adjusts for the case where omega matter is held fixed (and is converted to omega CDM)
        ωc = ωc * (H0/100)^2 - ωb - 0.00064419153
    end
    # Constructs parameter vector given samples drawn
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    # Extracts f and sigma8 values for each tracer using BAO emulator
    fsigma8_info = Effort.get_BAO(cosmo_params, BAO_emu)
    f_all = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                 "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_all = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                      "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Draws EFT nuisance parameters
    eft_ranges = Dict("b1p" => [0, 6], "b2p" =>  [-15, 5], "bsp" => [-10, 15], "alpha0p" => [-100, 400], "alpha2p" => [-800, 200],
                      "st0p" => [-80, 80], "st2p" => [-200, 200])
    #b1p_vector = Vector{Any}(undef, length(tracer_vector))
    #b2p_vector = Vector{Any}(undef, length(tracer_vector))
    #b3p_vector = Vector{Any}(undef, length(tracer_vector))
    #bsp_vector = Vector{Any}(undef, length(tracer_vector))
    #alpha0p_vector = Vector{Any}(undef, length(tracer_vector))
    #alpha2p_vector = Vector{Any}(undef, length(tracer_vector))
    #alpha4p_vector = Vector{Any}(undef, length(tracer_vector))
    #st0p_vector = Vector{Any}(undef, length(tracer_vector))
    #st2p_vector = Vector{Any}(undef, length(tracer_vector))
    #st4p_vector = Vector{Any}(undef, length(tracer_vector))
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
            b2p_BGS ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
            b3p_BGS = 0
            bsp_BGS ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
            alpha0p_BGS ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
            alpha2p_BGS ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
            st2p_BGS ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
            b2p_LRG1 ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
            alpha0p_LRG1 ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
            alpha2p_LRG1 ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
            st2p_LRG1 ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
            b2p_LRG2 ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
            alpha0p_LRG2 ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
            alpha2p_LRG2 ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
            st2p_LRG2 ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
            b2p_LRG3 ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
            alpha0p_LRG3 ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
            alpha2p_LRG3 ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
            st2p_LRG3 ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
            b2p_ELG2 ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
            alpha0p_ELG2 ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
            alpha2p_ELG2 ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
            st2p_ELG2 ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
            b2p_QSO ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
            b3p_QSO = 0
            bsp_QSO ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
            alpha0p_QSO ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
            alpha2p_QSO ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
            st2p_QSO ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
    #for i in 1:length(tracer_vector)
    #    tracer = tracer_vector[i]
    #    b1p_vector[i] ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
    #    b2p_vector[i] ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
    #    b3p_vector[i] = 0 
    #    bsp_vector[i] ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
    #    alpha0p_vector[i] ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
    ##    alpha2p_vector[i] ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
    #    alpha4p_vector[i] = 0
    #    st0p_vector[i] ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
    #    st2p_vector[i] ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
    #    st4p_vector[i] = 0
        #b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = [b1p_vector[i], b2p_vector[i], b3p_vector[i], bsp_vector[i],
                                                                           #alpha0p_vector[i], alpha2p_vector[i], alpha4p_vector[i],
                                                                           #st0p_vector[i], st2p_vector[i], st4p_vector[i]]
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


#@everywhere @model function model_FS(D_FS_all, iΓ_FS_all, fixed_value, param, variation, emus_FS, emu_BAO, tracer_vector, nd_used, fsat_used, sigv_used, kin_used, wmat_used)
#    # Draws cosmological parameters
#    cosmo_params = Vector{Any}(undef, 7) # 7 cosmological parameters: ln10As, ns, H0, ωb, ωc, w0, wa
#    if variation == "LCDM"
#        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc"]
#        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.9649, 0.042], "H0" => [50, 80], "ωb" => [0.02218, 0.00055], "ωc" => [0.09, 0.25])
#        cosmo_params[6], cosmo_params[7] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
#    elseif variation == "w0waCDM"
#        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
#        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.9649, 0.042], "H0" => [50, 80], "ωb" => [0.02218, 0.00055], "ωc" => [0.09, 0.25],
#                            "w0" => [-2, 0.5], "wa" => [-3, 1.64])
#    end
#    for i in 1:length(cosmo_params_list)
#        if cosmo_params_list[i] == param
#            cosmo_params[i] = fixed_value
#        elseif cosmo_params_list[i] in ["ns", "ωb"]
#            cosmo_params[i] ~ Normal(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])#################################### need to change for ALL functions!!!!!!!!!!##########################################
#        else
#            cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
#        end
#    end
#    # Constructs the parameter vector given the samples drawn above
#    cosmo_params = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], 
#                    cosmo_params[5], cosmo_params[6], cosmo_params[7]]
#    # Extracts f and sigma8 values for each of the tracers using BAO emulator
#    f_sigma8 = Effort.get_BAO(cosmo_params, emu_BAO)
#    f_all = Dict("BGS" => f_sigma8[2], "LRG1" => f_sigma8[3], "LRG2" => f_sigma8[4], "LRG3" => f_sigma8[5], "ELG2" => f_sigma8[7], "QSO" => f_sigma8[8])
#    sigma8_all = Dict("BGS" => f_sigma8[9], "LRG1" => f_sigma8[10], "LRG2" => f_sigma8[11], "LRG3" => f_sigma8[12], "ELG2" => f_sigma8[14], "QSO" => f_sigma8[15])
#    # Draws EFT nuisance parameters
#    eft_ranges = Dict("b1p" => [0, 6], "b2p" =>  [-15, 5], "bsp" => [-10, 15], "alpha0p" => [-100, 400], "alpha2p" => [-800, 200], "st0p" => [-80, 80], "st2p" => [-200, 200])
#    b1p_vector = Vector{Any}(undef, length(tracer_vector))
#    b2p_vector = Vector{Any}(undef, length(tracer_vector))
#    b3p_vector = Vector{Any}(undef, length(tracer_vector))
#    bsp_vector = Vector{Any}(undef, length(tracer_vector))
#    alpha0p_vector = Vector{Any}(undef, length(tracer_vector))
#    alpha2p_vector = Vector{Any}(undef, length(tracer_vector))
#    alpha4p_vector = Vector{Any}(undef, length(tracer_vector))
#    st0p_vector = Vector{Any}(undef, length(tracer_vector))
#    st2p_vector = Vector{Any}(undef, length(tracer_vector))
#    st4p_vector = Vector{Any}(undef, length(tracer_vector))
#    # Iterates through each of the tracers
#    for i in 1:length(tracer_vector)
#        tracer = tracer_vector[i]
#        b1p_vector[i] ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
#        b2p_vector[i] ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
#        b3p_vector[i] = 0 
#        bsp_vector[i] ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
#        alpha0p_vector[i] ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
#        alpha2p_vector[i] ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
#        alpha4p_vector[i] = 0
#        st0p_vector[i] ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
#        st2p_vector[i] ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
#        st4p_vector[i] = 0
#        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = [b1p_vector[i], b2p_vector[i], b3p_vector[i], bsp_vector[i],
#                                                                            alpha0p_vector[i], alpha2p_vector[i], alpha4p_vector[i], st0p_vector[i], st2p_vector[i], st4p_vector[i]]
#        # Converts physical to Eulerian basis
#        f, sigma8 = f_all[tracer], sigma8_all[tracer]
#        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
#        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
#        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
#        st0e = st0p/(nd_used[tracer]); st2e = st2p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^2; st4e = st4p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^4
#        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
#        # Combines cosmological and EFT parameters into one vector for the FS theory function
#        cosmo_eft_params = vcat(cosmo_params, eft_params)
#        # Calculates FS theory vector given the parameters
#        prediction = iΓ_FS_all[tracer] * (wmat_used[tracer] * theory_FS(cosmo_eft_params, emus_FS[tracer], kin_used[tracer]))
#        D_FS_all[tracer] ~ MvNormal(prediction, I)
#    end
#    # Adds in the BBN/ns10 priors (since not including CMB)
#    ns_prior = [0.9649, 0.042]
#    ωb_prior = [0.02218, 0.00055]
#    sigma_ωb = ωb_prior[2]
#    dωb = cosmo_params[4] - ωb_prior[1]
#    sigma_ns = ns_prior[2]
#    dns = cosmo_params[2] - ns_prior[1]
#    Turing.@addlogprob! - 0.5 * dωb^2/sigma_ωb^2
#    Turing.@addlogprob! - 0.5 * dns^2/sigma_ns^2
#    return nothing
#end

@everywhere @model function model_BAO(D_BAO_all, iΓ_BAO_all, D_Lya, iΓ_Lya, fixed_value, param, variation, emu_BAO, tracer_vector, zeff_used)
    # Draws cosmological parameters
    cosmo_params = Vector{Any}(undef, 7) # 7 cosmological parameters: ln10As, ns, H0, ωb, ωc, w0, wa
    if variation == "LCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc"]
        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25])
        cosmo_params[6], cosmo_params[7] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
    elseif variation == "w0waCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25],
                            "w0" => [-2, 0.5], "wa" => [-3, 1.64])
    end
    for i in 1:length(cosmo_params_list)
        if cosmo_params_list[i] == param
            cosmo_params[i] = fixed_value
        else
            cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
        end
    end
    # Constructs the parameter vector given the samples drawn above
    cosmo_params = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], 
                    cosmo_params[5], cosmo_params[6], cosmo_params[7]]
    # Constructs theory vector and associates data vector with the model
    for tracer in tracer_vector
        prediction = iΓ_BAO_all[tracer] * theory_BAO(cosmo_params, emu_BAO, zeff_used[tracer], tracer)
        D_BAO_all[tracer] ~ MvNormal(prediction, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction = iΓ_Lya * theory_BAO(cosmo_params, emu_BAO, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction, I)
    # Adds in the BBN/ns10 priors (since not including CMB)
    ns_prior = [0.9649, 0.042]
    ωb_prior = [0.02218, 0.00055]
    sigma_ωb = ωb_prior[2]
    dωb = cosmo_params[4] - ωb_prior[1]
    sigma_ns = ns_prior[2]
    dns = cosmo_params[2] - ns_prior[1]
    Turing.@addlogprob! - 0.5 * dωb^2/sigma_ωb^2
    Turing.@addlogprob! - 0.5 * dns^2/sigma_ns^2
    return nothing
end

@everywhere @model function model_FS_BAO(D_FS_BAO_all, iΓ_FS_BAO_all, D_Lya, iΓ_Lya, fixed_value, param, variation, emus_FS, emu_BAO, tracer_vector, zeff_used, nd_used, fsat_used, sigv_used, kin_used, wmat_used)
    # Draws cosmological parameters
    cosmo_params = Vector{Any}(undef, 7) # 7 cosmological parameters: ln10As, ns, H0, ωb, ωc, w0, wa
    if variation == "LCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc"]
        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25])
        cosmo_params[6], cosmo_params[7] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
    elseif variation == "w0waCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25],
                            "w0" => [-2, 0.5], "wa" => [-3, 1.64])
    end
    for i in 1:length(cosmo_params_list)
        if cosmo_params_list[i] == param
            cosmo_params[i] = fixed_value
        else
            cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
        end
    end
    # Constructs the parameter vector given the samples drawn above
    cosmo_params = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], 
                    cosmo_params[5], cosmo_params[6], cosmo_params[7]]
    # Extracts f and sigma8 values for each of the tracers using BAO emulator
    f_sigma8 = Effort.get_BAO(cosmo_params, emu_BAO)
    f_all = Dict("BGS" => f_sigma8[2], "LRG1" => f_sigma8[3], "LRG2" => f_sigma8[4], "LRG3" => f_sigma8[5], "ELG2" => f_sigma8[7], "QSO" => f_sigma8[8])
    sigma8_all = Dict("BGS" => f_sigma8[9], "LRG1" => f_sigma8[10], "LRG2" => f_sigma8[11], "LRG3" => f_sigma8[12], "ELG2" => f_sigma8[14], "QSO" => f_sigma8[15])
    # Draws EFT nuisance parameters
    eft_ranges = Dict("b1p" => [0, 6], "b2p" =>  [-15, 5], "bsp" => [-10, 15], "alpha0p" => [-100, 400], "alpha2p" => [-800, 200], "st0p" => [-80, 80], "st2p" => [-200, 200])
    b1p_vector = Vector{Any}(undef, length(tracer_vector))
    b2p_vector = Vector{Any}(undef, length(tracer_vector))
    b3p_vector = Vector{Any}(undef, length(tracer_vector))
    bsp_vector = Vector{Any}(undef, length(tracer_vector))
    alpha0p_vector = Vector{Any}(undef, length(tracer_vector))
    alpha2p_vector = Vector{Any}(undef, length(tracer_vector))
    alpha4p_vector = Vector{Any}(undef, length(tracer_vector))
    st0p_vector = Vector{Any}(undef, length(tracer_vector))
    st2p_vector = Vector{Any}(undef, length(tracer_vector))
    st4p_vector = Vector{Any}(undef, length(tracer_vector))
    # Iterates through each of the tracers
    for i in 1:length(tracer_vector)
        tracer = tracer_vector[i]
        b1p_vector[i] ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
        b2p_vector[i] ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
        b3p_vector[i] = 0 
        bsp_vector[i] ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
        alpha0p_vector[i] ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
        alpha2p_vector[i] ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
        alpha4p_vector[i] = 0
        st0p_vector[i] ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
        st2p_vector[i] ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
        st4p_vector[i] = 0
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = [b1p_vector[i], b2p_vector[i], b3p_vector[i], bsp_vector[i],
                                                                            alpha0p_vector[i], alpha2p_vector[i], alpha4p_vector[i], st0p_vector[i], st2p_vector[i], st4p_vector[i]]
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_used[tracer]); st2e = st2p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^2; st4e = st4p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one vector for the FS theory function
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS-BAO joint theory vector given the parameters
        prediction = iΓ_FS_BAO_all[tracer] * vcat(wmat_used[tracer]*theory_FS(cosmo_eft_params, emus_FS[tracer], kin_used[tracer]), 
                                                  theory_BAO(cosmo_params, emu_BAO, zeff_used[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction = iΓ_Lya * theory_BAO(cosmo_params, emu_BAO, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction, I)
    # Adds in the BBN/ns10 priors (since not including CMB)
    ns_prior = [0.9649, 0.042]
    ωb_prior = [0.02218, 0.00055]
    sigma_ωb = ωb_prior[2]
    dωb = cosmo_params[4] - ωb_prior[1]
    sigma_ns = ns_prior[2]
    dns = cosmo_params[2] - ns_prior[1]
    Turing.@addlogprob! - 0.5 * dωb^2/sigma_ωb^2
    Turing.@addlogprob! - 0.5 * dns^2/sigma_ns^2
    return nothing
end

@everywhere @model function model_CMB(D_CMB, iΓ_CMB, fixed_value, param, variation, emus_CMB)
    # Draws cosmological parameters
    cosmo_params = Vector{Any}(undef, 9) # 9 cosmological parameters: ln10As, ns, H0, ωb, ωc, τ, w0, wa, mν (reorders later)
    cosmo_params[9] = 0.06 # fixes the neutrino mass
    if variation == "LCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "τ"]
        cosmo_ranges = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "τ" => [0.02, 0.12])
        cosmo_params[7], cosmo_params[8] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
    elseif variation == "w0waCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "w0", "wa"]
        cosmo_ranges = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "τ" => [0.02, 0.12],
                            "w0" => [-2, 0.49], "wa" => [-3, 1.64])
    end
    for i in 1:length(cosmo_params_list)
        if cosmo_params_list[i] == param
            cosmo_params[i] = fixed_value
        else
            if cosmo_params_list[i] == "τ"
                cosmo_params[i] ~ Normal(0.0506, 0.0086)
            else
                cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
            end
        end
    end
    yₚ ~ Normal(1.0, 0.0025)
    # Constructs the parameter vector given the samples drawn above (changes order to match the emulator)
    cosmo_params = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], cosmo_params[5],
                    cosmo_params[6], cosmo_params[9], cosmo_params[7], cosmo_params[8]]
    # Constructs theory vector and associates data vector with model
    prediction = iΓ_CMB * theory_CMB(cosmo_params, emus_CMB)
    D_CMB ~ MvNormal(prediction, I)
    return nothing
end
                
@everywhere @model function model_FS_BAO_CMB(D_FS_BAO_all, iΓ_FS_BAO_all, D_Lya, iΓ_Lya, D_CMB, iΓ_CMB, fixed_value, param, variation, emus_FS, emu_BAO, emus_CMB, tracer_vector, zeff_used, nd_used, fsat_used, sigv_used, kin_used, wmat_used)
    # Draws cosmological parameters
    cosmo_params = Vector{Any}(undef, 9) # 9 cosmological parameters: ln10As, ns, H0, ωb, ωc, τ, w0, wa, mν (reorders later)
    cosmo_params[9] = 0.06 # fixes neutrino mass
    if variation == "LCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "τ"]
        cosmo_ranges = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "τ" => [0.02, 0.12])
        cosmo_params[7], cosmo_params[8] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
    elseif variation == "w0waCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "w0", "wa"]
        cosmo_ranges = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "τ" => [0.02, 0.12],
                            "w0" => [-2, 0.49], "wa" => [-3, 1.64])
    end
    for i in 1:length(cosmo_params_list)
        if cosmo_params_list[i] == param
            cosmo_params[i] = fixed_value
        else
            if cosmo_params_list[i] == "τ"
                cosmo_params[i] ~ Normal(0.0506, 0.0086)
            else
                cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
            end
        end
    end
    yₚ ~ Normal(1.0, 0.0025)
    # Constructs the parameter vector given the samples drawn above (changes order to match the emulator)
    cosmo_params_LSS = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], cosmo_params[5], 
                        cosmo_params[7], cosmo_params[8]]
    cosmo_params_CMB = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], cosmo_params[5],
                        cosmo_params[6], cosmo_params[9], cosmo_params[7], cosmo_params[8]]
    # Extracts f and sigma8 values for each of the tracers using BAO emulator
    f_sigma8 = Effort.get_BAO(cosmo_params_LSS, emu_BAO)
    f_all = Dict("BGS" => f_sigma8[2], "LRG1" => f_sigma8[3], "LRG2" => f_sigma8[4], "LRG3" => f_sigma8[5], "ELG2" => f_sigma8[7], "QSO" => f_sigma8[8])
    sigma8_all = Dict("BGS" => f_sigma8[9], "LRG1" => f_sigma8[10], "LRG2" => f_sigma8[11], "LRG3" => f_sigma8[12], "ELG2" => f_sigma8[14], "QSO" => f_sigma8[15])
    # Draws EFT nuisance parameters
    eft_ranges = Dict("b1p" => [0, 6], "b2p" =>  [-15, 5], "bsp" => [-10, 15], "alpha0p" => [-100, 400], "alpha2p" => [-800, 200], "st0p" => [-80, 80], "st2p" => [-200, 200])
    b1p_vector = Vector{Any}(undef, length(tracer_vector))
    b2p_vector = Vector{Any}(undef, length(tracer_vector))
    b3p_vector = Vector{Any}(undef, length(tracer_vector))
    bsp_vector = Vector{Any}(undef, length(tracer_vector))
    alpha0p_vector = Vector{Any}(undef, length(tracer_vector))
    alpha2p_vector = Vector{Any}(undef, length(tracer_vector))
    alpha4p_vector = Vector{Any}(undef, length(tracer_vector))
    st0p_vector = Vector{Any}(undef, length(tracer_vector))
    st2p_vector = Vector{Any}(undef, length(tracer_vector))
    st4p_vector = Vector{Any}(undef, length(tracer_vector))
    # Iterates through each of the tracers
    for i in 1:length(tracer_vector)
        tracer = tracer_vector[i]
        b1p_vector[i] ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
        b2p_vector[i] ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
        b3p_vector[i] = 0 
        bsp_vector[i] ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
        alpha0p_vector[i] ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
        alpha2p_vector[i] ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
        alpha4p_vector[i] = 0
        st0p_vector[i] ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
        st2p_vector[i] ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
        st4p_vector[i] = 0
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = [b1p_vector[i], b2p_vector[i], b3p_vector[i], bsp_vector[i],
                                                                            alpha0p_vector[i], alpha2p_vector[i], alpha4p_vector[i], st0p_vector[i], st2p_vector[i], st4p_vector[i]]
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_used[tracer]); st2e = st2p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^2; st4e = st4p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one vector for the FS theory function
        cosmo_eft_params = vcat(cosmo_params_LSS, eft_params)
        # Calculates FS-BAO joint theory vector given the parameters
        prediction = iΓ_FS_BAO_all[tracer] * vcat(wmat_used[tracer]*theory_FS(cosmo_eft_params, emus_FS[tracer], kin_used[tracer]), 
                                                theory_BAO(cosmo_params_LSS, emu_BAO, zeff_used[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction = iΓ_Lya * theory_BAO(cosmo_params_LSS, emu_BAO, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction, I)
    # Constructs theory vector and associates data with model
    prediction = iΓ_CMB * theory_CMB(cosmo_params_CMB, emus_CMB)
    D_CMB ~ MvNormal(prediction, I)
    return nothing
    # Does not add BBN or ns10 priors since including CMB data
end

@everywhere @model function model_SN(D_SN, iΓ_SN, z_SN, fixed_value, param, variation)
    # Draws cosmological parameters
    cosmo_params = Vector{Any}(undef, 7) # 7 cosmological parameters: ln10As, ns, H0, ωb, ωc, w0, wa
    if variation == "LCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc"]
        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25])
        cosmo_params[6], cosmo_params[7] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
    elseif variation == "w0waCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
        cosmo_ranges = Dict("ln10As" => [2.0, 3.5], "ns" => [0.8, 1.1], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.25],
                            "w0" => [-2, 0.5], "wa" => [-3, 1.64])
    end
    for i in 1:length(cosmo_params_list)
        if cosmo_params_list[i] == param
            cosmo_params[i] = fixed_value
        else
            cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
        end
    end
    # Constructs the parameter vector given the samples drawn above
    cosmo_params = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], 
                    cosmo_params[5], cosmo_params[6], cosmo_params[7]]
    Mb ~ Uniform(-5, 5)
    # Constructs theory vector and associates data vector with model
    prediction = iΓ_SN * theory_SN(cosmo_params, Mb, z_SN)
    D_SN ~ MvNormal(prediction, I)
    return nothing
end

@everywhere @model function model_FS_BAO_CMB_SN(D_FS_BAO_all, iΓ_FS_BAO_all, D_Lya, iΓ_Lya, D_CMB, iΓ_CMB, D_SN, iΓ_SN, z_SN, fixed_value, param, variation, emus_FS, emu_BAO, emus_CMB, tracer_vector, zeff_used, nd_used, fsat_used, sigv_used, kin_used, wmat_used)
    # Draws cosmological parameters
    cosmo_params = Vector{Any}(undef, 9) # 9 cosmological parameters: ln10As, ns, H0, ωb, ωc, τ, w0, wa, mν (reorders later)
    cosmo_params[9] = 0.06 # fixes neutrino mass
    if variation == "LCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "τ"]
        cosmo_ranges = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "τ" => [0.02, 0.12])
        cosmo_params[7], cosmo_params[8] = -1, 0 # sets w0, wa = -1, 0 for LCDM specifically
    elseif variation == "w0waCDM"
        cosmo_params_list = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "w0", "wa"]
        cosmo_ranges = Dict("ln10As" => [2.5, 3.5], "ns" => [0.88, 1.05], "H0" => [50, 80], "ωb" => [0.02, 0.025], "ωc" => [0.09, 0.2], "τ" => [0.02, 0.12],
                            "w0" => [-2, 0.49], "wa" => [-3, 1.64])
    end
    for i in 1:length(cosmo_params_list)
        if cosmo_params_list[i] == param
            cosmo_params[i] = fixed_value
        else
            if cosmo_params_list[i] == "τ"
                cosmo_params[i] ~ Normal(0.0506, 0.0086)
            else
                cosmo_params[i] ~ Uniform(cosmo_ranges[cosmo_params_list[i]][1], cosmo_ranges[cosmo_params_list[i]][2])
            end
        end
    end
    yₚ ~ Normal(1.0, 0.0025)
    # Constructs the parameter vector given the samples drawn above (changes order to match the emulator)
    cosmo_params_LSS = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], cosmo_params[5], 
                        cosmo_params[7], cosmo_params[8]]
    cosmo_params_CMB = [cosmo_params[1], cosmo_params[2], cosmo_params[3], cosmo_params[4], cosmo_params[5],
                        cosmo_params[6], cosmo_params[9], cosmo_params[7], cosmo_params[8]]
    Mb ~ Uniform(-5, 5)
    # Extracts f and sigma8 values for each of the tracers using BAO emulator
    f_sigma8 = Effort.get_BAO(cosmo_params_LSS, emu_BAO)
    f_all = Dict("BGS" => f_sigma8[2], "LRG1" => f_sigma8[3], "LRG2" => f_sigma8[4], "LRG3" => f_sigma8[5], "ELG2" => f_sigma8[7], "QSO" => f_sigma8[8])
    sigma8_all = Dict("BGS" => f_sigma8[9], "LRG1" => f_sigma8[10], "LRG2" => f_sigma8[11], "LRG3" => f_sigma8[12], "ELG2" => f_sigma8[14], "QSO" => f_sigma8[15])
    # Draws EFT nuisance parameters
    eft_ranges = Dict("b1p" => [0, 6], "b2p" =>  [-15, 5], "bsp" => [-10, 15], "alpha0p" => [-100, 400], "alpha2p" => [-800, 200], "st0p" => [-80, 80], "st2p" => [-200, 200])
    b1p_vector = Vector{Any}(undef, length(tracer_vector))
    b2p_vector = Vector{Any}(undef, length(tracer_vector))
    b3p_vector = Vector{Any}(undef, length(tracer_vector))
    bsp_vector = Vector{Any}(undef, length(tracer_vector))
    alpha0p_vector = Vector{Any}(undef, length(tracer_vector))
    alpha2p_vector = Vector{Any}(undef, length(tracer_vector))
    alpha4p_vector = Vector{Any}(undef, length(tracer_vector))
    st0p_vector = Vector{Any}(undef, length(tracer_vector))
    st2p_vector = Vector{Any}(undef, length(tracer_vector))
    st4p_vector = Vector{Any}(undef, length(tracer_vector))
    # Iterates through each of the tracers
    for i in 1:length(tracer_vector)
        tracer = tracer_vector[i]
        b1p_vector[i] ~ Uniform(eft_ranges["b1p"][1], eft_ranges["b1p"][2])
        b2p_vector[i] ~ Uniform(eft_ranges["b2p"][1], eft_ranges["b2p"][2])
        b3p_vector[i] = 0 
        bsp_vector[i] ~ Uniform(eft_ranges["bsp"][1], eft_ranges["bsp"][2])
        alpha0p_vector[i] ~ Uniform(eft_ranges["alpha0p"][1], eft_ranges["alpha0p"][2])
        alpha2p_vector[i] ~ Uniform(eft_ranges["alpha2p"][1], eft_ranges["alpha2p"][2])
        alpha4p_vector[i] = 0
        st0p_vector[i] ~ Uniform(eft_ranges["st0p"][1], eft_ranges["st0p"][2])
        st2p_vector[i] ~ Uniform(eft_ranges["st2p"][1], eft_ranges["st2p"][2])
        st4p_vector[i] = 0
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = [b1p_vector[i], b2p_vector[i], b3p_vector[i], bsp_vector[i],
                                                                            alpha0p_vector[i], alpha2p_vector[i], alpha4p_vector[i], st0p_vector[i], st2p_vector[i], st4p_vector[i]]
        # Converts physical to Eulerian basis
        f, sigma8 = f_all[tracer], sigma8_all[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_used[tracer]); st2e = st2p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^2; st4e = st4p/(nd_used[tracer])*(fsat_used[tracer])*(sigv_used[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one vector for the FS theory function
        cosmo_eft_params = vcat(cosmo_params_LSS, eft_params)
        # Calculates FS-BAO joint theory vector given the parameters
        prediction = iΓ_FS_BAO_all[tracer] * vcat(wmat_used[tracer]*theory_FS(cosmo_eft_params, emus_FS[tracer], kin_used[tracer]), 
                                                theory_BAO(cosmo_params_LSS, emu_BAO, zeff_used[tracer], tracer))
        D_FS_BAO_all[tracer] ~ MvNormal(prediction, I)
    end
    # Adds Lya BAO as a stand alone (since uncorrelated with other tracers)
    prediction = iΓ_Lya * theory_BAO(cosmo_params_LSS, emu_BAO, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction, I)
    # Constructs CMB theory vector and associates data with model
    prediction = iΓ_CMB * theory_CMB(cosmo_params_CMB, emus_CMB)
    D_CMB ~ MvNormal(prediction, I)
    # Constructs SN theory vector and associates data with model
    prediction = iΓ_SN * theory_SN(cosmo_params_LSS, Mb, z_SN)
    D_SN ~ MvNormal(prediction, I)
    return nothing
    # Does not add BBN or ns10 priors since including CMB data
end

@everywhere function run_worker(fixed_value)
    # Runs the worker for a given parameter value in the profile likelihood.
    # (Performs LBFGS minimization of chi-squared).

    if dataset == "FS"
        if variation == "LCDM"
            if param == "ln10As"
                fit_model = model_FS(D_FS_all) | (ln10As=fixed_value, w0=-1, wa=0)
            elseif param == "H0"
                fit_model = model_FS(D_FS_all) | (H0=fixed_value, w0=-1, wa=0)
            elseif param == "Ωm"
                fit_model = model_FS(D_FS_all) | (ωc=fixed_value, w0=-1, wa=0)
            end
        elseif variation == "w0waCDM"
            if param == "ln10As"
                fit_model = model_FS(D_FS_all) | (ln10As=fixed_value,)
            elseif param == "H0"
                fit_model = model_FS(D_FS_all) | (H0=fixed_value,)
            elseif param == "Ωm"
                fit_model = model_FS(D_FS_all) | (ωc=fixed_value,)
            elseif param == "w0"
                fit_model = model_FS(D_FS_all) | (w0=fixed_value,)
            elseif param == "wa"
                fit_model = model_FS(D_FS_all) | (wa=fixed_value,)
            end
        end
    end

    profile_values_array = SharedArray{Float64}(n_runs)
    if dataset == "BAO"
        bestfit_values_array = SharedArray{Float64}((ncosmo_tot-1), n_runs)
    else
        bestfit_values_array = SharedArray{Float64}((ncosmo_tot-1)+7*size(tracer_vector)[1], n_runs)
    end

    for i in 1:n_runs
        fit_result = maximum_a_posteriori(fit_model, LBFGS(m=50))#, P=Diagonal([1/0.01, 1/1, 1/0.0001, 1/0.01, 1/0.1, 1/1, 1/1, 1/10, 1/10, 1/1, 1/1, 1/0.1, 1/1, 1/1, 1/10, 1/10, 1/1, 1/1, 1/0.1, 1/1, 1/1, 1/10, 1/10, 1/1, 1/1, 1/0.1, 1/1, 1/1, 1/10, 1/10, 1/1, 1/1, 1/0.1, 1/1, 1/1, 1/10, 1/10, 1/1, 1/1, 1/0.1, 1/1, 1/1, 1/10, 1/10, 1/1, 1/1])))#, P=preconditioning_matrix)) # add initial values if needed!###########################################################
        profile_values_array[i] = fit_result.lp
        bestfit_values_array[:, i] = fit_result.values.array
    end
    return [profile_values_array, bestfit_values_array]
end

    ## Specifies the given model for the worker
    #if dataset == "FS" # change these to the conditioning syntax##################################################################################################
    #    fit_model = model_FS(D_LSS_all, iΓ_LSS_all, fixed_value, param, variation, FS_emus, BAO_emu, tracer_vector, nd_all, fsat_all, sigv_all, kin_all, wmat_all)
    #elseif dataset == "BAO"
    #    fit_model = model_BAO(D_LSS_all, iΓ_LSS_all, D_Lya, iΓ_Lya, fixed_value, param, variation, BAO_emu, tracer_vector, zeff_all)
    #elseif dataset == "FS+BAO"
    #    fit_model = model_FS_BAO(D_LSS_all, iΓ_LSS_all, D_Lya, iΓ_Lya, fixed_value, param, variation, FS_emus, BAO_emu, tracer_vector, zeff_all, nd_all, fsat_all, sigv_all, kin_all, wmat_all)
    #elseif dataset == "FS+BAO+CMB"
    #    fit_model = model_FS_BAO_CMB(D_LSS_all, iΓ_LSS_all, D_Lya, iΓ_Lya, D_CMB, iΓ_CMB, fixed_vaue, param, variation, FS_emus, BAO_emu, CMB_emus, tracer_vector, zeff_all, nd_all, fsat_all, sigv_all, kin_all, wmat_all)
    #elseif dataset == "FS+BAO+CMB+SN"
    #    if SN_type == "DESY5"
    #        D_SN = D_DESY5SN
    #        iΓ_SN = iΓ_DESY5SN
    #        z_SN = z_DESY5SN
    #    elseif SN_type == "PantheonPlus"
    #        D_SN = D_PantheonPlusSN
    #        iΓ_SN = iΓ_PantheonPlusSN
    #        z_SN = z_PantheonPlusSN
    #    elseif SN_type == "Union3"
    #        D_SN = D_Union3SN
    #        iΓ_SN = iΓ_Union3SN
    #        z_SN = z_Union3SN
    #    end
    #    fit_model = model_FS_BAO_CMB_SN(D_LSS_all, iΓ_LSS_all, D_Lya, iΓ_Lya, D_CMB, iΓ_CMB, D_SN, iΓ_SN, z_SN, fixed_value, param, variation, FS_emus, BAO_emu, CMB_emus, tracer_vector, zeff_all, nd_all, fsat_all, sigv_all, kin_all, wmat_all)
    #end##

    ## Specifies the number of bestfit parameters
    #if dataset in ["FS", "BAO", "FS+BAO"]
    #    if variation == "LCDM"
    #        ncosmo = 5
    #    elseif variation == "w0waCDM"
    #        ncosmo = 7
    #    end
    #elseif dataset == "FS+BAO+CMB"
    #    if variation == "LCDM"
    #        ncosmo = 8
    #    elseif variation == "w0waCDM"
    #        ncosmo = 10
    #    end
    #elseif dataset == "FS+BAO+CMB+SN"
    #    if variation == "LCDM"
    #        ncosmo = 9
    #    elseif variation == "w0waCDM"
    #        ncosmo = 11
    #    end
    #end
    #profile_values_array = SharedArray{Float64}(n_runs)
    #if dataset == "BAO"
    #    bestfit_values_array = SharedArray{Float64}((ncosmo-1), n_runs)
    #else
    #    bestfit_values_array = SharedArray{Float64}((ncosmo-1)+7*size(tracer_vector)[1], n_runs)
    #end#

    ## Initiates the given preconditioning matrix (to help with LBFGS convergence)
    #preconditioning_steps = Dict("ln10As" => 0.1, "ns" => 0.01, "H0" => 1, "ωb" => 0.0001, "ωc" => 0.01, "τ" => 0.01, "w0" => 1, "wa" => 1, "yₚ" => 0.001, "Mb" => 0.1, "b1p" => 0.1, "b2p" => 1, "bsp" => 1, "alpha0p" => 10, "alpha2p" => 10, "st0p" => 1, "st2p" => 1)##############################################
    #if dataset in ["FS", "BAO", "FS+BAO"]
    #    if variation == "LCDM"
    #        preconditioning_entries = ["ln10As", "ns", "H0", "ωb", "ωc"]
    #    elseif variation == "w0waCDM"
    #        preconditioning_entries = ["ln10As", "ns", "H0", "ωb", "ωc", "w0", "wa"]
    #    end
    #elseif dataset == "FS+BAO+CMB"
    #    if variation == "LCDM"
    #        preconditioning_entries = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ"]
    #    elseif variation == "w0waCDM"
    #        preconditioning_entries = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "w0", "wa", "yₚ"]
    #    end
    #elseif dataset == "FS+BAO+CMB+SN"
    #    if variation == "LCDM"
    #        preconditioning_entries = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "yₚ", "Mb"]
    #    elseif variation == "w0waCDM"
    #        preconditioning_entries = ["ln10As", "ns", "H0", "ωb", "ωc", "τ", "w0", "wa", "yₚ", "Mb"]
    #    end
    #end
    #if dataset != "BAO"
    #    for i in 1:length(tracer_vector)
    #        append!(preconditioning_entries, ["b1p", "b2p", "bsp", "alpha0p", "alpha2p", "st0p", "st2p"]) # adds EFT parameters to list of preconditioning parameters (in all cases)
    #    end
    #end
    ## Removes the parameter actually being queried
    #filter!(x -> x ≠ param, preconditioning_entries)
    ## Actually constructs the preconditioning matrix
    #preconditioning_diagonal_entries = [1/preconditioning_steps[preconditioning_entries[i]] for i in 1:length(preconditioning_entries)]
    #preconditioning_matrix = Diagonal(preconditioning_diagonal_entries)#

    ## Performs many independent runs in order to reach global minima
    #for i in 1:n_runs
    #    fit_result = maximum_likelihood(fit_model, LBFGS(m=30))#, P=preconditioning_matrix)) # add initial values if needed!
    #    profile_values_array[i] = fit_result.lp
    #    bestfit_values_array[:, i] = fit_result.values.array
    #end
    ## Returns array of all best fits for each of the different runs
    #return [profile_values_array, bestfit_values_array]
#end

main()