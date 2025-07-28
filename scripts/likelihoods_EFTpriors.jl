using Pkg
Pkg.activate(".")
include("priors.jl")
include("y1_datasets.jl")
include("theory_models.jl")
include("emulators.jl")


@model function model_FS_bay(D_FS_dict)
    """Likelihood for full-shape only."""
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
    f_dict = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                  "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_dict = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                       "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_priors["b1p_BGS"][1], eft_priors["b1p_BGS"][2])
            b2p_BGS ~ Normal(eft_priors["b2p_BGS"][1], eft_priors["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Normal(eft_priors["bsp_BGS"][1], eft_priors["bsp_BGS"][2])
            alpha0p_BGS ~ Normal(eft_priors["alpha0p_BGS"][1], eft_priors["alpha0p_BGS"][2])
            alpha2p_BGS ~ Normal(eft_priors["alpha2p_BGS"][1], eft_priors["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Normal(eft_priors["st0p_BGS"][1], eft_priors["st0p_BGS"][2])
            st2p_BGS ~ Normal(eft_priors["st2p_BGS"][1], eft_priors["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_priors["b1p_LRG1"][1], eft_priors["b1p_LRG1"][2])
            b2p_LRG1 ~ Normal(eft_priors["b2p_LRG1"][1], eft_priors["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Normal(eft_priors["bsp_LRG1"][1], eft_priors["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Normal(eft_priors["alpha0p_LRG1"][1], eft_priors["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Normal(eft_priors["alpha2p_LRG1"][1], eft_priors["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Normal(eft_priors["st0p_LRG1"][1], eft_priors["st0p_LRG1"][2])
            st2p_LRG1 ~ Normal(eft_priors["st2p_LRG1"][1], eft_priors["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_priors["b1p_LRG2"][1], eft_priors["b1p_LRG2"][2])
            b2p_LRG2 ~ Normal(eft_priors["b2p_LRG2"][1], eft_priors["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Normal(eft_priors["bsp_LRG2"][1], eft_priors["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Normal(eft_priors["alpha0p_LRG2"][1], eft_priors["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Normal(eft_priors["alpha2p_LRG2"][1], eft_priors["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Normal(eft_priors["st0p_LRG2"][1], eft_priors["st0p_LRG2"][2])
            st2p_LRG2 ~ Normal(eft_priors["st2p_LRG2"][1], eft_priors["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_priors["b1p_LRG3"][1], eft_priors["b1p_LRG3"][2])
            b2p_LRG3 ~ Normal(eft_priors["b2p_LRG3"][1], eft_priors["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Normal(eft_priors["bsp_LRG3"][1], eft_priors["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Normal(eft_priors["alpha0p_LRG3"][1], eft_priors["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Normal(eft_priors["alpha2p_LRG3"][1], eft_priors["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Normal(eft_priors["st0p_LRG3"][1], eft_priors["st0p_LRG3"][2])
            st2p_LRG3 ~ Normal(eft_priors["st2p_LRG3"][1], eft_priors["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_priors["b1p_ELG2"][1], eft_priors["b1p_ELG2"][2])
            b2p_ELG2 ~ Normal(eft_priors["b2p_ELG2"][1], eft_priors["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Normal(eft_priors["bsp_ELG2"][1], eft_priors["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Normal(eft_priors["alpha0p_ELG2"][1], eft_priors["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Normal(eft_priors["alpha2p_ELG2"][1], eft_priors["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Normal(eft_priors["st0p_ELG2"][1], eft_priors["st0p_ELG2"][2])
            st2p_ELG2 ~ Normal(eft_priors["st2p_ELG2"][1], eft_priors["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_priors["b1p_QSO"][1], eft_priors["b1p_QSO"][2])
            b2p_QSO ~ Normal(eft_priors["b2p_QSO"][1], eft_priors["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Normal(eft_priors["bsp_QSO"][1], eft_priors["bsp_QSO"][2])
            alpha0p_QSO ~ Normal(eft_priors["alpha0p_QSO"][1], eft_priors["alpha0p_QSO"][2])
            alpha2p_QSO ~ Normal(eft_priors["alpha2p_QSO"][1], eft_priors["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Normal(eft_priors["st0p_QSO"][1], eft_priors["st0p_QSO"][2])
            st2p_QSO ~ Normal(eft_priors["st2p_QSO"][1], eft_priors["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_dict[tracer], sigma8_dict[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_dict[tracer]); st2e = st2p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^2; st4e = st4p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS theory vector given parameters
        prediction_FS = iΓ_FS_dict[tracer]*(wmat_dict[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_dict[tracer]))
        D_FS_dict[tracer] ~ MvNormal(prediction_FS, I)
    end
end

@model function model_BAO_bay(D_BAO_dict, D_Lya)
    """Likelihood for BAO only."""
    # Draws cosmological parameters
    ln10As = 3.044 # ln10As and ns not fit for BAO only (set to constant values for emulator)
    ns = 0.9649               
    H0 ~ Uniform(cosmo_ranges_FS_BAO["H0"][1], cosmo_ranges_FS_BAO["H0"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges_FS_BAO["ωb"][1], cosmo_ranges_FS_BAO["ωb"][2])            
    ωc ~ Uniform(cosmo_ranges_FS_BAO["ωc"][1], cosmo_ranges_FS_BAO["ωc"][2])
    w0 ~ Uniform(cosmo_ranges_FS_BAO["w0"][1], cosmo_ranges_FS_BAO["w0"][2])
    wa ~ Uniform(cosmo_ranges_FS_BAO["wa"][1], cosmo_ranges_FS_BAO["wa"][2])
    cosmo_params = [ln10As, ns, H0, ωb, ωc, w0, wa]
    for tracer in tracer_vector
        prediction_BAO = iΓ_BAO_dict[tracer] * theory_BAO(cosmo_params, BAO_emu, redshift_eff[tracer], tracer)
        D_BAO_dict[tracer] ~ MvNormal(prediction_BAO, I)
    end
    # Adds Lya BAO as a stand-alone (uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
end

@model function model_FS_BAO_bay(D_FS_BAO_dict, D_Lya)
    """Likelihood for full-shape and BAO joint."""
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
    f_dict = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                  "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_dict = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                       "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_priors["b1p_BGS"][1], eft_priors["b1p_BGS"][2])
            b2p_BGS ~ Normal(eft_priors["b2p_BGS"][1], eft_priors["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Normal(eft_priors["bsp_BGS"][1], eft_priors["bsp_BGS"][2])
            alpha0p_BGS ~ Normal(eft_priors["alpha0p_BGS"][1], eft_priors["alpha0p_BGS"][2])
            alpha2p_BGS ~ Normal(eft_priors["alpha2p_BGS"][1], eft_priors["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Normal(eft_priors["st0p_BGS"][1], eft_priors["st0p_BGS"][2])
            st2p_BGS ~ Normal(eft_priors["st2p_BGS"][1], eft_priors["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_priors["b1p_LRG1"][1], eft_priors["b1p_LRG1"][2])
            b2p_LRG1 ~ Normal(eft_priors["b2p_LRG1"][1], eft_priors["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Normal(eft_priors["bsp_LRG1"][1], eft_priors["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Normal(eft_priors["alpha0p_LRG1"][1], eft_priors["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Normal(eft_priors["alpha2p_LRG1"][1], eft_priors["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Normal(eft_priors["st0p_LRG1"][1], eft_priors["st0p_LRG1"][2])
            st2p_LRG1 ~ Normal(eft_priors["st2p_LRG1"][1], eft_priors["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_priors["b1p_LRG2"][1], eft_priors["b1p_LRG2"][2])
            b2p_LRG2 ~ Normal(eft_priors["b2p_LRG2"][1], eft_priors["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Normal(eft_priors["bsp_LRG2"][1], eft_priors["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Normal(eft_priors["alpha0p_LRG2"][1], eft_priors["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Normal(eft_priors["alpha2p_LRG2"][1], eft_priors["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Normal(eft_priors["st0p_LRG2"][1], eft_priors["st0p_LRG2"][2])
            st2p_LRG2 ~ Normal(eft_priors["st2p_LRG2"][1], eft_priors["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_priors["b1p_LRG3"][1], eft_priors["b1p_LRG3"][2])
            b2p_LRG3 ~ Normal(eft_priors["b2p_LRG3"][1], eft_priors["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Normal(eft_priors["bsp_LRG3"][1], eft_priors["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Normal(eft_priors["alpha0p_LRG3"][1], eft_priors["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Normal(eft_priors["alpha2p_LRG3"][1], eft_priors["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Normal(eft_priors["st0p_LRG3"][1], eft_priors["st0p_LRG3"][2])
            st2p_LRG3 ~ Normal(eft_priors["st2p_LRG3"][1], eft_priors["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_priors["b1p_ELG2"][1], eft_priors["b1p_ELG2"][2])
            b2p_ELG2 ~ Normal(eft_priors["b2p_ELG2"][1], eft_priors["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Normal(eft_priors["bsp_ELG2"][1], eft_priors["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Normal(eft_priors["alpha0p_ELG2"][1], eft_priors["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Normal(eft_priors["alpha2p_ELG2"][1], eft_priors["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Normal(eft_priors["st0p_ELG2"][1], eft_priors["st0p_ELG2"][2])
            st2p_ELG2 ~ Normal(eft_priors["st2p_ELG2"][1], eft_priors["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_priors["b1p_QSO"][1], eft_priors["b1p_QSO"][2])
            b2p_QSO ~ Normal(eft_priors["b2p_QSO"][1], eft_priors["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Normal(eft_priors["bsp_QSO"][1], eft_priors["bsp_QSO"][2])
            alpha0p_QSO ~ Normal(eft_priors["alpha0p_QSO"][1], eft_priors["alpha0p_QSO"][2])
            alpha2p_QSO ~ Normal(eft_priors["alpha2p_QSO"][1], eft_priors["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Normal(eft_priors["st0p_QSO"][1], eft_priors["st0p_QSO"][2])
            st2p_QSO ~ Normal(eft_priors["st2p_QSO"][1], eft_priors["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_dict[tracer], sigma8_dict[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_dict[tracer]); st2e = st2p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^2; st4e = st4p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_dict[tracer]*vcat(wmat_dict[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_dict[tracer]),
                                                        theory_BAO(cosmo_params, BAO_emu, redshift_eff[tracer], tracer))
        D_FS_BAO_dict[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand-alone (uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
end

@model function model_FS_BAO_CMB_bay(D_FS_BAO_dict, D_Lya, D_CMB)
    """Likelihood for full-shape, BAO and CMB joint."""
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
    f_dict = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                  "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_dict = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                       "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_priors["b1p_BGS"][1], eft_priors["b1p_BGS"][2])
            b2p_BGS ~ Normal(eft_priors["b2p_BGS"][1], eft_priors["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Normal(eft_priors["bsp_BGS"][1], eft_priors["bsp_BGS"][2])
            alpha0p_BGS ~ Normal(eft_priors["alpha0p_BGS"][1], eft_priors["alpha0p_BGS"][2])
            alpha2p_BGS ~ Normal(eft_priors["alpha2p_BGS"][1], eft_priors["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Normal(eft_priors["st0p_BGS"][1], eft_priors["st0p_BGS"][2])
            st2p_BGS ~ Normal(eft_priors["st2p_BGS"][1], eft_priors["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_priors["b1p_LRG1"][1], eft_priors["b1p_LRG1"][2])
            b2p_LRG1 ~ Normal(eft_priors["b2p_LRG1"][1], eft_priors["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Normal(eft_priors["bsp_LRG1"][1], eft_priors["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Normal(eft_priors["alpha0p_LRG1"][1], eft_priors["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Normal(eft_priors["alpha2p_LRG1"][1], eft_priors["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Normal(eft_priors["st0p_LRG1"][1], eft_priors["st0p_LRG1"][2])
            st2p_LRG1 ~ Normal(eft_priors["st2p_LRG1"][1], eft_priors["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_priors["b1p_LRG2"][1], eft_priors["b1p_LRG2"][2])
            b2p_LRG2 ~ Normal(eft_priors["b2p_LRG2"][1], eft_priors["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Normal(eft_priors["bsp_LRG2"][1], eft_priors["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Normal(eft_priors["alpha0p_LRG2"][1], eft_priors["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Normal(eft_priors["alpha2p_LRG2"][1], eft_priors["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Normal(eft_priors["st0p_LRG2"][1], eft_priors["st0p_LRG2"][2])
            st2p_LRG2 ~ Normal(eft_priors["st2p_LRG2"][1], eft_priors["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_priors["b1p_LRG3"][1], eft_priors["b1p_LRG3"][2])
            b2p_LRG3 ~ Normal(eft_priors["b2p_LRG3"][1], eft_priors["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Normal(eft_priors["bsp_LRG3"][1], eft_priors["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Normal(eft_priors["alpha0p_LRG3"][1], eft_priors["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Normal(eft_priors["alpha2p_LRG3"][1], eft_priors["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Normal(eft_priors["st0p_LRG3"][1], eft_priors["st0p_LRG3"][2])
            st2p_LRG3 ~ Normal(eft_priors["st2p_LRG3"][1], eft_priors["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_priors["b1p_ELG2"][1], eft_priors["b1p_ELG2"][2])
            b2p_ELG2 ~ Normal(eft_priors["b2p_ELG2"][1], eft_priors["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Normal(eft_priors["bsp_ELG2"][1], eft_priors["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Normal(eft_priors["alpha0p_ELG2"][1], eft_priors["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Normal(eft_priors["alpha2p_ELG2"][1], eft_priors["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Normal(eft_priors["st0p_ELG2"][1], eft_priors["st0p_ELG2"][2])
            st2p_ELG2 ~ Normal(eft_priors["st2p_ELG2"][1], eft_priors["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_priors["b1p_QSO"][1], eft_priors["b1p_QSO"][2])
            b2p_QSO ~ Normal(eft_priors["b2p_QSO"][1], eft_priors["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Normal(eft_priors["bsp_QSO"][1], eft_priors["bsp_QSO"][2])
            alpha0p_QSO ~ Normal(eft_priors["alpha0p_QSO"][1], eft_priors["alpha0p_QSO"][2])
            alpha2p_QSO ~ Normal(eft_priors["alpha2p_QSO"][1], eft_priors["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Normal(eft_priors["st0p_QSO"][1], eft_priors["st0p_QSO"][2])
            st2p_QSO ~ Normal(eft_priors["st2p_QSO"][1], eft_priors["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_dict[tracer], sigma8_dict[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_dict[tracer]); st2e = st2p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^2; st4e = st4p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_dict[tracer]*vcat(wmat_dict[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_dict[tracer]),
                                                        theory_BAO(cosmo_params_FS_BAO, BAO_emu, redshift_eff[tracer], tracer))
        D_FS_BAO_dict[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand-alone (uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
end

@model function model_FS_BAO_CMB_SN_bay(D_FS_BAO_dict, D_Lya, D_CMB, iΓ_SN, D_SN, z_SN, SN_type)
    """Likelihood for full-shape, BAO, CMB and SN joint."""
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
    f_dict = Dict("BGS" => fsigma8_info[2], "LRG1" => fsigma8_info[3], "LRG2" => fsigma8_info[4], "LRG3" => fsigma8_info[5], 
                  "ELG2" => fsigma8_info[7], "QSO" => fsigma8_info[8])
    sigma8_dict = Dict("BGS" => fsigma8_info[9], "LRG1" => fsigma8_info[10], "LRG2" => fsigma8_info[11], "LRG3" => fsigma8_info[12], 
                       "ELG2" => fsigma8_info[14], "QSO" => fsigma8_info[15])
    # Iterates through each tracer
    for tracer in tracer_vector
        if tracer == "BGS"
            b1p_BGS ~ Uniform(eft_priors["b1p_BGS"][1], eft_priors["b1p_BGS"][2])
            b2p_BGS ~ Normal(eft_priors["b2p_BGS"][1], eft_priors["b2p_BGS"][2])
            b3p_BGS = 0
            bsp_BGS ~ Normal(eft_priors["bsp_BGS"][1], eft_priors["bsp_BGS"][2])
            alpha0p_BGS ~ Normal(eft_priors["alpha0p_BGS"][1], eft_priors["alpha0p_BGS"][2])
            alpha2p_BGS ~ Normal(eft_priors["alpha2p_BGS"][1], eft_priors["alpha2p_BGS"][2])
            alpha4p_BGS = 0
            st0p_BGS ~ Normal(eft_priors["st0p_BGS"][1], eft_priors["st0p_BGS"][2])
            st2p_BGS ~ Normal(eft_priors["st2p_BGS"][1], eft_priors["st2p_BGS"][2])
            st4p_BGS = 0
            eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
        elseif tracer == "LRG1"
            b1p_LRG1 ~ Uniform(eft_priors["b1p_LRG1"][1], eft_priors["b1p_LRG1"][2])
            b2p_LRG1 ~ Normal(eft_priors["b2p_LRG1"][1], eft_priors["b2p_LRG1"][2])
            b3p_LRG1 = 0
            bsp_LRG1 ~ Normal(eft_priors["bsp_LRG1"][1], eft_priors["bsp_LRG1"][2])
            alpha0p_LRG1 ~ Normal(eft_priors["alpha0p_LRG1"][1], eft_priors["alpha0p_LRG1"][2])
            alpha2p_LRG1 ~ Normal(eft_priors["alpha2p_LRG1"][1], eft_priors["alpha2p_LRG1"][2])
            alpha4p_LRG1 = 0
            st0p_LRG1 ~ Normal(eft_priors["st0p_LRG1"][1], eft_priors["st0p_LRG1"][2])
            st2p_LRG1 ~ Normal(eft_priors["st2p_LRG1"][1], eft_priors["st2p_LRG1"][2])
            st4p_LRG1 = 0
            eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
        elseif tracer == "LRG2"
            b1p_LRG2 ~ Uniform(eft_priors["b1p_LRG2"][1], eft_priors["b1p_LRG2"][2])
            b2p_LRG2 ~ Normal(eft_priors["b2p_LRG2"][1], eft_priors["b2p_LRG2"][2])
            b3p_LRG2 = 0
            bsp_LRG2 ~ Normal(eft_priors["bsp_LRG2"][1], eft_priors["bsp_LRG2"][2])
            alpha0p_LRG2 ~ Normal(eft_priors["alpha0p_LRG2"][1], eft_priors["alpha0p_LRG2"][2])
            alpha2p_LRG2 ~ Normal(eft_priors["alpha2p_LRG2"][1], eft_priors["alpha2p_LRG2"][2])
            alpha4p_LRG2 = 0
            st0p_LRG2 ~ Normal(eft_priors["st0p_LRG2"][1], eft_priors["st0p_LRG2"][2])
            st2p_LRG2 ~ Normal(eft_priors["st2p_LRG2"][1], eft_priors["st2p_LRG2"][2])
            st4p_LRG2 = 0
            eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
        elseif tracer == "LRG3"
            b1p_LRG3 ~ Uniform(eft_priors["b1p_LRG3"][1], eft_priors["b1p_LRG3"][2])
            b2p_LRG3 ~ Normal(eft_priors["b2p_LRG3"][1], eft_priors["b2p_LRG3"][2])
            b3p_LRG3 = 0
            bsp_LRG3 ~ Normal(eft_priors["bsp_LRG3"][1], eft_priors["bsp_LRG3"][2])
            alpha0p_LRG3 ~ Normal(eft_priors["alpha0p_LRG3"][1], eft_priors["alpha0p_LRG3"][2])
            alpha2p_LRG3 ~ Normal(eft_priors["alpha2p_LRG3"][1], eft_priors["alpha2p_LRG3"][2])
            alpha4p_LRG3 = 0
            st0p_LRG3 ~ Normal(eft_priors["st0p_LRG3"][1], eft_priors["st0p_LRG3"][2])
            st2p_LRG3 ~ Normal(eft_priors["st2p_LRG3"][1], eft_priors["st2p_LRG3"][2])
            st4p_LRG3 = 0
            eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
        elseif tracer == "ELG2"
            b1p_ELG2 ~ Uniform(eft_priors["b1p_ELG2"][1], eft_priors["b1p_ELG2"][2])
            b2p_ELG2 ~ Normal(eft_priors["b2p_ELG2"][1], eft_priors["b2p_ELG2"][2])
            b3p_ELG2 = 0
            bsp_ELG2 ~ Normal(eft_priors["bsp_ELG2"][1], eft_priors["bsp_ELG2"][2])
            alpha0p_ELG2 ~ Normal(eft_priors["alpha0p_ELG2"][1], eft_priors["alpha0p_ELG2"][2])
            alpha2p_ELG2 ~ Normal(eft_priors["alpha2p_ELG2"][1], eft_priors["alpha2p_ELG2"][2])
            alpha4p_ELG2 = 0
            st0p_ELG2 ~ Normal(eft_priors["st0p_ELG2"][1], eft_priors["st0p_ELG2"][2])
            st2p_ELG2 ~ Normal(eft_priors["st2p_ELG2"][1], eft_priors["st2p_ELG2"][2])
            st4p_ELG2 = 0
            eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
        elseif tracer == "QSO"
            b1p_QSO ~ Uniform(eft_priors["b1p_QSO"][1], eft_priors["b1p_QSO"][2])
            b2p_QSO ~ Normal(eft_priors["b2p_QSO"][1], eft_priors["b2p_QSO"][2])
            b3p_QSO = 0
            bsp_QSO ~ Normal(eft_priors["bsp_QSO"][1], eft_priors["bsp_QSO"][2])
            alpha0p_QSO ~ Normal(eft_priors["alpha0p_QSO"][1], eft_priors["alpha0p_QSO"][2])
            alpha2p_QSO ~ Normal(eft_priors["alpha2p_QSO"][1], eft_priors["alpha2p_QSO"][2])
            alpha4p_QSO = 0
            st0p_QSO ~ Normal(eft_priors["st0p_QSO"][1], eft_priors["st0p_QSO"][2])
            st2p_QSO ~ Normal(eft_priors["st2p_QSO"][1], eft_priors["st2p_QSO"][2])
            st4p_QSO = 0
            eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]            
        end
        b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = eft_params_physical
        # Converts physical to Eulerian basis
        f, sigma8 = f_dict[tracer], sigma8_dict[tracer]
        b1l = b1p/sigma8-1; b2l = b2p/sigma8^2; b3l = b3p/sigma8^3; bsl = bsp/sigma8^2
        b1e = b1l+1; b2e = 8/21*b1l+b2l; bse = bsl-2/7*b1l; b3e = 3*b3l+b1l
        alpha0e = (1+b1l)^2*alpha0p; alpha2e = f*(1+b1l)*(alpha0p+alpha2p); alpha4e = f*(f*alpha2p+(1+b1l)*alpha4p); alpha6e = f^2*alpha4p
        st0e = st0p/(nd_dict[tracer]); st2e = st2p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^2; st4e = st4p/(nd_dict[tracer])*(fsat_dict[tracer])*(sigv_dict[tracer])^4
        eft_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
        # Combines cosmological and EFT parameters into one theory vector
        cosmo_eft_params = vcat(cosmo_params_FS_BAO, eft_params)
        # Calculates FS/BAO theory vector given parameters
        prediction_FS_BAO = iΓ_FS_BAO_dict[tracer]*vcat(wmat_dict[tracer]*theory_FS(cosmo_eft_params, FS_emus[tracer], kin_dict[tracer]),
                                                        theory_BAO(cosmo_params_FS_BAO, BAO_emu, redshift_eff[tracer], tracer))
        D_FS_BAO_dict[tracer] ~ MvNormal(prediction_FS_BAO, I)
    end
    # Adds Lya BAO as a stand-alone (uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
    # Adds SN contribution
    prediction_SN = iΓ_SN * theory_SN(cosmo_params_FS_BAO, Mb, z_SN, SN_type)
    D_SN ~ MvNormal(prediction_SN, I)
end

@model function model_BAO_CMB_bay(D_BAO_dict, D_Lya, D_CMB)
    """Likelihood for BAO and CMB joint."""
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
    for tracer in tracer_vector
        prediction_BAO = iΓ_BAO_dict[tracer] * theory_BAO(cosmo_params_FS_BAO, BAO_emu, redshift_eff[tracer], tracer)
        D_BAO_dict[tracer] ~ MvNormal(prediction_BAO, I)
    end
    # Adds Lya BAO as a stand-alone (uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
end

@model function model_BAO_CMB_SN_bay(D_BAO_dict, D_Lya, D_CMB, iΓ_SN, D_SN, z_SN, SN_type)
    """Likelihood for BAO, CMB and SN joint."""
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
    for tracer in tracer_vector
        prediction_BAO = iΓ_BAO_dict[tracer] * theory_BAO(cosmo_params_FS_BAO, BAO_emu, redshift_eff[tracer], tracer)
        D_BAO_dict[tracer] ~ MvNormal(prediction_BAO, I)
    end
    # Adds Lya BAO as a stand-alone (uncorrelated with other tracers)
    prediction_Lya = iΓ_Lya * theory_BAO(cosmo_params_FS_BAO, BAO_emu, 2.33, "Lya")
    D_Lya ~ MvNormal(prediction_Lya, I)
    # Adds CMB contribution
    prediction_CMB = iΓ_CMB * theory_CMB(cosmo_params_CMB, CMB_emus) ./ (yₚ^2)
    D_CMB ~ MvNormal(prediction_CMB, I)
    # Adds SN contribution
    prediction_SN = iΓ_SN * theory_SN(cosmo_params_FS_BAO, Mb, z_SN, SN_type)
    D_SN ~ MvNormal(prediction_SN, I)
end

