using Pkg
Pkg.activate(".")
include("priors.jl")
include("dr1_datasets.jl")
include("theory_vectors.jl")
include("emulators.jl")
using LinearAlgebra
using Turing


function physical_to_eulerian_basis(physical_params, f, sigma8, nd, fsat, sigv)
    # Converts physical to Eulerian basis
    b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p = physical_params
    b1l = b1p / sigma8 - 1
    b2l = b2p / sigma8 ^ 2
    b3l = b3p / sigma8 ^ 3
    bsl = bsp / sigma8 ^ 2
    b1e = b1l + 1
    b2e = 8/21 * b1l + b2l
    b3e = 3 * b3l + b1l
    bse = bsl - 2/7 * b1l
    alpha0e = (1 + b1l) ^ 2 * alpha0p 
    alpha2e = f * (1 + b1l) * (alpha0p + alpha2p)
    alpha4e = f * (f * alpha2p + (1 + b1l) * alpha4p)
    alpha6e = f ^ 2 * alpha4p
    st0e = st0p / nd
    st2e = st2p / nd * fsat * sigv^2
    st4e = st4p / nd * fsat * sigv^4
    eulerian_params = [b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e]
    return eulerian_params
end


function prediction_FS(cosmo_params, eft_params_physical, tracer)
    # Assembles cosmological parameters to apply to BAO emulator (to get f and sigma8)
    ln10Aₛ, nₛ, h, ωb, ωc, w0, wa = cosmo_params
    cosmo_params_BAO_emu = vcat([redshift_eff[tracer]], [ln10Aₛ, nₛ, h*100, ωb, ωc, 0.06, w0, wa])
    BAO_result = AbstractCosmologicalEmulators.run_emulator(cosmo_params_BAO_emu, nothing, BAO_ln10As_emu)
    f_tracer, sigma8_tracer = BAO_result[7], BAO_result[2]
    # Converts physical (b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p) to
    # Eulerian (b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e) basis
    eft_params_eulerian = physical_to_eulerian_basis(eft_params_physical, f_tracer, sigma8_tracer, 
                                                     nd_dict[tracer], fsat_dict[tracer], sigv_dict[tracer])
    # Calculates theory vector and applies window matrix and whitening transformation
    prediction = iΓ_FS_dict[tracer] * (wmat_dict[tracer] * theory_FS(
                 cosmo_params, eft_params_eulerian, redshift_eff[tracer], FS_emus, kin_dict[tracer]))   
    return prediction
end


function prediction_FS_BAO(cosmo_params, eft_params_physical, tracer)
    # Assembles cosmological parameters to apply to BAO emulator (to get f and sigma8)
    ln10Aₛ, nₛ, h, ωb, ωc, w0, wa = cosmo_params
    cosmo_params_BAO_emu = vcat([redshift_eff[tracer]], [ln10Aₛ, nₛ, h*100, ωb, ωc, 0.06, w0, wa])
    BAO_result = AbstractCosmologicalEmulators.run_emulator(cosmo_params_BAO_emu, nothing, BAO_ln10As_emu)
    f_tracer, sigma8_tracer = BAO_result[7], BAO_result[2]
    # Converts physical (b1p, b2p, b3p, bsp, alpha0p, alpha2p, alpha4p, st0p, st2p, st4p) to
    # Eulerian (b1e, b2e, b3e, bse, alpha0e, alpha2e, alpha4e, alpha6e, st0e, st2e, st4e) basis
    eft_params_eulerian = physical_to_eulerian_basis(eft_params_physical, f_tracer, sigma8_tracer, 
                                                     nd_dict[tracer], fsat_dict[tracer], sigv_dict[tracer])
    # Calculates theory vectors for full shape and BAO and concatenates together then applies whitening
    prediction1 = wmat_dict[tracer] * theory_FS(cosmo_params, eft_params_eulerian, redshift_eff[tracer], FS_emus, kin_dict[tracer])
    prediction2 = theory_BAO(cosmo_params, redshift_eff[tracer], BAO_ln10As_emu, tracer)
    prediction = iΓ_FS_BAO_dict[tracer] * vcat(prediction1, prediction2)
    return prediction
end


function prediction_BAO(cosmo_params, tracer)
    prediction = iΓ_BAO_dict[tracer] * theory_BAO(cosmo_params, redshift_eff[tracer], BAO_ln10As_emu, tracer)
    return prediction
end


function prediction_CMB(cosmo_params, τ, yₚ)
    # Assembles the cosmological parameters to apply to CMB emulator (different from others)
    ln10Aₛ, nₛ, h, ωb, ωc, w0, wa = cosmo_params
    cosmo_params_CMB_emu = [ln10Aₛ, nₛ, 100*h, ωb, ωc, τ, 0.06, w0, wa]
    prediction = iΓ_CMB * theory_CMB(cosmo_params_CMB_emu, CMB_emus) ./ (yₚ^2)
    return prediction
end


function prediction_SN(cosmo_params, M, iΓ_SN, z_SN, SN_type)
    prediction = iΓ_SN * theory_SN(cosmo_params, M, z_SN, SN_type)
    return prediction
end
    

@model function model_FS(D_FS_dict, tracer_vector, freq_or_bay)
    # Samples cosmological parameters
    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
    nₛ ~ Truncated(Normal(cosmo_priors["nₛ"][1], cosmo_priors["nₛ"][2]), cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2])
    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
    # Samples EFT nuisance parameters for each tracer
    # (b3p, alpha4p, st4p are fixed to zero)
    for tracer in tracer_vector
        if freq_or_bay == "freq"
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
        elseif freq_or_bay == "bay"
            if tracer == "BGS"
                b1p_BGS ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_BGS ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_BGS = 0
                bsp_BGS ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_BGS ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_BGS ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_BGS = 0
                st0p_BGS ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_BGS ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_BGS = 0
                eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
            elseif tracer == "LRG1"
                b1p_LRG1 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG1 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG1 = 0
                bsp_LRG1 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG1 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG1 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG1 = 0
                st0p_LRG1 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG1 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG1 = 0
                eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
            elseif tracer == "LRG2"
                b1p_LRG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG2 = 0
                bsp_LRG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG2 = 0
                st0p_LRG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG2 = 0
                eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
            elseif tracer == "LRG3"
                b1p_LRG3 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG3 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG3 = 0
                bsp_LRG3 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG3 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG3 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG3 = 0
                st0p_LRG3 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG3 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG3 = 0
                eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
            elseif tracer == "ELG2"
                b1p_ELG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_ELG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_ELG2 = 0
                bsp_ELG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_ELG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_ELG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_ELG2 = 0
                st0p_ELG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_ELG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_ELG2 = 0
                eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
            elseif tracer == "QSO"
                b1p_QSO ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_QSO ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_QSO = 0
                bsp_QSO ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_QSO ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_QSO ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_QSO = 0
                st0p_QSO ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_QSO ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_QSO = 0
                eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]
            end
        end
        D_FS_dict[tracer] ~ MvNormal(prediction_FS(cosmo_params, eft_params_physical, tracer), I)
    end
end


@model function model_FS_BAO(D_FS_BAO_dict, D_Lya, tracer_vector, freq_or_bay)
    # Samples cosmological parameters
    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
    nₛ ~ Truncated(Normal(cosmo_priors["nₛ"][1], cosmo_priors["nₛ"][2]), cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2])
    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
    # Samples EFT nuisance parameters for each tracer
    # (b3p, alpha4p, st4p are fixed to zero)
    for tracer in tracer_vector
        if freq_or_bay == "freq"
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
        elseif freq_or_bay == "bay"
            if tracer == "BGS"
                b1p_BGS ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_BGS ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_BGS = 0
                bsp_BGS ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_BGS ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_BGS ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_BGS = 0
                st0p_BGS ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_BGS ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_BGS = 0
                eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
            elseif tracer == "LRG1"
                b1p_LRG1 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG1 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG1 = 0
                bsp_LRG1 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG1 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG1 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG1 = 0
                st0p_LRG1 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG1 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG1 = 0
                eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
            elseif tracer == "LRG2"
                b1p_LRG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG2 = 0
                bsp_LRG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG2 = 0
                st0p_LRG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG2 = 0
                eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
            elseif tracer == "LRG3"
                b1p_LRG3 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG3 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG3 = 0
                bsp_LRG3 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG3 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG3 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG3 = 0
                st0p_LRG3 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG3 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG3 = 0
                eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
            elseif tracer == "ELG2"
                b1p_ELG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_ELG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_ELG2 = 0
                bsp_ELG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_ELG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_ELG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_ELG2 = 0
                st0p_ELG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_ELG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_ELG2 = 0
                eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
            elseif tracer == "QSO"
                b1p_QSO ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_QSO ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_QSO = 0
                bsp_QSO ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_QSO ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_QSO ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_QSO = 0
                st0p_QSO ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_QSO ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_QSO = 0
                eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]
            end
        end
        D_FS_BAO_dict[tracer] ~ MvNormal(prediction_FS_BAO(cosmo_params, eft_params_physical, tracer), I)
    end
    D_Lya ~ MvNormal(iΓ_Lya * theory_BAO(cosmo_params, 2.33, BAO_ln10As_emu, "Lya"), I) # standalone BAO Lya
end


@model function model_FS_BAO_CMB(D_FS_BAO_dict, D_Lya, D_CMB, tracer_vector, freq_or_bay)
    # Samples cosmological parameters
    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
    nₛ ~ Uniform(cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2]) # no ns/omegab priors for CMB
    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
    ωb ~ Uniform(cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
    τ ~ Truncated(Normal(cosmo_priors["τ"][1], cosmo_priors["τ"][2]), cosmo_ranges["τ"][1], cosmo_ranges["τ"][2])
    yₚ ~ Truncated(Normal(cosmo_priors["yₚ"][1], cosmo_priors["yₚ"][2]), cosmo_ranges["yₚ"][1], cosmo_ranges["yₚ"][2])
    # Samples EFT nuisance parameters for each tracer
    # (b3p, alpha4p, st4p are fixed to zero)
    for tracer in tracer_vector
        if freq_or_bay == "freq"
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
        elseif freq_or_bay == "bay"
            if tracer == "BGS"
                b1p_BGS ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_BGS ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_BGS = 0
                bsp_BGS ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_BGS ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_BGS ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_BGS = 0
                st0p_BGS ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_BGS ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_BGS = 0
                eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
            elseif tracer == "LRG1"
                b1p_LRG1 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG1 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG1 = 0
                bsp_LRG1 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG1 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG1 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG1 = 0
                st0p_LRG1 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG1 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG1 = 0
                eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
            elseif tracer == "LRG2"
                b1p_LRG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG2 = 0
                bsp_LRG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG2 = 0
                st0p_LRG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG2 = 0
                eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
            elseif tracer == "LRG3"
                b1p_LRG3 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG3 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG3 = 0
                bsp_LRG3 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG3 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG3 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG3 = 0
                st0p_LRG3 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG3 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG3 = 0
                eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
            elseif tracer == "ELG2"
                b1p_ELG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_ELG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_ELG2 = 0
                bsp_ELG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_ELG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_ELG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_ELG2 = 0
                st0p_ELG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_ELG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_ELG2 = 0
                eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
            elseif tracer == "QSO"
                b1p_QSO ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_QSO ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_QSO = 0
                bsp_QSO ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_QSO ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_QSO ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_QSO = 0
                st0p_QSO ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_QSO ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_QSO = 0
                eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]
            end
        end
        D_FS_BAO_dict[tracer] ~ MvNormal(prediction_FS_BAO(cosmo_params, eft_params_physical, tracer), I)
    end
    D_Lya ~ MvNormal(iΓ_Lya * theory_BAO(cosmo_params, 2.33, BAO_ln10As_emu, "Lya"), I) # standalone BAO Lya
    D_CMB ~ MvNormal(prediction_CMB(cosmo_params, τ, yₚ), I)
end


@model function model_FS_BAO_CMB_SN(D_FS_BAO_dict, D_Lya, D_CMB, D_SN, iΓ_SN, z_SN, SN_type, tracer_vector, freq_or_bay)
    # Samples cosmological parameters
    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
    nₛ ~ Uniform(cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2]) # no ns/omegab priors for CMB
    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
    ωb ~ Uniform(cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
    τ ~ Truncated(Normal(cosmo_priors["τ"][1], cosmo_priors["τ"][2]), cosmo_ranges["τ"][1], cosmo_ranges["τ"][2])
    yₚ ~ Truncated(Normal(cosmo_priors["yₚ"][1], cosmo_priors["yₚ"][2]), cosmo_ranges["yₚ"][1], cosmo_ranges["yₚ"][2])
    if SN_type == "DESY5"
        M ~ Uniform(cosmo_ranges["M_D5"][1], cosmo_ranges["M_D5"][2])
    elseif SN_type == "PantheonPlus"
        M ~ Uniform(cosmo_ranges["M_PP"][1], cosmo_ranges["M_PP"][2])
    elseif SN_type == "Union3"
        M ~ Uniform(cosmo_ranges["M_U3"][1], cosmo_ranges["M_U3"][2])
    end
    # Samples EFT nuisance parameters for each tracer
    # (b3p, alpha4p, st4p are fixed to zero)
    for tracer in tracer_vector
        if freq_or_bay == "freq"
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
        elseif freq_or_bay == "bay"
            if tracer == "BGS"
                b1p_BGS ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_BGS ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_BGS = 0
                bsp_BGS ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_BGS ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_BGS ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_BGS = 0
                st0p_BGS ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_BGS ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_BGS = 0
                eft_params_physical = [b1p_BGS, b2p_BGS, b3p_BGS, bsp_BGS, alpha0p_BGS, alpha2p_BGS, alpha4p_BGS, st0p_BGS, st2p_BGS, st4p_BGS]
            elseif tracer == "LRG1"
                b1p_LRG1 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG1 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG1 = 0
                bsp_LRG1 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG1 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG1 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG1 = 0
                st0p_LRG1 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG1 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG1 = 0
                eft_params_physical = [b1p_LRG1, b2p_LRG1, b3p_LRG1, bsp_LRG1, alpha0p_LRG1, alpha2p_LRG1, alpha4p_LRG1, st0p_LRG1, st2p_LRG1, st4p_LRG1]
            elseif tracer == "LRG2"
                b1p_LRG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG2 = 0
                bsp_LRG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG2 = 0
                st0p_LRG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG2 = 0
                eft_params_physical = [b1p_LRG2, b2p_LRG2, b3p_LRG2, bsp_LRG2, alpha0p_LRG2, alpha2p_LRG2, alpha4p_LRG2, st0p_LRG2, st2p_LRG2, st4p_LRG2]
            elseif tracer == "LRG3"
                b1p_LRG3 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_LRG3 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_LRG3 = 0
                bsp_LRG3 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_LRG3 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_LRG3 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_LRG3 = 0
                st0p_LRG3 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_LRG3 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_LRG3 = 0
                eft_params_physical = [b1p_LRG3, b2p_LRG3, b3p_LRG3, bsp_LRG3, alpha0p_LRG3, alpha2p_LRG3, alpha4p_LRG3, st0p_LRG3, st2p_LRG3, st4p_LRG3]
            elseif tracer == "ELG2"
                b1p_ELG2 ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_ELG2 ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_ELG2 = 0
                bsp_ELG2 ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_ELG2 ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_ELG2 ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_ELG2 = 0
                st0p_ELG2 ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_ELG2 ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_ELG2 = 0
                eft_params_physical = [b1p_ELG2, b2p_ELG2, b3p_ELG2, bsp_ELG2, alpha0p_ELG2, alpha2p_ELG2, alpha4p_ELG2, st0p_ELG2, st2p_ELG2, st4p_ELG2]
            elseif tracer == "QSO"
                b1p_QSO ~ Uniform(eft_priors["b1p"][1], eft_priors["b1p"][2])
                b2p_QSO ~ Normal(eft_priors["b2p"][1], eft_priors["b2p"][2])
                b3p_QSO = 0
                bsp_QSO ~ Normal(eft_priors["bsp"][1], eft_priors["bsp"][2])
                alpha0p_QSO ~ Normal(eft_priors["alpha0p"][1], eft_priors["alpha0p"][2])
                alpha2p_QSO ~ Normal(eft_priors["alpha2p"][1], eft_priors["alpha2p"][2])
                alpha4p_QSO = 0
                st0p_QSO ~ Normal(eft_priors["st0p"][1], eft_priors["st0p"][2])
                st2p_QSO ~ Normal(eft_priors["st2p"][1], eft_priors["st2p"][2])
                st4p_QSO = 0
                eft_params_physical = [b1p_QSO, b2p_QSO, b3p_QSO, bsp_QSO, alpha0p_QSO, alpha2p_QSO, alpha4p_QSO, st0p_QSO, st2p_QSO, st4p_QSO]
            end
        end
        D_FS_BAO_dict[tracer] ~ MvNormal(prediction_FS_BAO(cosmo_params, eft_params_physical, tracer), I)
    end
    D_Lya ~ MvNormal(iΓ_Lya * theory_BAO(cosmo_params, 2.33, BAO_ln10As_emu, "Lya"), I) # standalone BAO Lya
    D_CMB ~ MvNormal(prediction_CMB(cosmo_params, τ, yₚ), I)
    D_SN ~ MvNormal(prediction_SN(cosmo_params, M, iΓ_SN, z_SN, SN_type), I)
end


#@model function model_BAO(D_BAO_dict, D_Lya, tracer_vector)
#    # Samples cosmological parameters
#    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
#    nₛ ~ Truncated(Normal(cosmo_priors["nₛ"][1], cosmo_priors["nₛ"][2]), cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2])
#    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
#    ωb ~ Truncated(Normal(cosmo_priors["ωb"][1], cosmo_priors["ωb"][2]), cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
#    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
#    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
#    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
#    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
#    for tracer in tracer_vector
#        D_BAO_dict[tracer] ~ MvNormal(prediction_BAO(cosmo_params, tracer), I)
#    end
#    D_Lya ~ MvNormal(iΓ_Lya * theory_BAO(cosmo_params, 2.33, BAO_ln10As_emu, "Lya"), I) # standalone BAO Lya
#end


#@model function model_BAO_CMB(D_BAO_dict, D_Lya, D_CMB, tracer_vector)
#    # Samples cosmological parameters
#    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
#    nₛ ~ Uniform(cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2]) # no ns/omegab priors for CMB
#    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
#    ωb ~ Uniform(cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
#    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
#    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
#    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
#    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
#    τ ~ Truncated(Normal(cosmo_priors["τ"][1], cosmo_priors["τ"][2]), cosmo_ranges["τ"][1], cosmo_ranges["τ"][2])
#    yₚ ~ Truncated(Normal(cosmo_priors["yₚ"][1], cosmo_priors["yₚ"][2]), cosmo_ranges["yₚ"][1], cosmo_ranges["yₚ"][2])
#    for tracer in tracer_vector
#        D_BAO_dict[tracer] ~ MvNormal(prediction_BAO(cosmo_params, tracer), I)
#    end
#    D_Lya ~ MvNormal(iΓ_Lya * theory_BAO(cosmo_params, 2.33, BAO_ln10As_emu, "Lya"), I) # standalone BAO Lya
#    D_CMB ~ MvNormal(prediction_CMB(cosmo_params, τ, yₚ), I)
#end


#@model function model_BAO_CMB_SN(D_BAO_dict, D_Lya, D_CMB, D_SN, iΓ_SN, z_SN, SN_type, tracer_vector)
#    # Samples cosmological parameters
#    ln10Aₛ ~ Uniform(cosmo_ranges["ln10Aₛ"][1], cosmo_ranges["ln10Aₛ"][2])
#    nₛ ~ Uniform(cosmo_ranges["nₛ"][1], cosmo_ranges["nₛ"][2]) # no ns/omegab priors for CMB
#    h ~ Uniform(cosmo_ranges["h"][1], cosmo_ranges["h"][2])
#    ωb ~ Uniform(cosmo_ranges["ωb"][1], cosmo_ranges["ωb"][2])
#    ωc ~ Uniform(cosmo_ranges["ωc"][1], cosmo_ranges["ωc"][2])
#    w0 ~ Uniform(cosmo_ranges["w0"][1], cosmo_ranges["w0"][2])
#    wa ~ Uniform(cosmo_ranges["wa"][1], cosmo_ranges["wa"][2])
#    cosmo_params = [ln10Aₛ, nₛ, h, ωb, ωc, w0, wa]
#    τ ~ Truncated(Normal(cosmo_priors["τ"][1], cosmo_priors["τ"][2]), cosmo_ranges["τ"][1], cosmo_ranges["τ"][2])
#    yₚ ~ Truncated(Normal(cosmo_priors["yₚ"][1], cosmo_priors["yₚ"][2]), cosmo_ranges["yₚ"][1], cosmo_ranges["yₚ"][2])
#    if SN_type == "DESY5"
#        M ~ Uniform(cosmo_ranges["M_D5"][1], cosmo_ranges["M_D5"][2])
#    elseif SN_type == "PantheonPlus"
#        M ~ Uniform(cosmo_ranges["M_PP"][1], cosmo_ranges["M_PP"][2])
#    elseif SN_type == "Union3"
#        M ~ Uniform(cosmo_ranges["M_U3"][1], cosmo_ranges["M_U3"][2])
#    end
#    for tracer in tracer_vector
#        D_BAO_dict[tracer] ~ MvNormal(prediction_BAO(cosmo_params, tracer), I)
#    end
#    D_Lya ~ MvNormal(iΓ_Lya * theory_BAO(cosmo_params, 2.33, BAO_ln10As_emu, "Lya"), I) # standalone BAO Lya
#    D_CMB ~ MvNormal(prediction_CMB(cosmo_params, τ, yₚ), I)
#    D_SN ~ MvNormal(prediction_SN(cosmo_params, M, iΓ_SN, z_SN, SN_type), I)
#end