using Pkg
Pkg.activate(".")
using Effort
using AbstractCosmologicalEmulators
using Capse
using DataInterpolations


# Initiates the fiducial cosmology (for purposes of applying AP to the power spectrum multipoles)
cosmology_fid = Effort.w0waCDMCosmology(ln10Aₛ=3.044, nₛ=0.9649, h=0.6736, ωb=0.02237, ωc=0.12, mν=0.06, w0=-1.0, wa=0.0, ωk=0.0)


function theory_FS(cosmo_params, eft_params, z, emu_components, k_input)        
    # Constructs the cosmology
    cosmology = Effort.w0waCDMCosmology(ln10Aₛ=cosmo_params[1], nₛ=cosmo_params[2], h=cosmo_params[3], ωb=cosmo_params[4],
                                        ωc=cosmo_params[5], mν=0.06, w0=cosmo_params[6], wa=cosmo_params[7], ωk=0.0)
    # Computes growth factor
    D = Effort.D_f_z(z, cosmology)[1]                                                                              
    # Input for the emulator
    emulator_input = [z, cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h*100, cosmology.ωb, cosmology.ωc, 
                      cosmology.mν, cosmology.w0, cosmology.wa]
    # The emulator components
    mono_emu = emu_components[1]
    quad_emu = emu_components[2]
    hexa_emu = emu_components[3]
    # Extracts the multipoles without AP yet applied
    pk0 = Effort.get_Pℓ(emulator_input, D, eft_params, mono_emu)
    pk2 = Effort.get_Pℓ(emulator_input, D, eft_params, quad_emu)
    pk4 = Effort.get_Pℓ(emulator_input, D, eft_params, hexa_emu)
    # Calculates the multipoles with AP applied
    q_par, q_perp = Effort.q_par_perp(z, cosmology, cosmology_fid)
    pk0_AP, pk2_AP, pk4_AP = Effort.apply_AP(vec(mono_emu.P11.kgrid), k_input, pk0, pk2, pk4, q_par, q_perp, n_GL_points=8)
    # Stacks the multipoles together
    pk_AP = vcat(pk0_AP, pk2_AP, pk4_AP)
    return pk_AP
end


function theory_BAO(cosmo_params, z, emu, tracer)
    # Constructs the cosmology
    cosmology = Effort.w0waCDMCosmology(ln10Aₛ=cosmo_params[1], nₛ=cosmo_params[2], h=cosmo_params[3], ωb=cosmo_params[4],
                                        ωc=cosmo_params[5], mν=0.06, w0=cosmo_params[6], wa=cosmo_params[7], ωk=0.0)
    # Calculates AP distortion contributions
    H_true = cosmology.h * Effort.E_z(z, cosmology) # factor of 100 irrelevant since dividing anyway
    H_fid = cosmology_fid.h * Effort.E_z(z, cosmology_fid)
    DA_true = Effort.r_z(z, cosmology) # comoving angular diameter distance
    DA_fid = Effort.r_z(z, cosmology_fid)
    # Input for the emulator
    emulator_input = [z, cosmology.ln10Aₛ, cosmology.nₛ, cosmology.h*100, cosmology.ωb, cosmology.ωc, 
                      cosmology.mν, cosmology.w0, cosmology.wa]
    emulator_input_fid = [z, cosmology_fid.ln10Aₛ, cosmology_fid.nₛ, cosmology_fid.h*100, cosmology_fid.ωb, 
                          cosmology_fid.ωc, cosmology_fid.mν, cosmology_fid.w0, cosmology_fid.wa]
    # Extracts the rsdrag calculations
    rsdrag_true = AbstractCosmologicalEmulators.run_emulator(emulator_input, nothing, emu)[3]
    rsdrag_fid = AbstractCosmologicalEmulators.run_emulator(emulator_input_fid, nothing, emu)[3]
    # Calculates the compression parameters
    alpha_par = (H_fid * rsdrag_fid) / (H_true * rsdrag_true)
    alpha_perp = (DA_true * rsdrag_fid) / (DA_fid * rsdrag_true)
    alpha_iso = cbrt(alpha_par * alpha_perp^2) # had to change since ^(1/3) causes domain issues for negatives
    if tracer in ["LRG1", "LRG2", "LRG3", "ELG2", "Lya"]
       compressed_params = [alpha_par, alpha_perp]
    elseif tracer in ["BGS", "QSO"]
        compressed_params = [alpha_iso]
    end
    return compressed_params
end


function theory_CMB(cosmo_params, emu_components)
    # Note: CMB cosmology parameters are [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa] unlike others
    TT_emu = emu_components[1]
    TE_emu = emu_components[2]
    EE_emu = emu_components[3]
    lsTT = 2:2508
    lsTE = 2:1996
    facTT = lsTT .* (lsTT .+ 1) ./ (2*π)
    facTE = lsTE .* (lsTE .+ 1) ./ (2*π)
    angular_power = PlanckLite.bin_Cℓ(Capse.get_Cℓ(cosmo_params, TT_emu)[1:2507] ./ facTT,
                                      Capse.get_Cℓ(cosmo_params, TE_emu)[1:1995] ./ facTE,
                                      Capse.get_Cℓ(cosmo_params, EE_emu)[1:1995] ./ facTE)
    return angular_power
end


function theory_SN(cosmo_params, M, z_SN, SN_type)
    # Constructs the cosmology
    cosmology = Effort.w0waCDMCosmology(ln10Aₛ=cosmo_params[1], nₛ=cosmo_params[2], h=cosmo_params[3], ωb=cosmo_params[4],
                                        ωc=cosmo_params[5], mν=0.06, w0=cosmo_params[6], wa=cosmo_params[7], ωk=0.0)
    # Uses interpolation since there are a large number of redshifts
    z_interp = Array(LinRange(0, maximum(z_SN)+0.05, 50))
    DL_interp = Effort.r_z(z_interp, cosmology) .* (1 .+ z_interp)
    # Applies the interpolation function to the supernovae redshifts specifically
    DL_SN = DataInterpolations.QuadraticSpline(DL_interp, z_interp).(z_SN)
    # Converts to distance modulus (slightly different depending on dataset)
    if SN_type == "DESY5"
        dist_mod = 5 .* log10.(DL_SN) .+ 25 .+ M
    elseif SN_type == "PantheonPlus"
        dist_mod = 5 .* log10.(DL_SN) .+ 25 .+ M
    elseif SN_type == "Union3"
        dist_mod = 5 .* log10.(100 .* DL_SN .* cosmo_params[3]) .+ 25 .+ M
    end
    return dist_mod
end