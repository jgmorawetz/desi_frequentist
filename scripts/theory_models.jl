using Pkg
Pkg.activate(".")
using Effort
using Capse
using PlanckLite
using DataInterpolations


function theory_FS(theta_FS, emu_FS_components, kin)
    """Constructs theory vector for full-shape power spectrum multipoles."""
    cosmo_params = theta_FS[1:7] # [ln10As, ns, H0, ωb, ωc, w0, wa] cosmological parameter vector
    eft_params = theta_FS[8:18] # [b1, b2, b3, bs, alpha0, alpha2, alpha4, alpha6, st0, st2, st4] EFT nuisance parameter vector
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

function theory_BAO(theta_BAO, emu_BAO, zeff, tracer)
    """Constructs theory vector for post-reconstruction BAO compressed parameters."""
    # theta_BAO is cosmological parameter vector [ln10As, ns, H0, ωb, ωc, w0, wa]
    theta_BAO_fid = [3.044, 0.9649, 67.36, 0.02237, 0.1200, -1, 0] # fiducial planck 2018 cosmology
    h_fid = theta_BAO_fid[3]/100; Ωcb_fid = (theta_BAO_fid[4]+theta_BAO_fid[5])/h_fid^2; w0_fid=theta_BAO_fid[6]; wa_fid=theta_BAO_fid[7]
    h_true = theta_BAO[3]/100; Ωcb_true = (theta_BAO[4]+theta_BAO[5])/h_true^2; w0_true=theta_BAO[6]; wa_true=theta_BAO[7]
    mν_fixed = 0.06
    # Computes H(z) and D_A(z) for fid and model cosmologies
    H_fid = h_fid * Effort._E_z(zeff, Ωcb_fid, h_fid; mν=mν_fixed, w0=w0_fid, wa=wa_fid)
    H_true = h_true * Effort._E_z(zeff, Ωcb_true, h_true; mν=mν_fixed, w0=w0_true, wa=wa_true)
    DA_fid = Effort._r_z(zeff, Ωcb_fid, h_fid; mν=mν_fixed, w0=w0_fid, wa=wa_fid)
    DA_true = Effort._r_z(zeff, Ωcb_true, h_true; mν=mν_fixed, w0=w0_true, wa=wa_true)
    # Computes rs_drag from emulator
    rsdrag_fid = Effort.get_BAO(theta_BAO_fid, emu_BAO)[1] # rs_drag is first entry
    rsdrag_true = Effort.get_BAO(theta_BAO, emu_BAO)[1]
    # Converts to alpha par and perp (or iso) components
    alpha_par = (H_fid * rsdrag_fid) / (H_true * rsdrag_true)
    alpha_perp = (DA_true * rsdrag_fid) / (DA_fid * rsdrag_true)
    alpha_iso = (alpha_par * alpha_perp^2)^(1/3)
    # Returns either [alpha_par, alpha_perp] or [alpha_iso] depending on the tracer
    if tracer in ["LRG1", "LRG2", "LRG3", "ELG2", "Lya"]
        return [alpha_par, alpha_perp]
    elseif tracer in ["BGS", "QSO"]
        return [alpha_iso]
    end
end

function theory_CMB(theta_CMB, emu_CMB_components)
    """Constructs theory vector for CMB TT, TE, EE angular power spectra."""
    # theta_CMB is cosmological parameter vector [ln10As, ns, H0, ωb, ωc, τ, mν, w0, wa]
    emu_TT = emu_CMB_components[1]
    emu_TE = emu_CMB_components[2]
    emu_EE = emu_CMB_components[3]
    lsTT = 2:2508
    lsTE = 2:1996
    facTT = lsTT .* (lsTT .+ 1) ./ (2*π)
    facTE = lsTE .* (lsTE .+ 1) ./ (2*π)
    return PlanckLite.bin_Cℓ(Capse.get_Cℓ(theta_CMB, emu_TT)[1:2507] ./ facTT,
                             Capse.get_Cℓ(theta_CMB, emu_TE)[1:1995] ./ facTE,
                             Capse.get_Cℓ(theta_CMB, emu_EE)[1:1995] ./ facTE)
end

function theory_SN(theta_SN, Mb, z_SN, SN_type)
    """Constructs theory vector for supernovae (from either DESY5, PantheonPlus or Union3)."""
    # theta_SN is cosmological parameter vector [ln10As, ns, H0, ωb, ωc, w0, wa]
    h = theta_SN[3]/100; Ωcb = (theta_SN[4]+theta_SN[5])/h^2; w0 = theta_SN[6]; wa = theta_SN[7]
    mν_fixed = 0.06
    z_interp = Array(LinRange(0, 2.5, 50)) # uses interpolation to not have to calculate for all supernovae redshifts
    DL_interp = Effort._r_z.(z_interp, Ωcb, h; mν=mν_fixed, w0=w0, wa=wa)
    DL_SN = DataInterpolations.QuadraticSpline(DL_interp, z_interp).(z_SN) .* (1 .+ z_SN)
    if SN_type == "DESY5SN"
        return 5 .* log10.(DL_SN) .+ 25 .+ Mb
    elseif SN_type == "PantheonPlusSN"
        return 5 .* log10.(DL_SN) .+ 25 .+ Mb
    elseif SN_type == "Union3SN"
        return 5 .* log10.(100 .* DL_SN .* h) .+ 25 .+ Mb
    end
end