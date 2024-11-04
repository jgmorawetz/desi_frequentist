
using Distributed
using Statistics
using Plots
using Random
using BenchmarkTools
using LaTeXStrings
using Distributions
using SimpleChains
using Static
using NPZ
using StatsPlots
using ForwardDiff
using LinearAlgebra
using ProgressMeter
using Turing
using Pathfinder
using Optim
using AdvancedHMC
using DelimitedFiles
using Printf
using Effort


########################### Loads the emulator ###############################

function load_component(component, ℓ, folder, k_grid, sky)
    if component == "11"
        k_number = (nk-1)*3
    elseif component == "loop"
        k_number = (nk-1)*12
    elseif component == "ct"
        k_number = (nk-1)*6
    else
        error("You didn't choose a viable component!")
    end
    mlpd = SimpleChain(
      static(8),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(tanh, 64),
      TurboDense(identity, k_number)
    )
    weights = npzread(folder*"weights_P_"*component*"_lcdm_l_"*string(ℓ)*"_sky_"*string(sky)*".npy")
    outMinMax = npzread(folder*"outMinMax_P_"*component*"_lcdm_l_"*string(ℓ)*"_sky_"*string(sky)*".npy")
    inMinMax = npzread(folder*"inMinMax_lcdm_l_"*string(ℓ)*"_sky_"*string(sky)*".npy")
    sc_emu = Effort.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)
    if component == "11"
        comp_emu = Effort.P11Emulator(TrainedEmulator = sc_emu, kgrid = k_grid,
               InMinMax=inMinMax, OutMinMax = outMinMax)
    elseif component == "loop"
        comp_emu = Effort.PloopEmulator(TrainedEmulator = sc_emu, kgrid = k_grid,
               InMinMax=inMinMax, OutMinMax = outMinMax)
    elseif component == "ct"
        comp_emu = Effort.PctEmulator(TrainedEmulator = sc_emu, kgrid = k_grid,
               InMinMax=inMinMax, OutMinMax = outMinMax)
    else
        error("You didn't choose a viable component!")
    end
    return comp_emu
end

function load_multipole(ℓ, folder, k_grid, sky)
    P11 = load_component("11", ℓ, folder, k_grid, sky)
    Ploop = load_component("loop", ℓ, folder, k_grid, sky)
    Pct = load_component("ct", ℓ, folder, k_grid, sky)
    emulator = Effort.PℓEmulator(P11 = P11, Ploop = Ploop, Pct = Pct)
    return emulator
end

emudir="effortEmu_w0wamnu/";
nk = 49;
x = Array(LinRange(0.02,0.5,nk));
k_grid = zeros(nk-1);
for i in 1:(nk-1)
    k_grid[i] = (x[i]+x[i+1])/2
end
z_idx = 1
if z_idx == 0
    @info "You choose z = 0.5!"
elseif z_idx == 1
    @info "You choose z = 0.8!"
elseif z_idx == 2
    @info "You choose z = 1.1!"
elseif z_idx == 3
    @info "You choose z = 1.4!"
else
    @error "You didn't select a viable redshift!"
end
Mono_Emu = load_multipole(0, emudir, k_grid, z_idx)
Quad_Emu = load_multipole(2, emudir, k_grid, z_idx);
cosmo=readdlm("data/abacus_cosmo.txt", ' ');


############################## COMPUTING THEORY VECTOR AS FUNCTION OF PARAMETERS ##################################

function theory(θ, n, z, Mono_Emu, Quad_Emu, n_bar)
    # θ[1:6] cosmoparams, ln_10_As, ns, h, ωb, ωc, Mν
    # θ[7:13] bias
    # θ[14:15] stoch
    h = θ[3]
    ωb = θ[4]
    ωc = θ[5]
    Mν = θ[6]
    w0 = θ[7]
    wa = θ[8]
    Ωc = ωc/h/h
    Ωb = ωb/h/h
    f = Effort._f_z(z, Ωc, Ωb, h, Mν, w0, wa);
    my_θ = deepcopy(θ)
    my_θ[13] /= (0.7^2)
    my_θ[14] /= (0.35^2)
    my_θ[15] /= (0.35^2)
    k_grid = Mono_Emu.P11.kgrid
    stoch_0, stoch_2 = Effort.get_stoch_terms(θ[16], θ[17], θ[18], n_bar, k_grid)
    return vcat((Effort.get_Pℓ(my_θ[1:8], vcat(my_θ[9:15]), f, Mono_Emu) .+ stoch_0)[1:n],
                (Effort.get_Pℓ(my_θ[1:8], vcat(my_θ[9:15]), f, Quad_Emu) .+ stoch_2)[1:n])
end;


################################## SETS UP A MOCK OBSERVATION DATA VECTOR #################################

ln10As = cosmo[1,4]
ns     = cosmo[1,5]
h      = cosmo[1,3]
ωb     = cosmo[1,1]
ωc     = cosmo[1,2]
w0     = cosmo[1,6]
wa     = cosmo[1,7]
Mν     = 0.06
b1     = 2.1
b2     = 1.8
b3     = -4.0        
b4     = -0.02
cct    = -2.4
cr1    = -1.4
cr2    = 0
cϵ0    = 0.3
cϵ1    = 0
cϵ2    = -4.8;
n = 18
n_bar = 5e-4
z = 0.8
θ = [ln10As, ns, h, ωb, ωc, Mν, w0, wa, b1, b2, b3, b4, cct, cr1, cr2, cϵ0, cϵ1, cϵ2]
cov_20=readdlm("data/CovaPTcov_gaussian_AbacusFid_z0.8_n5e-4_Mono_Quad_Hexa.dat", ' ',Float64)
indices = [3:20;33:50]
cov_20 = cov_20[indices,indices];
cosmoidx = 1;
datavec4test = theory(θ, n, z, Mono_Emu, Quad_Emu, n_bar);


#################################### MODEL FOR THE DATA ########################################

Gamma = sqrt(cov_20)
iGamma = inv(Gamma)
D = iGamma * datavec4test
@model function model_fixed(D, iGamma, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, fixed_value, param_idx, mle_or_map)
    # uses a BBN prior since omega baryon is poorly constrained otherwise (mu=0.02237, sigma=0.0005)
    if mle_or_map == "MLE"
        ln10As ~ Uniform(2.5, 3.3)
        ns ~ Uniform(0.7, 1.1)
        h ~ Uniform(0.6, 0.8)
        ωb ~ Normal(0.02237, 0.0005) # BBN prior
        ωc ~ Uniform(0.085, 0.2) 
        w0 = cosmo[cosmoidx,6] # only LCDM parameters vary
        wa = cosmo[cosmoidx,7]
        Mν = 0.06
        b1 ~ Uniform(0., 12.)
        b2 ~ Uniform(-24., 24.)
        b3 ~ Uniform(-90., 90.)        
        b4 ~ Uniform(-24., 24.)
        cct ~ Uniform(-36., 36.)
        cr1 ~ Uniform(-72., 72.)
        cr2 = 0 # setting cr2 and ce1 to 0 since they are degenerate with others
        cϵ0 ~ Uniform(-24., 24.)
        cϵ1 = 0
        cϵ2 ~ Uniform(-72., 72.)
    elseif mle_or_map == "MAP"
        ln10As ~ Uniform(2.5, 3.3)
        ns ~ Uniform(0.7, 1.1)
        h ~ Uniform(0.6, 0.8) 
        ωb ~ Normal(0.02237, 0.0005) # BBN prior
        ωc ~ Uniform(0.085, 0.2)
        w0 = cosmo[cosmoidx,6] 
        wa = cosmo[cosmoidx,7]
        Mν = 0.06
        b1 ~ Uniform(0., 4.) 
        b2 ~ Uniform(-4., 4.)
        b3 ~ Normal(0., 10.)        
        b4 ~ Normal(0., 2.)
        cct ~ Normal(0., 4.)
        cr1 ~ Normal(0., 8.)
        cr2 = 0
        cϵ0 ~ Normal(0., 2.)
        cϵ1 = 0
        cϵ2 ~ Normal(0., 4.)
    end
    param_list = [b1, b2, b3, b4, cct, cr1, cr2, cϵ0, cϵ1, cϵ2]
    θ = [ln10As, ns, h, ωb, ωc, Mν, w0, wa, param_list...]
    θ[param_idx] = fixed_value
    Prediction = iGamma * theory(θ, n, z, Mono_Emu, Quad_Emu, n_bar)
    D ~ MvNormal(Prediction, I)
    return nothing
end


function run_optimize(N, model, mle_or_map)
    max_lp = -Inf
    best_fit = nothing
    if mle_or_map == "MLE"
        for i in 1:N
            # Your model fitting here, replace `model_20` and `MAP()` with your actual model and prior
            current_fit = optimize(model, MAP(), Optim.Options(iterations=10000, allow_f_increases=true))
            # Check if the current fit's lp is the largest we've seen
            if current_fit.lp > max_lp
                max_lp = current_fit.lp
                best_fit = current_fit
            end
        end
        return best_fit
    elseif mle_or_map == "MAP"
        for i in 1:N
            # Your model fitting here, replace `model_20` and `MAP()` with your actual model and prior
            current_fit = optimize(model, MAP(), Optim.Options(iterations=10000, allow_f_increases=true))
            # Check if the current fit's lp is the largest we've seen
            if current_fit.lp > max_lp
                max_lp = current_fit.lp
                best_fit = current_fit
            end
        end
        return best_fit
    end
end;


function compute_profile_likelihood(param_idx, param_values, D, iGamma, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, mle_or_map)
    bins = length(param_values)
    profile_likelihood = Vector{Float64}(undef, bins)
    n_runs = 20
    # Loop over each value to evaluate the profile likelihood
    if mle_or_map == "MLE"
        for i in 1:bins
            @info "Optimization number" i
            fixed_value = param_values[i]
            # Fit the modified model
            fit_result = run_optimize(n_runs, model_fixed(D, iGamma, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, fixed_value, param_idx, "MLE"), "MLE")
            # Store the log-posterior value (chi2) for the current fixed parameter value
            profile_likelihood[i] = fit_result.lp
        end
        return param_values, profile_likelihood
    elseif mle_or_map == "MAP"
        for i in 1:bins
            @info "Optimization number" i
            fixed_value = param_values[i]
            # Fit the modified model
            fit_result = run_optimize(n_runs, model_fixed(D, iGamma, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, fixed_value, param_idx, "MAP"), "MAP")
            # Store the log-posterior value (chi2) for the current fixed parameter value
            profile_likelihood[i] = fit_result.lp
        end
        return param_values, profile_likelihood
    end  
end;

# Sets the parameter index to generate profile likelihood for (only one to vary between scripts)
param_idx = 1

if param_idx == 1
    param_values = collect(2.5:0.1:3.3) #######
    name = "ln10As"
elseif param_idx == 2
    param_values = collect(0.7:0.1:1.1) ########
    name = "ns"
elseif param_idx == 3
    param_values = collect(0.6:0.05:0.8) ##########
    name = "h"
elseif param_idx == 4
    param_values = collect(0.02037:0.00025:0.02437) ##########
    name = "omegab"
elseif param_idx == 5
    param_values = collect(0.085:0.025:0.2) ############
    name = "omegac"
elseif param_idx == 9
    param_values = collect(0:0.2:4)############ 
    name = "b1"


@time param_values, profile_likelihood_mle = compute_profile_likelihood(param_idx, param_values, D, iGamma, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, "MLE")
@time param_values, profile_likelihood_map = compute_profile_likelihood(param_idx, param_values, D, iGamma, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, "MAP")

using Plots

delta_chi2_mle = -(profile_likelihood_mle .- maximum(profile_likelihood_mle))
delta_chi2_map = -(profile_likelihood_map .- maximum(profile_likelihood_map))
npzwrite("/home/jgmorawe/FrequentistExample/$(name)_synthetic_BBNprior_paramvalues.npy", param_values)
npzwrite("/home/jgmorawe/FrequentistExample/$(name)_MLE_synthetic_BBNprior_deltachisquared.npy", delta_chi2_mle)
npzwrite("/home/jgmorawe/FrequentistExample/$(name)_MAP_synthetic_BBNprior_deltachisquared.npy", delta_chi2_map)
