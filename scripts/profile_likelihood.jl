
using Pkg
Pkg.activate("/home/jgmorawe/FrequentistExample")
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

# FUNCTIONS FOR LOADING
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

# CONFIGURATIONS

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
#    Hexa_Emu = load_multipole(4, emudir, k_grid, z_idx);
println("load multipole emulators")################################

cosmo=readdlm("data/abacus_cosmo.txt", ' '); #read some abacus cosmology, 1 for c000, 2~32 for 130~160
println("reads in cosmo data")################################
# BASIC EXAMPLE

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

    f = Effort._f_z(z, Ωc, Ωb, h, Mν, w0, wa); # unsure what f represents here
    my_θ = deepcopy(θ) # unsure what this is doing here
    my_θ[13] /= (0.7^2)
    my_θ[14] /= (0.35^2)
    my_θ[15] /= (0.35^2)
    k_grid = Mono_Emu.P11.kgrid
    stoch_0, stoch_2 = Effort.get_stoch_terms(θ[16], θ[17], θ[18], n_bar, k_grid)
    return vcat((Effort.get_Pℓ(my_θ[1:8], vcat(my_θ[9:15]), f, Mono_Emu) .+ stoch_0)[1:n],
                (Effort.get_Pℓ(my_θ[1:8], vcat(my_θ[9:15]), f, Quad_Emu) .+ stoch_2)[1:n])
#               (Effort.get_Pℓ(my_θ[1:8], vcat(my_θ[9:15]), f, Hexa_Emu))[1:n])
end;

# i'm assuming these are just initial guesses??
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

n = 18 #number of data point, first n in your k_grid # unsure what n represents here??
n_bar = 5e-4
z = 0.8 #redshift, note this is not redshift index
θ = [ln10As, ns, h, ωb, ωc, Mν, w0, wa, b1, b2, b3, b4, cct, cr1, cr2, cϵ0, cϵ1, cϵ2]

benchmark_result = @benchmark theory(θ, n, z, Mono_Emu, Quad_Emu, n_bar)
println(benchmark_result)######################################


# LETS TRY PROFILE LIKELIHOOD
# generate some synthetic data
println("define new functions and quantities")####################################

cov_20=readdlm("data/CovaPTcov_gaussian_AbacusFid_z0.8_n5e-4_Mono_Quad_Hexa.dat", ' ',Float64)
indices = [3:20;33:50] # unsure what the indices here represent
cov_20=cov_20[indices,indices];
#println(cov_20)
cosmoidx = 1; #using abacus c000 cosmology, just for example
datavec4test=theory(θ, n, z, Mono_Emu, Quad_Emu, n_bar);
θ
println("reads in covariance matrix and stuff")################################### 
@model function model_fixed(data, cov, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, fixed_value, param_idx)
    ln10As = cosmo[cosmoidx,4] # want to eventually make these into priors as well
    ns     = cosmo[cosmoidx,5]
    h      = cosmo[cosmoidx,3]
    ωb     = cosmo[cosmoidx,1]
    ωc     = cosmo[cosmoidx,2]
    w0     = cosmo[cosmoidx,6]
    wa     = cosmo[cosmoidx,7]
    Mν     = 0.06

    # Free parameters except the one being profiled
    b1     ~ Uniform(0., 12.)
    b2     ~ Uniform(-24., 24.)
    b3     ~ Uniform(-90., 90.)        
    b4     ~ Uniform(-24., 24.)
    cct    ~ Uniform(-36., 36.)
    cr1    ~ Uniform(-72., 72.)
    cr2    = 0
    cϵ0    ~ Uniform(-24., 24.)
    cϵ1    = 0
    cϵ2    ~ Uniform(-72., 72.)

    # Fix the parameter at param_idx to fixed_value
    param_list = [b1, b2, b3, b4, cct, cr1, cr2, cϵ0, cϵ1, cϵ2]
    param_list[param_idx] = fixed_value

    θ = [ln10As, ns, h, ωb, ωc, Mν, w0, wa, param_list...]

    prediction = theory(θ, n, z, Mono_Emu, Quad_Emu, n_bar)

    data ~ MvNormal(prediction, cov)
    return nothing
end

function run_optimize(N,model)
    # Initialize variables to track the best fit
    max_lp = -Inf
    best_fit = nothing

    for i in 1:N
        # Your model fitting here, replace `model_20` and `MAP()` with your actual model and prior
        current_fit = optimize(model, MLE(), Optim.Options(iterations=10000, allow_f_increases=true)) #if you change MLE() to MAP(), it will look for the maximum a posterior rather than the maximum likelihood

        # Check if the current fit's lp is the largest we've seen
        if current_fit.lp > max_lp
            max_lp = current_fit.lp
            best_fit = current_fit
        end
    end
    return best_fit
end;


function compute_profile_likelihood(param_idx, param_values, data, cov, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar)
    # param_idx: Index of the parameter to be analyzed (e.g., 1 for b1, 2 for b2, etc.)
    # param_values: Vector of values for the parameter to be analyzed
    # Other arguments are required inputs for the modified model

    bins = length(param_values)
#    # Define helper function
 #   function compute_likelihood(fixed_value)
 #       fit_result = run_optimize(5, model_fixed(data, cov, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, fixed_value, param_idx))
 #       return fit_result.lp
  #  end
  #  # Parallelize computation using pmap
  #  profile_likelihood = pmap(compute_likelihood, param_values)
    profile_likelihood = Vector{Float64}(undef, bins)
    param_values2 = Vector{Float64}(undef, bins)

    # Loop over each value to evaluate the profile likelihood
    for i in 1:bins
        fixed_value = param_values[i]

        # Fit the modified model
        fit_result = run_optimize(8, model_fixed(data, cov, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, fixed_value, param_idx))##### changed from 5 to 8
     #   println(fit_result, "\n")#############################
     #   println(fit_result.lp, "\n\n")###########################
        println(typeof(fit_result), typeof(fit_result.lp), "\n\n\n")##############################
        param_values2[i] = fit_result[2]######################################################
       # println(fit_result.value, "\n\n")
        # Store the log-posterior value (chi2) for the current fixed parameter value
        profile_likelihood[i] = fit_result.lp
    end

    return param_values, profile_likelihood, param_values2#####################################################
end;

param_idx = 1#param_idx = 1  # For b1
param_values=collect(1.9:0.01:2.3);#param_values = collect(1.7:0.01:2.5);#collect(1.7:0.01:2.5);  # Values for b1
println("defined more functions and parameters") #################################

benchmark_result = @benchmark run_optimize(8, model_fixed(datavec4test, cov_20, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar, 2, 1))
println(benchmark_result)#####################################

param_values, profile_likelihood, param_values2 = compute_profile_likelihood(param_idx, param_values, datavec4test, cov_20, n, z, cosmoidx, Mono_Emu, Quad_Emu, n_bar)###################################################
println("finished more calculations")####################################

using Plots

delta_chi2 = -(profile_likelihood .- maximum(profile_likelihood))
#plot(param_values, delta_chi2, xlabel="Parameter Values", ylabel="Δχ²", title="Profile Likelihood", legend=false)
plot(param_values2, delta_chi2, xlabel="Parameter Values", ylabel="Δχ²", title="Profile Likelihood", legend=false, seriestype=:scatter)####################################
savefig("/home/jgmorawe/projects/rrg-wperciva/jgmorawe/results/profile_likelihood_test.png")
println("finished plotting")####################################
