using Pkg
Pkg.activate("/home/jgmorawe/FrequentistExample")
using ArgParse
using Distributed

# Add worker processes (subtracting 1 for the master process)
n_bins = 50
addprocs(n_bins)

# Load necessary modules on all processes
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
    using SharedArrays
    using OptimizationOptimJL: ParticleSwarm
end

function main()

    param_label = "Om"#####################################
    config = ArgParseSettings()
    @add_arg_table config begin
        "--tracer"
        help = "Specify the tracer"
        arg_type = String
        required = true
        "--lower"
        help = "Specify the lower bound"
        arg_type = Float64
        required = true
        "--upper"
        help = "Specify the upper bound"
        arg_type = Float64
        required = true
        "--init_values"
        help = "Specify the initial guesses for the other parameters"
        arg_type = String
        required = true
        "--desidir"
        help = "DESI data directory"
        arg_type = String
        default = "/home/jgmorawe/scratch/DESI/"
        "--emudir"
        help = "Emulator directory"
        arg_type = String
        default = "/home/jgmorawe/FrequentistExample/DESI_emulator/batch_trained_velocileptors_james_effort_wcdm_20000/"
    end
    
    parsed_args = parse_args(config)
    tracer = parsed_args["tracer"]
    lower = parsed_args["lower"]
    upper = parsed_args["upper"]
    init_values = parse.(Float64, split(parsed_args["init_values"], ","))
    outpath = "/home/jgmorawe/FrequentistExample/profile_likelihoods_w0wacdm/$(tracer)_$(param_label).txt"
    desidir = parsed_args["desidir"]
    emudir = parsed_args["emudir"]
    
    function get_info(tracer, desidir)
        tracers = ["BGS","LRG1","LRG2","LRG3","ELG2","QSO"]
        redshift_ranges = ["z0.1-0.4", "z0.4-0.6", "z0.6-0.8", "z0.8-1.1", "z1.1-1.6", "z0.8-2.1"]
        redshift_eff = vec(readdlm(desidir*"zeff_pk.txt", ' '))
        nz = [5e-4, 5e-4, 5e-4, 3e-4, 5e-4, 3e-5]
        redshift_indexes = [1, 2, 3, 4, 6, 7]
        zranges = Dict(zip(tracers, redshift_ranges))
        zindexes = Dict(zip(tracers, redshift_indexes))
        nzall = Dict(zip(tracers, nz))
        zeffall = Dict(zip(tracers, redshift_eff))
        pkpath = Dict(tracer => desidir * "pk_" * tracer * "_" * zranges[tracer] * ".txt" for tracer in tracers)
        kinpath = Dict(tracer => desidir * "kin_" * tracer * "_" * zranges[tracer] * ".txt" for tracer in tracers)
        wmatpath = Dict(tracer => desidir * "wmatrix_" * tracer * "_" * zranges[tracer] * ".txt" for tracer in tracers)
        invcovpath = Dict(tracer => desidir * "invcov_pk_" * tracer * "_" * zranges[tracer] * ".txt" for tracer in tracers)
        pkall = Dict(tracer => vec(readdlm(pkpath[tracer], ' ')) for tracer in tracers)
        kinall = Dict(tracer => vec(readdlm(kinpath[tracer], ' ')) for tracer in tracers)
        wmatall = Dict(tracer => readdlm(wmatpath[tracer], ' ') for tracer in tracers)
        invcovall = Dict(tracer => readdlm(invcovpath[tracer], ' ') for tracer in tracers)
        covall = Dict(tracer => inv(invcovall[tracer]) for tracer in tracers)
        MonoPath = Dict(tracer => emudir * string(zindexes[tracer]) * "/0/" for tracer in tracers)
        QuadPath = Dict(tracer => emudir * string(zindexes[tracer]) * "/2/" for tracer in tracers)
        HexaPath = Dict(tracer => emudir * string(zindexes[tracer]) * "/4/" for tracer in tracers)
        tracer_vector = Vector{String}(split(tracer, ","))
        return (tracer_vector, MonoPath, QuadPath, HexaPath, nzall, zeffall, pkall, kinall, wmatall, covall)
    end
    (tracer_vector, MonoPath, QuadPath, HexaPath, nzall, zeffall, pkall, kinall, wmatall, covall) = get_info(tracer, desidir)

    param_values = range(lower, stop=upper, length=n_bins)
    profile_values = SharedArray{Float64}(n_bins)
    bestfit_values = SharedArray{Float64}(n_bins, 6+7*size(tracer_vector)[1]) # 6 cosmological parameters + 7*(number of tracers) total entries
    
    @everywhere tracer_vector = $tracer_vector
    @everywhere nzall = $nzall
    @everywhere zeffall = $zeffall
    @everywhere pkall = $pkall
    @everywhere kinall = $kinall
    @everywhere wmatall = $wmatall
    @everywhere covall = $covall
    @everywhere emudir = $emudir
    @everywhere MonoPath = $MonoPath
    @everywhere QuadPath = $QuadPath
    @everywhere HexaPath = $HexaPath
    @everywhere init_values = $init_values
    
    @everywhere begin
        Mono_Emu = Dict(tracer => Effort.load_multipole_noise_emulator(MonoPath[tracer]) for tracer in tracer_vector)
        Quad_Emu = Dict(tracer => Effort.load_multipole_noise_emulator(QuadPath[tracer]) for tracer in tracer_vector)
        Hexa_Emu = Dict(tracer => Effort.load_multipole_noise_emulator(HexaPath[tracer]) for tracer in tracer_vector)
    end
    @everywhere Mono_Emu = $(Mono_Emu)
    @everywhere Quad_Emu = $(Quad_Emu)
    @everywhere Hexa_Emu = $(Hexa_Emu)
    
    print("loading done, computing profile likelihood")
    
    N_runs = 50##########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    @sync @distributed for index in 1:length(param_values)
        @time (profile_values[index],bestfit_values[index,:]) = run_worker(param_values[index], tracer_vector, nzall, zeffall, pkall, kinall, wmatall, covall, Mono_Emu, Quad_Emu, Hexa_Emu, init_values, N_runs)
    end
    
    data_to_write = hcat(param_values, profile_values, bestfit_values)
    writedlm(outpath, data_to_write)
end

@everywhere function run_worker(fixed_value, tracer_vector, nzall, zeffall, pkall, kinall, wmatall, covall, Mono_Emu, Quad_Emu, Hexa_Emu, init_values, N)
    fit_model = model(fixed_value, tracer_vector, nzall, zeffall, pkall, kinall, wmatall, covall, Mono_Emu, Quad_Emu, Hexa_Emu)
    max_lp = -Inf
    best_fit = nothing
    for i in 1:N # runs N separate runs to avoid convergence issues
        fit_result = maximum_likelihood(fit_model, LBFGS(m=100, P=Diagonal([1/0.3, 1/0.01, 1/2, 1/0.0001, 1/0.1, 1/0.5, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1])))#, P=Diagonal([1/0.3, 1/0.01, 1/2, 1/0.0001, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1, 1/0.3, 1/1, 1/1, 1/10, 1/20, 1/1, 1/1])); initial_params=init_values)#_guess)#LBFGS(m=20, P=Diagonal([1/0.03, 1/2, 1/0.00055, 1/0.015, 1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1])))#m=100, P=Diagonal([1/0.03, 1/2, 1/0.00055, 1/0.015, 1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1,1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1,1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1,1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1,1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1,1/0.7, 1/3, 1/1, 1/50, 1/70, 1/5, 1/1])); initial_params=init_values)#1/0.01, 1/0.1, 1/0.0001, 1/0.005, 1/0.05, 1/1, 1/1, 1/2, 1/2, 1/50, 1/500, 1/0.05, 1/1, 1/1, 1/2, 1/2, 1/50, 1/500, 1/0.05, 1/1, 1/1, 1/2, 1/2, 1/50, 1/500]))) # sets rough step sizes corresponding to each dimension of parameter space###############################################################################################################################################
        if fit_result.lp > max_lp
            max_lp = fit_result.lp
            best_fit = fit_result
        end
    end
    return (best_fit.lp,best_fit.values.array)
end

@everywhere function theory(θ, Mono_Emu, Quad_Emu, Hexa_Emu, kin)
    my_θ = deepcopy(θ)

    PK0 = Effort.get_Pℓ(my_θ[1:7],my_θ[8:18],Mono_Emu)
    PK2 = Effort.get_Pℓ(my_θ[1:7],my_θ[8:18],Quad_Emu)
    PK4 = Effort.get_Pℓ(my_θ[1:7],my_θ[8:18],Hexa_Emu)

    Pks_fine = vcat(Effort._akima_spline(PK0, Mono_Emu.Pℓ.P11.kgrid, kin),
                    Effort._akima_spline(PK2, Quad_Emu.Pℓ.P11.kgrid, kin),
                    Effort._akima_spline(PK4, Hexa_Emu.Pℓ.P11.kgrid, kin))
    return Pks_fine
end


@everywhere @model function model(fixed_value, tracer_vector, nzall, zeffall, pkall, kinall, wmatall, covall, Mono_Emu, Quad_Emu, Hexa_Emu)

    ln10As ~ Uniform(2, 3.5)
    ns ~ Normal(0.9649, 0.042)
    h ~ Uniform(50, 80)
    ωb ~ Normal(0.02218, 0.00055)
    Om = fixed_value
    ωc = Om*h*h/10000-ωb-0.00064419153 # converts from cdm to total matter basis
    w0 ~ Uniform(-2, 0.5)
    wa ~ Uniform(-3, 1.641375)

    for tracer in tracer_vector
        if tracer == "BGS"
            b1_bgs ~ Uniform(-1., 4.)
            b2_bgs ~ Normal(0., 8)
            b3_bgs = 0        
            bs_bgs ~ Normal(0., 8)        
            alpha0_bgs ~ Normal(0., 12.5)
            alpha2_bgs ~ Normal(0., 12.5)
            alpha4_bgs = 0
            alpha6_bgs = 0
            sn_bgs_tilde ~ Normal(0., 2.)
            sn_bgs = sn_bgs_tilde.*1000.0            
            sn2_bgs_tilde ~ Normal(0., 5.)
            sn2_bgs = sn2_bgs_tilde.*10000.0
            sn4_bgs = 0

            θ_bgs = [ln10As, ns, h, ωb, ωc, w0, wa, b1_bgs, b2_bgs, b3_bgs, bs_bgs,
                     alpha0_bgs, alpha2_bgs, alpha4_bgs, alpha6_bgs, sn_bgs, sn2_bgs, sn4_bgs]

            thevec_bgs = theory(θ_bgs, Mono_Emu[tracer], Quad_Emu[tracer], Hexa_Emu[tracer], kinall[tracer])

            windowed_thevec_bgs = wmatall[tracer] * thevec_bgs
            cov_bgs = (covall[tracer] + covall[tracer]') / 2
            pkall[tracer] ~ MvNormal(windowed_thevec_bgs, cov_bgs)

        elseif tracer == "LRG1"
            b1_lrg1 ~ Uniform(-1., 4.)
            b2_lrg1 ~ Normal(0., 8)
            b3_lrg1 = 0        
            bs_lrg1 ~ Normal(0., 8)        
            alpha0_lrg1 ~ Normal(0., 12.5)
            alpha2_lrg1 ~ Normal(0., 12.5)
            alpha4_lrg1 = 0
            alpha6_lrg1 = 0
            sn_lrg1_tilde ~ Normal(0., 2.)
            sn_lrg1 = sn_lrg1_tilde.*1000.0            
            sn2_lrg1_tilde ~ Normal(0., 5.)
            sn2_lrg1 = sn2_lrg1_tilde.*10000.0
            sn4_lrg1 = 0

            θ_lrg1 = [ln10As, ns, h, ωb, ωc, w0, wa, b1_lrg1, b2_lrg1, b3_lrg1, bs_lrg1,
                      alpha0_lrg1, alpha2_lrg1, alpha4_lrg1, alpha6_lrg1, sn_lrg1, sn2_lrg1, sn4_lrg1]

            thevec_lrg1 = theory(θ_lrg1, Mono_Emu[tracer], Quad_Emu[tracer], Hexa_Emu[tracer], kinall[tracer])

            windowed_thevec_lrg1 = wmatall[tracer] * thevec_lrg1
            cov_lrg1 = (covall[tracer] + covall[tracer]') / 2
            pkall[tracer] ~ MvNormal(windowed_thevec_lrg1, cov_lrg1)

        elseif tracer == "LRG2"
            b1_lrg2 ~ Uniform(-1., 4.)
            b2_lrg2 ~ Normal(0., 8)
            b3_lrg2 = 0        
            bs_lrg2 ~ Normal(0., 8)        
            alpha0_lrg2 ~ Normal(0., 12.5)
            alpha2_lrg2 ~ Normal(0., 12.5)
            alpha4_lrg2 = 0
            alpha6_lrg2 = 0
            sn_lrg2_tilde ~ Normal(0., 2.)
            sn_lrg2 = sn_lrg2_tilde.*1000.0            
            sn2_lrg2_tilde ~ Normal(0., 5.)
            sn2_lrg2 = sn2_lrg2_tilde.*10000.0
            sn4_lrg2 = 0

            θ_lrg2 = [ln10As, ns, h, ωb, ωc, w0, wa, b1_lrg2, b2_lrg2, b3_lrg2, bs_lrg2,
                      alpha0_lrg2, alpha2_lrg2, alpha4_lrg2, alpha6_lrg2, sn_lrg2, sn2_lrg2, sn4_lrg2]

            thevec_lrg2 = theory(θ_lrg2, Mono_Emu[tracer], Quad_Emu[tracer], Hexa_Emu[tracer], kinall[tracer])

            windowed_thevec_lrg2 = wmatall[tracer] * thevec_lrg2
            cov_lrg2 = (covall[tracer] + covall[tracer]') / 2
            pkall[tracer] ~ MvNormal(windowed_thevec_lrg2, cov_lrg2)

        elseif tracer == "LRG3"
            b1_lrg3 ~ Uniform(-1., 4.)
            b2_lrg3 ~ Normal(0., 8)
            b3_lrg3 = 0        
            bs_lrg3 ~ Normal(0., 8)        
            alpha0_lrg3 ~ Normal(0., 12.5)
            alpha2_lrg3 ~ Normal(0., 12.5)
            alpha4_lrg3 = 0
            alpha6_lrg3 = 0
            sn_lrg3_tilde ~ Normal(0., 2.)
            sn_lrg3 = sn_lrg3_tilde.*1000.0            
            sn2_lrg3_tilde ~ Normal(0., 5.)
            sn2_lrg3 = sn2_lrg3_tilde.*10000.0
            sn4_lrg3 = 0

            θ_lrg3 = [ln10As, ns, h, ωb, ωc, w0, wa, b1_lrg3, b2_lrg3, b3_lrg3, bs_lrg3,
                      alpha0_lrg3, alpha2_lrg3, alpha4_lrg3, alpha6_lrg3, sn_lrg3, sn2_lrg3, sn4_lrg3]

            thevec_lrg3 = theory(θ_lrg3, Mono_Emu[tracer], Quad_Emu[tracer], Hexa_Emu[tracer], kinall[tracer])

            windowed_thevec_lrg3 = wmatall[tracer] * thevec_lrg3
            cov_lrg3 = (covall[tracer] + covall[tracer]') / 2
            pkall[tracer] ~ MvNormal(windowed_thevec_lrg3, cov_lrg3)

        elseif tracer == "ELG2"
            b1_elg2 ~ Uniform(-1., 4.)
            b2_elg2 ~ Normal(0., 8)
            b3_elg2 = 0        
            bs_elg2 ~ Normal(0., 8)        
            alpha0_elg2 ~ Normal(0., 12.5)
            alpha2_elg2 ~ Normal(0., 12.5)
            alpha4_elg2 = 0
            alpha6_elg2 = 0
            sn_elg2_tilde ~ Normal(0., 2.)
            sn_elg2 = sn_elg2_tilde.*1000.0            
            sn2_elg2_tilde ~ Normal(0., 5.)
            sn2_elg2 = sn2_elg2_tilde.*10000.0
            sn4_elg2 = 0

            θ_elg2 = [ln10As, ns, h, ωb, ωc, w0, wa, b1_elg2, b2_elg2, b3_elg2, bs_elg2,
                      alpha0_elg2, alpha2_elg2, alpha4_elg2, alpha6_elg2, sn_elg2, sn2_elg2, sn4_elg2]

            thevec_elg2 = theory(θ_elg2, Mono_Emu[tracer], Quad_Emu[tracer], Hexa_Emu[tracer], kinall[tracer])

            windowed_thevec_elg2 = wmatall[tracer] * thevec_elg2
            cov_elg2 = (covall[tracer] + covall[tracer]') / 2
            pkall[tracer] ~ MvNormal(windowed_thevec_elg2, cov_elg2)

        elseif tracer == "QSO"
            b1_qso ~ Uniform(-1., 4.)
            b2_qso ~ Normal(0., 8)
            b3_qso = 0        
            bs_qso ~ Normal(0., 8)        
            alpha0_qso ~ Normal(0., 12.5)
            alpha2_qso ~ Normal(0., 12.5)
            alpha4_qso = 0
            alpha6_qso = 0
            sn_qso_tilde ~ Normal(0., 2.)
            sn_qso = sn_qso_tilde.*1000.0          
            sn2_qso_tilde ~ Normal(0., 5.)
            sn2_qso = sn2_qso_tilde.*10000.0
            sn4_qso = 0

            θ_qso = [ln10As, ns, h, ωb, ωc, w0, wa, b1_qso, b2_qso, b3_qso, bs_qso,
                     alpha0_qso, alpha2_qso, alpha4_qso, alpha6_qso, sn_qso, sn2_qso, sn4_qso]

            thevec_qso = theory(θ_qso, Mono_Emu[tracer], Quad_Emu[tracer], Hexa_Emu[tracer], kinall[tracer])

            windowed_thevec_qso = wmatall[tracer] * thevec_qso
            cov_qso = (covall[tracer] + covall[tracer]') / 2
            pkall[tracer] ~ MvNormal(windowed_thevec_qso, cov_qso)
        end
    end
    ωb_BBN = 0.02218
    sigma_ωb = 0.00055 # adds in the BBN prior
    dωb = ωb - ωb_BBN
    ns_plk = 0.9649
    sigma_ns = 0.042
    dns = ns - ns_plk
    
    Turing.@addlogprob! - 0.5 * dωb^2/sigma_ωb^2
    Turing.@addlogprob! - 0.5 * dns^2/sigma_ns^2
    
    return nothing
end

# Call the main function to execute the program
main()