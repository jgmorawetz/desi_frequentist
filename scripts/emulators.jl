using Pkg
Pkg.activate(".")
using Effort
using Capse


# Relevant folder paths and details
home_dir = "/global/homes/j/jgmorawe/FrequentistExample1/FrequentistExample1/"
FS_emu_dir = home_dir * "/FS_emulator/batch_trained_velocileptors_james_effort_wcdm_20000/"
BAO_emu_dir = home_dir * "/BAO_emulator_ln10As_version/"
sigma8_emu_dir = home_dir * "/BAO_emulator_sigma8_version/"
CMB_emu_dir = home_dir * "/CMB_emulator/"
tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
redshift_indices = [1, 2, 3, 4, 6, 7]
redshift_indices = Dict(zip(tracers, redshift_indices))

# Reads in the FS emulators
mono_paths = Dict(tracer => FS_emu_dir * string(redshift_indices[tracer]) * "/0/" for tracer in tracers)
quad_paths = Dict(tracer => FS_emu_dir * string(redshift_indices[tracer]) * "/2/" for tracer in tracers)
hexa_paths = Dict(tracer => FS_emu_dir * string(redshift_indices[tracer]) * "/4/" for tracer in tracers)
FS_emus = Dict(tracer => [Effort.load_multipole_noise_emulator(mono_paths[tracer]),
                          Effort.load_multipole_noise_emulator(quad_paths[tracer]),
                          Effort.load_multipole_noise_emulator(hexa_paths[tracer])] for tracer in tracers)

# Reads in the BAO emulator
BAO_emu = Effort.load_BAO_emulator(BAO_emu_dir)

# Reads in the sigma8 emulator (for converting to ln10As basis)
sigma_emu = Effort.load_BAO_emulator(sigma8_emu_dir)

# Reads in the CMB emulators
TT_emu = Capse.load_emulator(CMB_emu_dir * "/TT/")
TE_emu = Capse.load_emulator(CMB_emu_dir * "/TE/")
EE_emu = Capse.load_emulator(CMB_emu_dir * "/EE/")
CMB_emus = [TT_emu, TE_emu, EE_emu]