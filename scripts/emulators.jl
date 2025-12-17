using Pkg
Pkg.activate(".")
using Effort
using AbstractCosmologicalEmulators
using Capse


# Folder paths
base_dir = "/home/jgmorawe/projects/rrg-wperciva/jgmorawe/frequentist_project/"
FS_emu_dir = base_dir * "/trained_effort_velocileptors_rept_mnuw0wacdm_200000/"
BAO_ln10As_emu_dir = base_dir * "/trained_ace_mnuw0wacdm_ln10As_basis_200000/"
BAO_sigma8_emu_dir = base_dir * "/trained_ace_mnuw0wacdm_sigma8_basis_200000/"
CMB_emu_dir = base_dir * "/trained_capse_mnuw0wacdm_40000/"

# Loads the monopole/quadrupole/hexadecapole emulators for full-shape
FS_emus = [Effort.load_multipole_emulator(FS_emu_dir * "/0/"),
           Effort.load_multipole_emulator(FS_emu_dir * "/2/"),
           Effort.load_multipole_emulator(FS_emu_dir * "/4/")]

# Loads the BAO emulators (both for ln10As and sigma8 bases)
BAO_ln10As_emu = AbstractCosmologicalEmulators.load_trained_emulator(BAO_ln10As_emu_dir)
BAO_sigma8_emu = AbstractCosmologicalEmulators.load_trained_emulator(BAO_sigma8_emu_dir)

# Loads the CMB emulators for TT, TE and EE
TT_emu = Capse.load_emulator(CMB_emu_dir * "/TT/")
TE_emu = Capse.load_emulator(CMB_emu_dir * "/TE/")
EE_emu = Capse.load_emulator(CMB_emu_dir * "/EE/")
CMB_emus = [TT_emu, TE_emu, EE_emu]