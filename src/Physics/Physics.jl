#####
##### Source-term physics bundles in the WW3 ST3/ST4 family.
#####
##### Adds three new source terms — `PressureCorrelationInput` (Janssen wind
##### input), `LocalSaturationDissipation` (Ardhuin ST4-style whitecapping),
##### `HasselmannDIA` (discrete-interaction quadruplet transfer) — and a
##### `MeanSpectrumPhysics` bundle that co-optimizes their evaluation by
##### precomputing per-grid-point state (stress cap, bulk moments, DIA
##### transfer field) via KernelAbstractions kernels once per tendency call.
#####
##### Every term is `<: AbstractSourceTerm` (from `src/Sources/Sources.jl`),
##### so the bundle integrates with the existing `sources=` machinery on
##### `SpectralWaveModel`.

import KernelAbstractions
import KernelAbstractions: @kernel, @index
import Oceananigans.Architectures: architecture, device

include("shared/drag.jl")
include("shared/parametric_tail.jl")
include("shared/substep_limiter.jl")
include("bundle_interface.jl")
include("WindInput/pressure_correlation.jl")
include("Dissipation/local_saturation.jl")
include("Dissipation/mean_spectrum.jl")
include("NonlinearInteractions/hasselmann_dia.jl")
include("Packages/mean_spectrum_physics.jl")
