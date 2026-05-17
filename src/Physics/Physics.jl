#####
##### Source-term physics for the spectral wave model.
#####
##### Adds:
#####
#####   - `PressureCorrelationInput` (Janssen-style wind input),
#####   - `LocalSaturationDissipation` (saturation-overshoot whitecapping),
#####   - `MeanSpectrumWhitecapping` (bulk-moments whitecapping),
#####   - `SymmetricQuadruplet`, a `method` for the existing
#####     `DiscreteInteractionApproximation` umbrella source term, computing
#####     the nonlinear quadruplet transfer via the single-λ symmetric model,
#####   - `PrecomputedSources`, a bundle that co-optimises evaluation of the
#####     above by precomputing per-grid-point state (stress cap, bulk
#####     moments, nonlinear-transfer field) via KernelAbstractions kernels
#####     once per tendency call.
#####
##### Every term is `<: AbstractSourceTerm` (defined in `src/Sources/Sources.jl`),
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
include("NonlinearInteractions/symmetric_quadruplet.jl")
include("Packages/precomputed_sources.jl")
