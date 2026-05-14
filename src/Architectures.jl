import Oceananigans
import Oceananigans.Architectures: AbstractArchitecture, CPU, GPU
import Oceananigans.Architectures: architecture, on_architecture

device_zeros(arch, ::Type{FT}, dims::Tuple) where FT =
    on_architecture(arch, zeros(FT, dims))
