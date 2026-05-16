# Examples

The `examples/` directory is a curated literate-tutorial sequence. Each
example is a valid Julia script (executable with
`julia --project=docs examples/<name>.jl`) and is also rendered into this
documentation by [Literate.jl](https://github.com/fredrikekre/Literate.jl)
with `execute=true`, so every figure and animation on the example pages
is the actual output of running that script during the docs build.

The four example pages walk a deliberate arc:

1. **[Quick Start](@ref)** — a small barotropic-vortex refraction
   simulation that touches the major Ripple constructs (`RectilinearGrid`,
   `PolarWaveVectorGrid`, `SpectralWaveModel`, `velocities`,
   `Simulation`, the diagnostic suite) in roughly fifty lines.
2. **[Source-Only Fetch-Limited Growth](@ref)** — a single-column
   `horizontal_advection=nothing` source-balance test that approaches an
   analytic equilibrium under wind input and whitecapping dissipation.
3. **[Bounded Wave Packet Dispersion](@ref)** — physical transport in a
   bounded one-dimensional channel; group-velocity dispersion fans out a
   compact wavenumber packet across the domain.
4. **[Wave Refraction Through A Barotropic Vortex](@ref)** — the
   production-resolution version of the quick start, with a three-panel
   animation showing `m₀`, `κᵣₘₛ`, and the mean direction evolving under
   the fused Doppler + refraction kernel.

Run any example from the repository root:

```bash
julia --startup-file=no --project=docs examples/quick_start.jl
```

The smoke harness runs every checked-in example end-to-end:

```bash
julia --startup-file=no --project=. test/examples_smoke/run_examples.jl
```

```@contents
Pages = [
    "generated/examples/quick_start.md",
    "generated/examples/source_only_fetch_limited_growth.md",
    "generated/examples/bounded_wave_packet_dispersion.md",
    "generated/examples/vortex_refraction.md",
]
Depth = 1
```
