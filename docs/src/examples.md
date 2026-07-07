# Examples

The `examples/` directory is a curated literate-tutorial sequence. Each
example is a valid Julia script (executable with
`julia --project=docs examples/<name>.jl`) and is also rendered into this
documentation by [Literate.jl](https://github.com/fredrikekre/Literate.jl)
with `execute=true`, so every figure and animation on the example pages
is the actual output of running that script during the docs build.

The example pages walk a deliberate arc:

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
4. **[Spectral Refraction by a Sheared Current](@ref)** — an isolated
   test of advection in spectral space: a uniform-in-space spectrum sees
   only ``c_\varphi``, and the mean direction at each ``y`` is checked
   against the linearised prediction
   ``\overline{\varphi}(y) \approx -T\,A\,\omega\,\cos(\omega y)``.
5. **[Wave Refraction Through A Barotropic Vortex](@ref)** — the
   production-resolution version of the quick start, with a three-panel
   animation showing `m₀`, `κᵣₘₛ`, and the mean direction evolving under
   the fused Doppler + refraction kernel.
6. **[Swell Generation by a Translating Idealized Hurricane](@ref)** — the
   ST3/ST4-equivalent physics bundle (Janssen wind input, saturation
   dissipation, Hasselmann DIA) driven by a translating Holland (1980)
   hurricane, showing the right-front extended-fetch enhancement and the
   trailing swell wake.
7. **[Narrow-Band Packet Dispersion](@ref)** — the "hello world" of the
   second model type, `NarrowBandWaveModel`: a Gaussian packet built on a
   carrier translates at the group velocity ``\omega_\kappa`` and spreads
   dispersively, illustrating the reconstituted dispersion relation and the
   amplitude (rather than action) formulation.

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
    "generated/examples/spectral_refraction_by_shear.md",
    "generated/examples/vortex_refraction.md",
    "generated/examples/translating_hurricane_swell.md",
    "generated/examples/narrow_band_packet_dispersion.md",
]
Depth = 1
```
