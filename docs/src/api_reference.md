# API Reference

This page groups the main exported names by workflow.

## Grids And Fields

- Architectures: `CPU`, `GPU`, `architecture`
- Physical grids: `RectilinearGrid`, `InfiniteDepth`, `horizontal_size`,
  `vertical_size`, `xnodes`, `ynodes`, `znodes`, `xfaces`, `yfaces`, `zfaces`,
  `xspacings`, `yspacings`, `zspacings`
- Product fields: `ProductGrid`, `ProductField`, `WaveActionField`, `grid`,
  `physical_grid`, `coordinate_grid`, `product_grid`, `interior`, `parent`,
  `physical_field`, `getnode`, `setnode!`, `set!`
- Locations and boundary conditions: `Center`, `Face`, `Periodic`, `Bounded`,
  `NoFlux`, `ProductBoundaryConditions`, `default_wave_action_bcs`

## Spectral Coordinates And Diagnostics

- Spectral grids: `CartesianWaveVectorGrid`, `PolarWaveVectorGrid`,
  `FrequencyDirectionGrid`
- Coordinate access: `coordinate_size`, `coordinate_centers`,
  `coordinate_faces`, `coordinate_spacings`, `k_components`,
  `radial_wavenumber`, `metric_jacobian`
- Exact finite-volume weights: `spectral_cell_measure`,
  `spectral_cell_measures`, `spectral_weight`, `spectral_weights`,
  `integrate_spectrum`
- Bulk diagnostics: `m0`, `first_moment`, `second_moment`,
  `mean_square_wavenumber`, `root_mean_square_wavenumber`,
  `mean_direction`, `peak_direction`, `mean_frequency`, `mean_period`,
  `peak_frequency`, `peak_period`, `significant_wave_height`,
  `total_action`, `deep_water_energy_density`,
  `total_deep_water_energy`, `mean_deep_water_group_speed`

## Models And Sources

- Model state: `SpectralWaveModel`, `Clock`, `fields`, `prognostic_fields`,
  `compute_tendencies!`, `time_step!`, `cfl`
- Physical transport: `Centered`, `UpwindBiased`, `WENO`, `FluxFormAdvection`
- Propagation smoothing: `AbstractPropagationSmoothing`, `SpatialAveraging`,
  `apply_propagation_smoothing!`
- Source composition: `GenericPhysics`, `NoPhysics`, `source_tendency`,
  `source_split`, `implicit_source_rate`
- Source pieces: `RelaxationToSpectrum`, `LinearWindInput`,
  `ExponentialWindInput`, `PowerLawWindInput`, `WaveAgeWindInput`,
  `WhitecappingDissipation`, `SaturationDissipation`,
  `FrequencyDissipation`, `WavenumberDissipation`, `BottomFriction`,
  `DepthLimitedBreaking`, `DirectionalDiffusion`, `DirectionalAdvection`,
  `RadialDiffusion`, `RadialAdvection`, `DiscreteInteractionApproximation`

## CWCM Coupling

- Kernels and transforms: `QKernel`, `QTransform`, `OnTheFlyQ`,
  `PrecomputeQWeights`, `CacheDopplerVelocity`,
  `CacheDopplerVelocityAndDerivative`
- Current coupling: `PrescribedLagrangianMeanCurrent`,
  `NoCurrentCoupling`, `AbstractCWCMCurrentCoupling`,
  `CWCMPrescribedCurrentCoupling`, `CWCMPseudomomentumCoupling`,
  `ZeroVelocities`, `PrescribedVelocities`, `PseudomomentumVelocities`,
  `q_cell_integral`, `q_cell_integral_kappa_derivative`,
  `compute_doppler_velocity!`, `compute_doppler_velocity_derivative!`,
  `update_coupling!`
- Pseudomomentum: `pseudomomentum_field`, `pseudomomentum_fields`,
  `compute_pseudomomentum`, `compute_pseudomomentum_cell_integrals`,
  `compute_pseudomomentum_cell_averages`,
  `compute_pseudomomentum_tendency_cell_averages!`,
  `cwcm_momentum_tendency_fields!`

## Validation

- Validation: `ValidationCase`, `ValidationResult`,
  `default_validation_cases`, `run_validation`, `validation_passed`,
  `write_validation_summary`, `read_validation_summary`,
  `compare_validation_summaries`, `run_performance_smoke`
