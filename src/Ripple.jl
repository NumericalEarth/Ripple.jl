module Ripple

include("Architectures.jl")
include("Locations.jl")
include("Grids.jl")

include("ProductFields/ProductFields.jl")
include("CoordinateGrids/CoordinateGrids.jl")
include("Diagnostics/Diagnostics.jl")
include("InitialConditions/InitialConditions.jl")
include("Coupling/Coupling.jl")
include("Forcing/Forcing.jl")
include("Sources/Sources.jl")
include("Models/Models.jl")
include("OceananigansIntegration.jl")
include("Validation/Validation.jl")

export CPU, GPU, architecture
export Center, Face, Flat
export Periodic, Bounded, NoFlux, ProductBoundaryConditions, default_wave_action_bcs
export RectilinearGrid, horizontal_size, vertical_size
export xnodes, ynodes, znodes, xfaces, yfaces, zfaces
export xspacings, yspacings, zspacings
export ProductField, WaveActionField
export grid, physical_grid, coordinate_grid, product_grid
export location, coordinate_location, product_location, active_product_location
export interior, boundary_conditions, parent, getnode, setnode!, fill_halo_regions!
export physical_field
export PolarWaveVectorGrid, CartesianWaveVectorGrid, FrequencyDirectionGrid
export coordinate_size, coordinate_centers, coordinate_faces, coordinate_spacings
export spectral_cell_measure, spectral_cell_measures, spectral_weight, spectral_weights
export spectral_first_moment_measures, spectral_second_moment_measures
export spectral_radial_power_measure, spectral_radial_power_average
export spectral_frequency_power_measure, spectral_frequency_power_average
export k_components, radial_wavenumber, metric_jacobian, integrate_spectrum
export m0, first_moment, second_moment, mean_square_wavenumber, root_mean_square_wavenumber
export mean_direction_vector, mean_direction, peak_direction
export peak_wavenumber, deep_water_peak_phase_speed, wave_age
export mean_frequency, mean_period, peak_frequency, peak_period
export significant_wave_height, total_action
export deep_water_energy_density, total_deep_water_energy, mean_deep_water_group_speed
export JONSWAPSpectrum, GaussianWavePacket, set!
export Clockwise, Counterclockwise, LinearStormTrack
export StationaryVortexWind, IdealizedHurricaneWind, HollandHurricaneWind
export wind_velocity, wind_speed, wind_angle
export QKernel, QTransform, OnTheFlyQ, CacheDopplerVelocity, CacheDopplerVelocityAndDerivative
export PrecomputeQWeights, vertical_nodes, vertical_faces
export PrescribedLagrangianMeanCurrent, NoCurrentCoupling, CWCMPrescribedCurrentCoupling
export AbstractLagrangianVelocities, ZeroVelocities, PrescribedVelocities, PseudomomentumVelocities
export q_value, q_cell_integral, q_cell_integral_kappa_derivative
export compute_doppler_velocity!, compute_doppler_velocity_derivative!, compute_pseudomomentum
export compute_wave_current_refraction_tendency!
export pseudomomentum_field, pseudomomentum_fields
export vertical_spacings, vertical_integral
export compute_pseudomomentum_cell_integrals, compute_pseudomomentum_cell_integrals!
export compute_pseudomomentum_cell_averages, compute_pseudomomentum_cell_averages!
export compute_pseudomomentum_tendency_cell_averages!, pseudomomentum_tendency_fields
export cwcm_momentum_tendency_fields!
export update_coupling!
export NoSource, SourceTermSet, RelaxationToSpectrum
export LinearWindInput, ExponentialWindInput, PowerLawWindInput, WaveAgeWindInput, SaturationDissipation
export WhitecappingDissipation, FrequencyDissipation, WavenumberDissipation
export MeanFrequencyDissipation, PeakFrequencyDissipation, MeanSquareWavenumberDissipation
export PeakWavenumberDissipation
export DepthLimitedBreaking, BottomFriction
export IceDamping, SwellDissipation, DirectionalDiffusion, DirectionalAdvection
export MeanDirectionDissipation
export RadialDiffusion, RadialAdvection, SpectralTransferInteraction, NonlinearSpectralTransfer
export SpectralTransferStencil, NonlinearInvariantTransfer
export TriadTransferInteraction, TriadSpectralTransfer
export QuadrupletTransferInteraction, DiscreteInteractionApproximation
export source_tendency, source_split, implicit_source_rate
export SpectralWaveModel, Clock, fields, prognostic_fields, compute_tendencies!, time_step!, cfl
export Centered, UpwindBiased, WENO, FluxFormAdvection
export AbstractGSEAlleviation, SpatialAveraging, apply_gse_alleviation!
export ValidationCase, ValidationResult, default_validation_cases, run_validation
export validation_passed, write_validation_summary, read_validation_summary
export ExternalComparisonResult, compare_validation_summaries
export ExternalMetric, parse_external_metrics, write_external_metrics_summary
export run_external_metrics_command
export ExternalModelInputDeck, ExternalModelLaunchPlan, ExternalModelLaunchProfile
export external_model_input_deck, write_external_model_input_deck
export external_model_launch_profile, external_model_launch_plan, run_external_model_launch_plan!
export external_model_executable_env_var, external_model_workdir_env_var
export parse_external_bulk_table, external_bulk_table_metrics
export write_external_bulk_metrics_summary
export PerformanceMetric, run_performance_smoke
export write_performance_summary, read_performance_summary

end
