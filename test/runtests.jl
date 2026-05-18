using Test
using Ripple

function write_ripple_test_summary(path, testset)
    # Julia 1.10 returns a 9-tuple; 1.12+ returns a `Test.TestCounts` struct
    # with named fields. Handle both by reading the fields off whatever
    # `get_test_counts` returns.
    counts = Test.get_test_counts(testset)
    passes       = counts.passes
    fails        = counts.fails
    errors       = counts.errors
    broken       = counts.broken
    child_passes = counts.cumulative_passes
    child_fails  = counts.cumulative_fails
    child_errors = counts.cumulative_errors
    child_broken = counts.cumulative_broken
    duration     = counts.duration

    total_passes = passes + child_passes
    total_fails = fails + child_fails
    total_errors = errors + child_errors
    total_broken = broken + child_broken
    total = total_passes + total_fails + total_errors + total_broken

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "name\tpassed\tfailed\terrored\tbroken\ttotal\tduration")
        println(io, join(("Ripple.jl", total_passes, total_fails, total_errors,
                          total_broken, total, duration), '\t'))
    end

    return path
end

ripple_testset = @testset "Ripple.jl" begin
    include("product_fields/indexing.jl")
    include("coordinate_grids/finite_volume_integration.jl")
    include("initial_conditions/jonswap.jl")
    include("initial_conditions/gaussian_wave_packet.jl")
    include("diagnostics/moments.jl")
    include("diagnostics/cfl.jl")
    include("forcing/winds.jl")
    include("coupling/q_transform.jl")
    include("sources/source_terms.jl")
    include("physics/shared.jl")
    include("physics/st3.jl")
    include("physics/symmetric_quadruplet.jl")
    include("integration/model_api.jl")
    include("integration/transport.jl")
    include("validation/validation_suite.jl")
    include("examples_smoke/run_examples.jl")
end

if haskey(ENV, "RIPPLE_TEST_SUMMARY") && !isempty(ENV["RIPPLE_TEST_SUMMARY"])
    write_ripple_test_summary(ENV["RIPPLE_TEST_SUMMARY"], ripple_testset)
end
