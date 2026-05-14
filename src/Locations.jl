import Oceananigans
import Oceananigans: location
import Oceananigans.Grids: Center, Face, Periodic, Bounded, Flat

struct NoFlux end

Base.:(==)(::NoFlux, ::NoFlux) = true
