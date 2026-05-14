struct GaussianWavePacket{FT}
    x0 :: FT
    y0 :: FT
    kx0 :: FT
    ky0 :: FT
    spatial_width :: FT
    spectral_width :: FT
    amplitude :: FT
end

function GaussianWavePacket(; x0=0.0, y0=0.0, kx0=1.0, ky0=0.0,
                              spatial_width=1.0, spectral_width=0.2, amplitude=1.0)
    values = promote(float(x0), float(y0), float(kx0), float(ky0),
                     float(spatial_width), float(spectral_width), float(amplitude))
    _, _, _, _, spatial_width_value, spectral_width_value, amplitude_value = values
    spatial_width_value > 0 || throw(ArgumentError("GaussianWavePacket spatial_width must be positive"))
    spectral_width_value > 0 || throw(ArgumentError("GaussianWavePacket spectral_width must be positive"))
    amplitude_value >= 0 || throw(ArgumentError("GaussianWavePacket amplitude must be nonnegative"))
    return GaussianWavePacket(values...)
end

function (g::GaussianWavePacket)(x, y, kx, ky)
    rsq = (x - g.x0)^2 + (y - g.y0)^2
    ksq = (kx - g.kx0)^2 + (ky - g.ky0)^2
    return g.amplitude * exp(-0.5rsq / g.spatial_width^2 - 0.5ksq / g.spectral_width^2)
end
