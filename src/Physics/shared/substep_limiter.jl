#####
##### Dynamic substep limiter for stiff source-term integration.
#####
##### Caps the per-cell action-density change in a single substep:
#####
#####     ΔN_m = min(ΔN_p, ΔN_r),  N^{n+1} = N^n + clamp(ΔN, -ΔN_m, ΔN_m)
#####
##### where
#####
#####     ΔN_p = X_p · α · (2π)^4 / (g²·σ·k³)    (Pierson-Moskowitz parametric bound)
#####     ΔN_r = X_r · max(N, X_f · max_cell(N))  (relative cap with filter floor)
#####
##### WW3 defaults: X_p = 0.15, X_r = 0.10, X_f = 0.05 (§3.6, Table 3.1).
##### Necessary when ST3/ST4 sources push some cells stiff while the global
##### Δt_g is dictated by transport CFL.

struct DynamicSubstepLimiter{FT}
    X_p     :: FT      # parametric bound coefficient
    X_r     :: FT      # relative bound coefficient
    X_f     :: FT      # filter floor (fraction of domain maximum)
    Δt_min  :: FT      # minimum allowed substep (safeguard)
    alpha   :: FT      # PM spectrum equilibrium level (typically 0.62e-4)
end

DynamicSubstepLimiter(; X_p=0.15, X_r=0.10, X_f=0.05,
                        Δt_min=0.5, alpha=0.62e-4) =
    DynamicSubstepLimiter(float(X_p), float(X_r), float(X_f),
                          float(Δt_min), float(alpha))

parametric_action_bound(lim::DynamicSubstepLimiter, σ, k; g=9.81) =
    lim.X_p * lim.alpha * (2 * pi)^4 / (g^2 * σ * k^3)
