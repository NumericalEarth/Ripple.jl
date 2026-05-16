#####
##### Diagnostic high-frequency tail.
#####
##### Above the prognostic cutoff f_hf, the spectrum is extended analytically
##### as a power law N(k > k_max, θ) = N(k_max, θ) · (k/k_max)^(-m/2 - 1).
##### For energy density E ∝ f^{-m}; the standard WW3 default is m = 5
##### (Toba/Phillips equilibrium range).
#####
##### Used by ST3/ST4 to compute the wave-supported stress integral τ_w
##### up to k_max,tail beyond the prognostic spectrum.

struct DiagnosticTail{FT}
    power   :: FT      # m in f^{-m}
    f_cut   :: FT      # cutoff frequency above which the tail kicks in (Hz)
    f_max   :: FT      # upper bound for τ_w integration (Hz)
end

DiagnosticTail(; power=5.0, f_cut=0.625, f_max=10.0) =
    DiagnosticTail(float(power), float(f_cut), float(f_max))

action_tail_factor(tail::DiagnosticTail, k_ratio) =
    k_ratio^(-tail.power / 2 - 1)
