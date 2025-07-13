function y = G(x)
%G Quantum entropy function.
%   y = G(x) computes the entropy-like quantity used in Leak.m.
%   This function corresponds to the von Neumann entropy of a
%   bosonic thermal state with mean photon number x.

% Ensure numerical stability for x close to zero
x = max(x, 0);

y = (x + 1) .* log2(x + 1) - x .* log2(x);
end
