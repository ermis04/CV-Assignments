function x = GenOMP(D, y, T0, err)
[~, K] = size(D);

x = zeros(K, 1);
r = y;
S = [];
while (norm(r, 2)^2 > err) && (length(S) < T0)
    correlations = abs(D' * r);
    if ~isempty(S)
        correlations(S) = -Inf;
    end


    [~, k] = max(correlations);

    if isinf(correlations(k))
        break;
    end

    S = [S, k];
    D_S = D(:, S);

    % Least squares λύση πάνω στα επιλεγμένα atoms.
    % Αυτό δίνει καλύτερους συντελεστές για όλα τα atoms του S.
    x_S = D_S \ y;
    r = y - D_S * x_S;
end
if ~isempty(S)
    x(S) = x_S;
end

end