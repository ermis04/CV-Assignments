function [D, X] = KSVD(D, X, Y)
[n, K] = size(D);

for k = 1:K

    % Βρίσκουμε τα training samples όπου χρησιμοποιείται το atom k.
    ind = find(X(k, :) ~= 0);

    % Αν το atom δεν χρησιμοποιείται πουθενά, το ξαναρχικοποιούμε.
    if isempty(ind)
        D(:, k) = randn(n, 1);
        D(:, k) = D(:, k) / norm(D(:, k), 2);
        continue;
    end

    % Error matrix χωρίς τη συνεισφορά του atom k.
    %
    % Y - D*X είναι το συνολικό σφάλμα.
    % Προσθέτουμε πίσω D(:,k)*X(k,:) για να αφαιρέσουμε
    % προσωρινά τη συνεισφορά του atom k από το D*X.
    E = Y - D * X + D(:, k) * X(k, :);

    % Κρατάμε μόνο τις στήλες όπου το atom k χρησιμοποιείται.
    E_r = E(:, ind);

    % Αν για κάποιο λόγο το reduced error είναι κενό, συνεχίζουμε.
    if isempty(E_r)
        continue;
    end

    % SVD στο reduced error.
    % Χρειαζόμαστε μόνο το πρώτο singular triplet.
    [U, S, V] = svd(E_r, 'econ');

    % Ενημέρωση atom.
    D(:, k) = U(:, 1);

    % Ενημέρωση μόνο των μη μηδενικών συντελεστών του X(k,:).
    X(k, :) = 0;
    X(k, ind) = S(1, 1) * V(:, 1)';
end

end