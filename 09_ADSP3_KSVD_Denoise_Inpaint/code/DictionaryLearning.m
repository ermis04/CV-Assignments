function [MSE, D, X] = DictionaryLearning(D, Y, err, T0, numEpochs, X)
% DictionaryLearning
%
% Εκτελεί Dictionary Learning με δύο βήματα ανά epoch:
% 1. Sparse coding με GenOMP: βρίσκουμε X με γνωστά D και Y.
% 2. Dictionary update με KSVD: ενημερώνουμε D με γνωστά X και Y.
%
% Inputs:
% D         : αρχικό dictionary, n x K
% Y         : training data, n x N
% err       : tolerated residual error για GenOMP
% T0        : μέγιστο πλήθος atoms ανά αναπαράσταση
MSE = zeros(1, numEpochs);

for epoch = 1:numEpochs

    disp(['epoch ', num2str(epoch)]);

    for i = 1:size(Y, 2)
        X(:, i) = GenOMP(D, Y(:, i), T0, err);
    end
    [D, X] = KSVD(D, X, Y);
    D = normalizeColumnsLocal(D);
    MSE(epoch) = norm(Y - D * X, 'fro')^2 / numel(Y);

    disp(['mse ', num2str(MSE(epoch))]);
end

end

function A = normalizeColumnsLocal(A)
    norms = sqrt(sum(A.^2, 1));
    norms(norms == 0) = 1;
    A = A ./ norms;
end