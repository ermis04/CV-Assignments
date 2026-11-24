clear;
close all;
clc;

rng(1);

train_set = dir('./dataset/train/*.jpg');
test_set  = dir('./dataset/test/*.jpg');

if isempty(train_set)
    error('No training images found in ./train/*.jpg');
end


patch_size = 8;

% Διαβάζουμε την πρώτη εικόνα μόνο για να πάρουμε τις διαστάσεις.
first_path = fullfile(train_set(1).folder, train_set(1).name);
first_im = imread(first_path);

if ndims(first_im) == 3
    first_im = rgb2gray(first_im);
end

first_im = im2double(first_im);

[H, W] = size(first_im);
H = floor(H / patch_size) * patch_size;
W = floor(W / patch_size) * patch_size;
patches_per_image = (H / patch_size) * (W / patch_size);
num_col_Y = patches_per_image * length(train_set);

% Το Y έχει μέγεθος 64 x number_of_patches, για patch_size = 8.
Y = zeros(patch_size^2, num_col_Y);

% Γεμίζουμε το Y με όλα τα patches από όλες τις training εικόνες.
patch_cnt = 1;

for i = 1:length(train_set)

    path = fullfile(train_set(i).folder, train_set(i).name);
    im = imread(path);

    if ndims(im) == 3
        im = rgb2gray(im);
    end

    im = im2double(im);

    % Κρατάμε τις ίδιες διαστάσεις για όλες τις εικόνες.
    im = im(1:H, 1:W);

    for patch_row = 1:patch_size:H
        for patch_column = 1:patch_size:W

            patch = im( ...
                patch_row:patch_row+patch_size-1, ...
                patch_column:patch_column+patch_size-1 ...
            );

            Y(:, patch_cnt) = reshape(patch, [patch_size^2, 1]);
            patch_cnt = patch_cnt + 1;
        end
    end
end

Y = normalizeColumns(Y);

numAtoms = 100;
D = randn(size(Y, 1), numAtoms);
D = normalizeColumns(D);
initial_D = D;
X = zeros(numAtoms, size(Y, 2));
err = 1e-4;
numEpochs = 20;
T0 = 20; % Κάθε patch θα αναπαρίσταται με το πολύ T0 atoms.

[MSE, new_D, X] = DictionaryLearning(D, Y, err, T0, numEpochs, X);

%%

figure;
semilogy(MSE, '-o', 'LineWidth', 1.5);
ylabel('$\|Y-DX\|_F^2 / numel(Y)$', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Epoch', 'Interpreter', 'latex', 'FontSize', 14);
title('Dictionary Learning Reconstruction Error');
grid on;


figure;
showDictionary(initial_D, patch_size, 'Initial Dictionary');

figure;
showDictionary(new_D, patch_size, 'Final Dictionary');

if isempty(test_set)
    warning('No testing images found in ./test/*.jpg');
else
    fprintf('\nTesting reconstruction errors:\n');

    for i = 1:length(test_set)

        test_path = fullfile(test_set(i).folder, test_set(i).name);
        test_im = imread(test_path);

        if ndims(test_im) == 3
            test_im = rgb2gray(test_im);
        end

        test_im = im2double(test_im);
        [H_test, W_test] = size(test_im);
        
        H_test = floor(H_test / patch_size) * patch_size;
        W_test = floor(W_test / patch_size) * patch_size;
        
        % Αν η εικόνα είναι μικρότερη από ένα patch, δεν μπορεί να χρησιμοποιηθεί.
        if H_test < patch_size || W_test < patch_size
            warning('Skipping %s because it is smaller than one patch.', test_set(i).name);
            continue;
        end
        
        % Κρατάμε μόνο το κομμάτι που χωρίζεται ακριβώς σε patches.
        test_im = test_im(1:H_test, 1:W_test);
        
        % Μετατροπή testing εικόνας σε matrix από patches.
        Y_test = imageToPatches(test_im, patch_size);

        % Κανονικοποίηση κάθε testing patch.
        Y_test = normalizeColumns(Y_test);

        X_test = zeros(numAtoms, size(Y_test, 2));
        for j = 1:size(Y_test, 2)
            X_test(:, j) = GenOMP(new_D, Y_test(:, j), T0, err);
        end

        % Ανακατασκευή testing patches.
        Y_test_rec = new_D * X_test;

        % Frobenius MSE για την testing εικόνα.
        test_mse = norm(Y_test - Y_test_rec, 'fro')^2 / numel(Y_test);

        fprintf('%s -> MSE = %.8f\n', test_set(i).name, test_mse);

        % Ανακατασκευή σε εικόνα για οπτικό έλεγχο.
        rec_im = patchesToImage(Y_test_rec, H_test, W_test, patch_size);

        figure;
        subplot(1, 2, 1);
        imshow(test_im, []);
        title('Original');

        subplot(1, 2, 2);
        imshow(rec_im, []);
        title(['Reconstructed, MSE = ', num2str(test_mse)]);
    end
end

function A = normalizeColumns(A)
    norms = sqrt(sum(A.^2, 1));
    norms(norms == 0) = 1;
    A = A ./ norms;
end

function Y = imageToPatches(im, patch_size)
    % Μετατρέπει μία εικόνα σε matrix Y.
    % Κάθε στήλη του Y είναι ένα vectorized patch.

    [H, W] = size(im);

    num_patches = (H / patch_size) * (W / patch_size);
    Y = zeros(patch_size^2, num_patches);

    patch_cnt = 1;

    for patch_row = 1:patch_size:H
        for patch_column = 1:patch_size:W

            patch = im( ...
                patch_row:patch_row+patch_size-1, ...
                patch_column:patch_column+patch_size-1 ...
            );

            Y(:, patch_cnt) = reshape(patch, [patch_size^2, 1]);
            patch_cnt = patch_cnt + 1;
        end
    end
end

function im = patchesToImage(Y, H, W, patch_size)
    % Μετατρέπει vectorized patches ξανά σε εικόνα.

    im = zeros(H, W);

    patch_cnt = 1;

    for patch_row = 1:patch_size:H
        for patch_column = 1:patch_size:W

            patch = reshape(Y(:, patch_cnt), [patch_size, patch_size]);

            im( ...
                patch_row:patch_row+patch_size-1, ...
                patch_column:patch_column+patch_size-1 ...
            ) = patch;

            patch_cnt = patch_cnt + 1;
        end
    end

    % Περιορισμός τιμών στο [0, 1] για imshow.
    im = min(max(im, 0), 1);
end

function showDictionary(D, patch_size, fig_title)
    % Οπτικοποιεί τα atoms του dictionary ως μικρά patches.

    numAtoms = size(D, 2);

    grid_size = ceil(sqrt(numAtoms));

    atom_min = min(D(:));
    atom_max = max(D(:));

    dictionary_image = ones( ...
        grid_size * patch_size, ...
        grid_size * patch_size ...
    );

    for k = 1:numAtoms

        row = floor((k - 1) / grid_size);
        col = mod(k - 1, grid_size);

        atom = reshape(D(:, k), [patch_size, patch_size]);

        % Κανονικοποίηση atom για εμφάνιση.
        atom = (atom - atom_min) / (atom_max - atom_min + eps);

        r_start = row * patch_size + 1;
        r_end = r_start + patch_size - 1;

        c_start = col * patch_size + 1;
        c_end = c_start + patch_size - 1;

        dictionary_image(r_start:r_end, c_start:c_end) = atom;
    end

    imshow(dictionary_image, []);
    title(fig_title);
end