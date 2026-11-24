clear;
close all;
clc;

rng(1);

% Φόρτωση δοσμένων μητρώων

load('D.mat');   % φορτώνει n_D
load('X.mat');   % φορτώνει X
load('Y.mat');   % φορτώνει Y

D = n_D;
D = normalizeColumns(D);

% Παράμετροι

patch_size = 8;
T0 = 20;
err = 1e-4;

snr_values = [0, 20, 50, 100];
missing_percentages = [0, 20, 50, 100];

display_snr = 20;
display_missing = 20;

% Ανάγνωση testing εικόνων

if exist('./test', 'dir')
    test_folder = './test';
elseif exist('./dataset/test', 'dir')
    test_folder = './dataset/test';
elseif exist('./dataset/testing', 'dir')
    test_folder = './dataset/testing';
else
    test_folder = './dataset';
end

test_set = [
    dir(fullfile(test_folder, '*.jpg'));
    dir(fullfile(test_folder, '*.jpeg'));
    dir(fullfile(test_folder, '*.png'));
    dir(fullfile(test_folder, '*.bmp'))
];

if isempty(test_set)
    error('Δεν βρέθηκαν testing εικόνες στον φάκελο: %s', test_folder);
end

%% Denoising

denoise_mse = zeros(length(test_set), length(snr_values));

fprintf('\nDenoising reconstruction errors:\n');

for i = 1:length(test_set)

    image_path = fullfile(test_set(i).folder, test_set(i).name);
    original = readGrayImage(image_path);
    original = cropToPatchMultiple(original, patch_size);

    for s = 1:length(snr_values)

        current_snr = snr_values(s);

        noisy = addGaussianNoiseSNR(original, current_snr);

        denoised = dictionaryReconstruct(noisy, D, patch_size, T0, err);

        denoise_mse(i, s) = mean((original(:) - denoised(:)).^2);

        fprintf('%s | SNR %d dB -> MSE = %.8f\n', ...
            test_set(i).name, current_snr, denoise_mse(i, s));

        if current_snr == display_snr
            figure;
            subplot(1, 3, 1);
            imshow(original, []);
            title('Original');

            subplot(1, 3, 2);
            imshow(noisy, []);
            title(['Noisy, SNR = ', num2str(current_snr), ' dB']);

            subplot(1, 3, 3);
            imshow(denoised, []);
            title(['Denoised, MSE = ', num2str(denoise_mse(i, s), '%.6f')]);

            sgtitle(['Denoising: ', test_set(i).name]);
        end
    end
end

denoise_table = array2table(denoise_mse, ...
    'VariableNames', compose('SNR_%ddB', snr_values), ...
    'RowNames', {test_set.name});

disp(' ');
disp('Denoising MSE table:');
disp(denoise_table);

figure;
bar(denoise_mse);
legend(compose('%d dB', snr_values), 'Location', 'best');
set(gca, 'XTickLabel', {test_set.name});
xtickangle(45);
ylabel('MSE');
title('Denoising MSE per image and SNR');
grid on;

%% Inpainting

inpaint_mse = zeros(length(test_set), length(missing_percentages));

fprintf('\nInpainting reconstruction errors:\n');

for i = 1:length(test_set)

    image_path = fullfile(test_set(i).folder, test_set(i).name);
    original = readGrayImage(image_path);
    original = cropToPatchMultiple(original, patch_size);

    for m = 1:length(missing_percentages)

        missing_percentage = missing_percentages(m);

        [missing_image, mask] = createMissingPixels(original, missing_percentage);

        recovered = dictionaryReconstruct(missing_image, D, patch_size, T0, err);

        recovered_final = missing_image;
        recovered_final(mask == 0) = recovered(mask == 0);

        inpaint_mse(i, m) = mean((original(:) - recovered_final(:)).^2);

        fprintf('%s | Missing %d%% -> MSE = %.8f\n', ...
            test_set(i).name, missing_percentage, inpaint_mse(i, m));

        if missing_percentage == display_missing
            figure;
            subplot(1, 3, 1);
            imshow(original, []);
            title('Original');

            subplot(1, 3, 2);
            imshow(missing_image, []);
            title(['Missing ', num2str(missing_percentage), '%']);

            subplot(1, 3, 3);
            imshow(recovered_final, []);
            title(['Recovered, MSE = ', num2str(inpaint_mse(i, m), '%.6f')]);

            sgtitle(['Inpainting: ', test_set(i).name]);
        end
    end
end

inpaint_table = array2table(inpaint_mse, ...
    'VariableNames', compose('Missing_%d_percent', missing_percentages), ...
    'RowNames', {test_set.name});

disp(' ');
disp('Inpainting MSE table:');
disp(inpaint_table);

figure;
bar(inpaint_mse);
legend(compose('%d%% missing', missing_percentages), 'Location', 'best');
set(gca, 'XTickLabel', {test_set.name});
xtickangle(45);
ylabel('MSE');
title('Inpainting MSE per image and missing percentage');
grid on;

% Local functions

function img = readGrayImage(image_path)
    img = imread(image_path);

    if ndims(img) == 3
        img = rgb2gray(img);
    end

    img = im2double(img);
end

function img = cropToPatchMultiple(img, patch_size)
    [H, W] = size(img);

    H_new = floor(H / patch_size) * patch_size;
    W_new = floor(W / patch_size) * patch_size;

    if H_new < patch_size || W_new < patch_size
        error('Η εικόνα είναι μικρότερη από ένα patch.');
    end

    img = img(1:H_new, 1:W_new);
end

function noisy = addGaussianNoiseSNR(original, snr_db)
    signal_power = mean(original(:).^2);
    noise_power = signal_power / (10^(snr_db / 10));
    noise_std = sqrt(noise_power);

    noise = noise_std * randn(size(original));

    noisy = original + noise;
    noisy = min(max(noisy, 0), 1);
end

function [missing_image, mask] = createMissingPixels(original, missing_percentage)
    missing_rate = missing_percentage / 100;

    random_values = rand(size(original));

    mask = double(random_values > missing_rate);

    missing_image = original .* mask;
end

function reconstructed_image = dictionaryReconstruct(image, D, patch_size, T0, err)
    [H, W] = size(image);

    Y_img = imageToPatches(image, patch_size);

    patch_norms = sqrt(sum(Y_img.^2, 1));
    patch_norms(patch_norms == 0) = 1;

    Y_norm = Y_img ./ patch_norms;

    K = size(D, 2);
    N = size(Y_norm, 2);

    X_img = zeros(K, N);

    for i = 1:N
        X_img(:, i) = GenOMP(D, Y_norm(:, i), T0, err);
    end

    Y_reconstructed = D * X_img;

    Y_reconstructed = Y_reconstructed .* patch_norms;

    reconstructed_image = patchesToImage(Y_reconstructed, H, W, patch_size);
    reconstructed_image = min(max(reconstructed_image, 0), 1);
end

function Y_img = imageToPatches(img, patch_size)
    [H, W] = size(img);

    num_patches = (H / patch_size) * (W / patch_size);

    Y_img = zeros(patch_size^2, num_patches);

    patch_cnt = 1;

    for row = 1:patch_size:H
        for col = 1:patch_size:W

            patch = img(row:row+patch_size-1, col:col+patch_size-1);

            Y_img(:, patch_cnt) = reshape(patch, [patch_size^2, 1]);

            patch_cnt = patch_cnt + 1;
        end
    end
end

function img = patchesToImage(Y_img, H, W, patch_size)
    img = zeros(H, W);

    patch_cnt = 1;

    for row = 1:patch_size:H
        for col = 1:patch_size:W

            patch = reshape(Y_img(:, patch_cnt), [patch_size, patch_size]);

            img(row:row+patch_size-1, col:col+patch_size-1) = patch;

            patch_cnt = patch_cnt + 1;
        end
    end
end

function A = normalizeColumns(A)
    norms = sqrt(sum(A.^2, 1));
    norms(norms == 0) = 1;

    A = A ./ norms;
end