clear; clc;

script_folder = pwd;

faces_folder = fullfile(script_folder, 'faces');
not_faces_folder = fullfile(script_folder, 'not_faces');

img_size = [16 16];

rng(1);
faces = read_jpg_images(faces_folder, img_size);

rng(2);
not_faces = read_jpg_images(not_faces_folder, img_size);

fprintf('Faces found: %d\n', size(faces, 1));
fprintf('Non-faces found: %d\n', size(not_faces, 1));

if size(faces, 1) < 1000 || size(not_faces, 1) < 1000
    error('Both folders must contain at least 1000 jpg images. Check faces_folder and not_faces_folder.');
end

faces = faces(randperm(size(faces, 1)), :);
not_faces = not_faces(randperm(size(not_faces, 1)), :);

faces = faces(1:1000, :);
not_faces = not_faces(1:1000, :);

faces_train = faces(1:800, :);
faces_test  = faces(801:1000, :);

nonfaces_train = not_faces(1:800, :);
nonfaces_test  = not_faces(801:1000, :);

% z = 1 για faces
% z = 0 για nonfaces

mu1 = mean(faces_train, 1);
mu0 = mean(nonfaces_train, 1);

lambda = 1e-2;


Sigma1 = cov(faces_train) + lambda * eye(size(faces_train, 2));
Sigma0 = cov(nonfaces_train) + lambda * eye(size(nonfaces_train, 2));

prior1 = 800 / 1600;
prior0 = 800 / 1600;

Xtest = [faces_test; nonfaces_test];
ytrue = [ones(200, 1); zeros(200, 1)];

ypred = zeros(400, 1);

for i = 1:400
    x = Xtest(i, :);

    score1 = log_gaussian(x, mu1, Sigma1) + log(prior1);
    score0 = log_gaussian(x, mu0, Sigma0) + log(prior0);

    if score1 > score0
        ypred(i) = 1;
    else
        ypred(i) = 0;
    end
end

accuracy = sum(ypred == ytrue) / length(ytrue);

fprintf("Accuracy: %.2f%%\n", accuracy * 100);

confusionchart(ytrue, ypred);


function X = read_jpg_images(folder, img_size)

    files = [dir(fullfile(folder, '*.jpg')); dir(fullfile(folder, '*.JPG'))];
    [~, unique_idx] = unique(fullfile({files.folder}, {files.name}));
    files = files(unique_idx);
    [~, order] = sort({files.name});
    files = files(order);

    X = zeros(length(files), prod(img_size));

    for i = 1:length(files)

        image_path = fullfile(folder, files(i).name);

        img = imread(image_path);

        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        img = imresize(img, img_size);
        img = im2double(img);

        X(i, :) = img(:)';
    end
end


function logp = log_gaussian(x, mu, Sigma)

    d = length(x);
    diff = x - mu;
    R = chol(Sigma);

    log_det = 2 * sum(log(diag(R)));
    z = diff / R;
    mahalanobis = z * z';

    logp = -0.5 * (d * log(2*pi) + log_det + mahalanobis);
end
