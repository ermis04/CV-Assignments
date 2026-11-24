clear; clc;

script_folder = pwd;

men_folder = fullfile(script_folder, 'men');
women_folder = fullfile(script_folder, 'women');

img_size = [16 16];

rng(1);
men = read_jpg_images(men_folder, img_size);

rng(2);
women = read_jpg_images(women_folder, img_size);

if size(men, 1) < 1000 || size(women, 1) < 1000
    error('Both folders must contain at least 1000 jpg images.');
end

men = men(randperm(size(men, 1)), :);
women = women(randperm(size(women, 1)), :);

men = men(1:1000, :);
women = women(1:1000, :);

men_train = men(1:800, :);
men_test  = men(801:1000, :);

women_train = women(1:800, :);
women_test  = women(801:1000, :);

% z = 1 for men
% z = 0 for women

mu1 = mean(men_train, 1);
mu0 = mean(women_train, 1);

lambda = 1e-2;
Sigma1 = cov(men_train) + lambda * eye(size(men_train, 2));
Sigma0 = cov(women_train) + lambda * eye(size(women_train, 2));

prior1 = 800 / 1600;
prior0 = 800 / 1600;

Xtest = [men_test; women_test];
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

fprintf('Accuracy: %.2f%%\n', accuracy * 100);

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
