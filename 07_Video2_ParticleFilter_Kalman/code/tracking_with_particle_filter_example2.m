%-----------------------------------------------------------------
%  Example 15.4 - Visual Tracking
%  Particle filter for tracking a circle moving with random walk
%-----------------------------------------------------------------

function tracking_with_particle_filter_example2

clear; close all;

DIM    = 300;   % image size (DIM x DIM)
CYCLES = 120;   % number of cycles
r      = 10;    % radius of the circle to track
C      = 25;     % noise covariance coefficient

% Initial position at image center
true_position = [floor(DIM/2), floor(DIM/2)];

% Noise model parameters
M = [0 0]';
V = C * [5 0.5; 0.5 5];

N = 50; % number of particles

% Initialize particles with Gaussian noise around true position
particle = mvnrnd(M', V, N) + repmat(true_position, N, 1);

% Uniform initial weights
w = ones(1, N) / N;

% ---- Figure setup ----
fig = figure('Name', 'Particle Filter Tracker', ...
             'Color', 'k', ...
             'Position', [100 100 500 550]);

for k = 1:CYCLES

    % --- Move true position (random walk, ±5 per axis) ---
    eta = floor(10 * rand(2,1) - 5);
    true_position(1) = eta(1) + true_position(1);
    true_position(2) = eta(2) + true_position(2);

    % Clamp to image bounds
    true_position(1) = max(r+1, min(DIM-r-1, true_position(1)));
    true_position(2) = max(r+1, min(DIM-r-1, true_position(2)));

    % --- Build and noise the target image ---
    target_image = plotCircle(r, true_position, DIM);
    target_image = imnoise(target_image, 'salt & pepper', 0.04);
    D = bwdist(target_image);

    % --- Compute observation likelihoods ---
    w = calculateObservationLikelihood(r, D, particle);

    if sum(w) == 0
        disp('Weights collapsed — reinitializing particles.');
        particle = mvnrnd(M', V, N) + repmat(true_position, N, 1);
        w = ones(1, N) / N;
        continue;
    end
    w = w / sum(w);

    % --- Visualization ---
    clf;
    ax = axes('Parent', fig, ...
              'Color', 'k', ...
              'XColor', 'w', 'YColor', 'w', ...
              'Position', [0.05 0.12 0.90 0.82]);
    hold(ax, 'on');

    % Show the noisy binary image as background
    imshow(target_image, 'Parent', ax);
    hold(ax, 'on');

    % Draw particles — red dashed circles
    viscircles(ax, particle, r * ones(N,1), ...
               'Color', 'r', ...
               'LineStyle', '--', ...
               'LineWidth', 0.8);

    % Draw true circle — solid blue
    viscircles(ax, true_position, r, ...
               'Color', 'b', ...
               'LineStyle', '-', ...
               'LineWidth', 2);

    % Compute and draw estimated position (weighted mean)
    est_pos = sum(particle .* repmat(w', 1, 2), 1);
    viscircles(ax, est_pos, r, ...
               'Color', 'g', ...
               'LineStyle', ':', ...
               'LineWidth', 1.5);

    axis(ax, 'equal', 'tight');
    xlim(ax, [1 DIM]);
    ylim(ax, [1 DIM]);

    title(ax, sprintf('Cycle %d / %d', k, CYCLES), ...
          'Color', 'w', 'FontSize', 11);

    legend(ax, ...
        [plot(NaN,NaN,'r--'), plot(NaN,NaN,'b-'), plot(NaN,NaN,'g:')], ...
        {'Particles','True position','Estimate'}, ...
        'TextColor','w','Color','k','Location','southoutside', ...
        'Orientation','horizontal','FontSize',9);

    drawnow;

    % --- Resample ---
    y = zeros(N, 2);
    for c = 1:N
        s      = performSampling(w);
        parent = [particle(s,1); particle(s,2)];
        y(c,:) = (parent + mvnrnd(M', V, 1)')';
    end
    particle = y;

    pause(0.1);
end

end


% ----------------------------------------------------------------
% Plot a white circle ring on a black DIM×DIM binary image
% ----------------------------------------------------------------
function outimage = plotCircle(r, position, DIM)
    t = 0:0.05:2*pi;
    i = round(r * cos(t) + position(1));
    j = round(r * sin(t) + position(2));

    outimage = zeros(DIM, DIM);
    valid = (i > 0 & i <= DIM & j > 0 & j <= DIM);
    i = i(valid);
    j = j(valid);

    for c = 1:length(i)
        outimage(i(c), j(c)) = 1;
    end
end


% ----------------------------------------------------------------
% Observation likelihood for all particles
%   For each particle, overlay its circle on the distance transform
%   and sum exp(-a * D) at the circle pixels
% ----------------------------------------------------------------
function out = calculateObservationLikelihood(r, D, particles)
    a   = 2;
    N   = size(particles, 1);
    out = zeros(1, N);
    DIM = size(D, 1);

    for i = 1:N
        mask     = plotCircle(r, particles(i,:), DIM);
        f        = exp(-a * D);
        out(i)   = sum(sum(mask .* f));
    end
end


% ----------------------------------------------------------------
% Weighted sampling (inverse CDF / roulette wheel)
% ----------------------------------------------------------------
function out = performSampling(f)
    x   = rand;
    acc = 0;

    for i = 1:length(f)
        acc = acc + f(i);
        if acc > x
            out = i;
            return;
        end
    end
    out = length(f); % fallback
end