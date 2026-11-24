clear; clc; close all;

%% ==========================
%  PARTICLE FILTER TRACKING
%  seven_up.avi
%  State: [x, y] = center of bounding box
%  Box size: fixed W, H
%  Likelihood: edge distance using bwdist
% ===========================

videoFile = 'seven_up.avi';

v = VideoReader(videoFile);

% Read first frame
frame = readFrame(v);

figure;
imshow(frame);
title('Select initial bounding box around the object');
rect = getrect;   % [left, top, width, height]

% Initial bounding box parameters
W = rect(3);
H = rect(4);

cx = rect(1) + W/2;
cy = rect(2) + H/2;

% Number of particles
N = 300;

% Parameters
sigma_init = 8;       % αρχική διασπορά particles γύρω από το αρχικό κουτί
sigma_motion = 10;    % θόρυβος κίνησης
a = 0.15;             % likelihood parameter: exp(-a*d)

% Number of sample points per side of rectangle
pointsPerSide = 40;

% Initialize particles around initial center
particles = zeros(N, 2);
particles(:,1) = cx + sigma_init * randn(N,1);
particles(:,2) = cy + sigma_init * randn(N,1);

weights = ones(N,1) / N;

% Video output
out = VideoWriter('seven_up_particle_result.avi');
open(out);

% For displaying
figure;

frameIndex = 1;

while true

    % Convert to grayscale
    if size(frame,3) == 3
        gray = rgb2gray(frame);
    else
        gray = frame;
    end

    % Edge detection
    edges = edge(gray, 'canny');

    % Distance transform:
    % D(y,x) = distance of pixel (x,y) from nearest edge
    D = bwdist(edges);

    %% ====================================
    %  1. PREDICT: move particles randomly
    % =====================================
    if frameIndex > 1
        particles(:,1) = particles(:,1) + sigma_motion * randn(N,1);
        particles(:,2) = particles(:,2) + sigma_motion * randn(N,1);
    end

    % Keep particles inside image bounds
    particles(:,1) = min(max(particles(:,1), 1), size(frame,2));
    particles(:,2) = min(max(particles(:,2), 1), size(frame,1));

    %% ====================================
    %  2. UPDATE: compute likelihood weights
    % =====================================
    weights = zeros(N,1);

    for i = 1:N
        px = particles(i,1);
        py = particles(i,2);

        % Sample points on rectangle boundary
        pts = rectangleBoundaryPoints(px, py, W, H, pointsPerSide);

        % Convert to pixel coordinates
        xs = round(pts(:,1));
        ys = round(pts(:,2));

        % Valid points inside image
        valid = xs >= 1 & xs <= size(D,2) & ys >= 1 & ys <= size(D,1);

        xs = xs(valid);
        ys = ys(valid);

        if isempty(xs)
            weights(i) = 0;
            continue;
        end

        % Get distances from nearest edges
        idx = sub2ind(size(D), ys, xs);
        distances = D(idx);

        % Mean distance of rectangle boundary from actual image edges
        d = mean(distances);

        % Likelihood
        weights(i) = exp(-a * d);
    end

    % Normalize weights
    s = sum(weights);

    if s == 0 || isnan(s)
        weights = ones(N,1) / N;
    else
        weights = weights / s;
    end

    %% ====================================
    %  3. ESTIMATE: weighted mean
    % =====================================
    estimate = sum(particles .* weights, 1);

    est_x = estimate(1);
    est_y = estimate(2);

    %% ====================================
    %  4. RESAMPLE
    % =====================================
    indices = systematicResampling(weights);
    particles = particles(indices,:);

    % After resampling, reset weights
    weights = ones(N,1) / N;

    %% ====================================
    %  5. DISPLAY RESULT
    % =====================================
    imshow(frame);
    hold on;

    % Draw particles
    plot(particles(:,1), particles(:,2), 'r.', 'MarkerSize', 6);

    % Draw estimated bounding box
    rectangle('Position', [est_x - W/2, est_y - H/2, W, H], ...
              'EdgeColor', 'g', ...
              'LineWidth', 2);

    % Draw center
    plot(est_x, est_y, 'bo', 'MarkerSize', 8, 'LineWidth', 2);

    title(sprintf('Frame %d', frameIndex));
    hold off;
    drawnow;

    % Save frame to output video
    resultFrame = getframe(gca);
    writeVideo(out, resultFrame);

    %% Read next frame
    if hasFrame(v)
        frame = readFrame(v);
        frameIndex = frameIndex + 1;
    else
        break;
    end
end

close(out);

disp('Done. Output saved as seven_up_particle_result.avi');


%% ============================================================
%  Function: sample points on non-rotated rectangle boundary
% ============================================================
function pts = rectangleBoundaryPoints(cx, cy, W, H, n)

    x1 = cx - W/2;
    x2 = cx + W/2;
    y1 = cy - H/2;
    y2 = cy + H/2;

    % Top side
    top = [linspace(x1, x2, n)', y1 * ones(n,1)];

    % Bottom side
    bottom = [linspace(x1, x2, n)', y2 * ones(n,1)];

    % Left side
    left = [x1 * ones(n,1), linspace(y1, y2, n)'];

    % Right side
    right = [x2 * ones(n,1), linspace(y1, y2, n)'];

    pts = [top; bottom; left; right];
end


%% ============================================================
%  Function: systematic resampling
%  Input: weights that sum to 1
%  Output: indices of selected particles
% ============================================================
function indices = systematicResampling(weights)

    N = length(weights);
    indices = zeros(N,1);

    edges = cumsum(weights);
    edges(end) = 1;  % avoid numerical issue

    u0 = rand / N;
    u = u0 + (0:N-1)' / N;

    j = 1;
    for i = 1:N
        while u(i) > edges(j)
            j = j + 1;
        end
        indices(i) = j;
    end
end