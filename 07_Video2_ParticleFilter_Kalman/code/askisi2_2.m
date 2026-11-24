clear; clc; close all;

files = dir(fullfile('Coke/img', '*.jpg'));
firstImg = imread(fullfile('Coke/img', files(1).name));

figure;
h = imshow(firstImg);
title(files(1).name, 'Interpreter', 'none');
rect = getrect;   % to draw the rectangle

W = rect(3);
H = rect(4);
cx = rect(1) + W/2;
cy = rect(2) + H/2;

N = 800;
sigma_init = 5;
sigma_motion = 30;
a = 0.1;
pointPerSide = 200;
paddingOverBBox = 150;

fileName = 'result2';
out = VideoWriter('result2.mp4', 'MPEG-4');
out.FrameRate = 24;
open(out);

theta0 = 0;  % αρχική γωνία σε radians

sigma_init_theta = deg2rad(5);
sigma_motion_theta = deg2rad(3);

particles = zeros(N, 3);
particles(:,1) = cx + sigma_init*randn(N,1);
particles(:,2) = cy + sigma_init*randn(N,1);
particles(:,3) = theta0 + sigma_init_theta*randn(N,1);

prev_estimate = [cx, cy, theta0];

firstFrame = true;

for k = 2:length(files)
    frame = imread(fullfile('Coke/img', files(k).name));

gray = rgb2gray(frame);
    
% Rotated search window around the previous estimate
searchW = W + 2 * paddingOverBBox;
searchH = H + 2 * paddingOverBBox;

searchCorners = rotatedRectangleCorners( ...
    prev_estimate(1), prev_estimate(2), ...
    searchW, searchH, prev_estimate(3));

% Mask of the rotated search window
searchMask = poly2mask( ...
    searchCorners(:,1), searchCorners(:,2), ...
    size(gray,1), size(gray,2));

% Axis-aligned bounding box only for computational efficiency
x1 = round(min(searchCorners(:,1)));
x2 = round(max(searchCorners(:,1)));
y1 = round(min(searchCorners(:,2)));
y2 = round(max(searchCorners(:,2)));

% Clamp inside image
x1 = max(1, x1);
x2 = min(size(gray, 2), x2);
y1 = max(1, y1);
y2 = min(size(gray, 1), y2);

% Run edge detector only in the local bounding area
edgesFull = false(size(gray));
roi = gray(y1:y2, x1:x2);
edgesROI = edge(roi, 'canny');
edgesFull(y1:y2, x1:x2) = edgesROI;

% Keep only edges inside the rotated search window
edgesFull = edgesFull & searchMask;

% Distance transform from edges inside rotated window only
D = bwdist(edgesFull);

    %particles move randomly with motion
    if firstFrame ~= true
        particles(:,1) = particles(:,1) + sigma_motion * randn(N,1);
        particles(:,2) = particles(:,2) + sigma_motion * randn(N,1);
        particles(:,3) = particles(:,3) + sigma_motion_theta * randn(N,1);

% κράτα τη γωνία στο [-pi, pi]
particles(:,3) = atan2(sin(particles(:,3)), cos(particles(:,3)));
    end

    %bounding the particles inside the frame
    particles(:,1) = min(max(particles(:,1), 1), size(frame,2));
    particles(:,2) = min(max(particles(:,2), 1), size(frame,1));

    % calculate likelihoods
    weights = zeros(N, 1);

    for i = 1:N
        px = particles(i, 1);
        py = particles(i, 2);
        theta = particles(i, 3);
    
        % Get the 4 sides of the candidate rectangle separately
        [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(px, py, W, H, theta, pointPerSide);
    
        % Distance of each side from the nearest image edges
        dTop    = sideDistance(topPts, D);
        dBottom = sideDistance(bottomPts, D);
        dLeft   = sideDistance(leftPts, D);
        dRight  = sideDistance(rightPts, D);
    
        % If any side is invalid/outside, reject this particle
        if isinf(dTop) || isinf(dBottom) || isinf(dLeft) || isinf(dRight)
            weights(i) = 0;
            continue;
        end
    
        % A particle is good only if all 4 sides are reasonably close to edges.
        d = mean([dTop, dBottom, dLeft, dRight]);
        weights(i) = exp(-a * d);
    end

    %normalization
    s = sum(weights);
    if s == 0 || isnan(s)
        weights = ones(N,1) / N;
    else
        weights = weights / s;
    end

    %estimate center of bunding box
    est_x = sum(particles(:,1) .* weights);
    est_y = sum(particles(:,2) .* weights);
    
    % circular mean για τη γωνία
    est_theta = atan2(sum(weights .* sin(particles(:,3))), ...
                      sum(weights .* cos(particles(:,3))));

prev_estimate = [est_x, est_y, est_theta];

    %resample based on updated center
    indices = systematicResampling(weights);
    particles = particles(indices,:);

    weights = ones(N, 1)/N;

    % Create frame for saving
    corners = rotatedRectangleCorners(est_x, est_y, W, H, est_theta);
    poly = reshape(corners', 1, []);
    
    outputFrame = frame;

searchPoly = reshape(searchCorners', 1, []);

outputFrame = insertShape(outputFrame, 'Polygon', searchPoly, ...
    'Color', 'yellow', 'LineWidth', 3);

    outputFrame = insertShape(outputFrame, 'Polygon', poly, ...
        'Color', 'green', 'LineWidth', 2);
    
    outputFrame = insertMarker(outputFrame, [est_x, est_y], ...
        'o', 'Color', 'blue', 'Size', 1);
    
    outputFrame = insertMarker(outputFrame, particles(:,1:2), ...
    'circle', 'Color', 'red', 'Size', 1);
    
    imshow(outputFrame);
    title(sprintf('Frame %d', k));
    drawnow;

    writeVideo(out, outputFrame);

    firstFrame = false;
end

close(out);
disp('Done. Output saved as result2.mp4');

function [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(cx, cy, W, H, theta, n)

    corners = rotatedRectangleCorners(cx, cy, W, H, theta);

    c1 = corners(1,:);
    c2 = corners(2,:);
    c3 = corners(3,:);
    c4 = corners(4,:);

    topPts = [linspace(c1(1), c2(1), n)', ...
              linspace(c1(2), c2(2), n)'];
    rightPts = [linspace(c2(1), c3(1), n)', ...
                linspace(c2(2), c3(2), n)'];
    bottomPts = [linspace(c3(1), c4(1), n)', ...
                 linspace(c3(2), c4(2), n)'];
    leftPts = [linspace(c4(1), c1(1), n)', ...
               linspace(c4(2), c1(2), n)'];
end


function d = sideDistance(pts, D)

    xs = round(pts(:,1));
    ys = round(pts(:,2));

    %διάνυσμα που κρατάει 0 ή 1 ανάλογα τη λογική τιμή μεγέθους xs και ys
    valid = xs >= 1 & xs <= size(D,2) & ys >= 1 & ys <= size(D,1);

    % If too much of this side is outside the image, reject it
    if sum(valid) < 0.8 * size(pts,1)
        d = inf;
        return;
    end

    %κρατάω μόνο valid σημεία
    xs = xs(valid);
    ys = ys(valid);

    idx = sub2ind(size(D), ys, xs);
    distances = D(idx);

    d = mean(distances);

end

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

function corners = rotatedRectangleCorners(cx, cy, W, H, theta)

    localCorners = [
        -W/2, -H/2;
         W/2, -H/2;
         W/2,  H/2;
        -W/2,  H/2
    ];

    R = [cos(theta), -sin(theta);
         sin(theta),  cos(theta)];

    corners = localCorners * R';
    corners(:,1) = corners(:,1) + cx;
    corners(:,2) = corners(:,2) + cy;
end