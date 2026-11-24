clear; clc; close all;

v = VideoReader('sevenup.mp4');
frame = readFrame(v);

%select the bounding box on a figure
figure;
imshow(frame);
title('Select initial bounding box around the object');
rect = getrect;   % to draw the rectangle

W = rect(3);
H = rect(4);
cx = rect(1) + W/2;
cy = rect(2) + H/2;

N = 400; %particles
sigma_init = 4;
sigma_motion = 2;
a = 0.18; %weight update constant
paddingOverBBox = 30;

particles = zeros(N, 2);
particles(:,1) = cx + sigma_init*randn(N,1);
particles(:,2) = cy + sigma_init*randn(N,1);

fileName = 'result1';
out = VideoWriter('result1.mp4', 'MPEG-4');
out.FrameRate = v.FrameRate;
open(out);

frameIndex = 1;
pointPerSide = 40;

prev_estimate = [cx, cy];
while true
    gray = rgb2gray(frame);
    
    % παράθυρο γύρω από την προηγούμενη εκτίμηση
    x1 = round(prev_estimate(1) - W/2 - paddingOverBBox);
    x2 = round(prev_estimate(1) + W/2 + paddingOverBBox);
    y1 = round(prev_estimate(2) - H/2 - paddingOverBBox);
    y2 = round(prev_estimate(2) + H/2 + paddingOverBBox);
    
    % clamp μέσα στα όρια της εικόνας
    x1 = max(1, x1);
    x2 = min(size(gray, 2), x2);
    y1 = max(1, y1);
    y2 = min(size(gray, 1), y2);
    
    % edge detector μόνο στο local παράθυρο
    edgesFull = false(size(gray));
    roi = gray(y1:y2, x1:x2);
    edgesROI = edge(roi, 'canny');
    edgesFull(y1:y2, x1:x2) = edgesROI;
    D = bwdist(edgesFull);

    %particles move randomly with motion
    if frameIndex > 1
        particles(:,1) = particles(:,1) + sigma_motion * randn(N,1);
        particles(:,2) = particles(:,2) + sigma_motion * randn(N,1);
    end

    %bounding the particles inside the frame
    particles(:,1) = min(max(particles(:,1), 1), size(frame,2));
    particles(:,2) = min(max(particles(:,2), 1), size(frame,1));

    % calculate likelihoods
    weights = zeros(N, 1);

    for i = 1:N
        px = particles(i, 1);
        py = particles(i, 2);
    
        % Get the 4 sides of the candidate rectangle separately
        [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(px, py, W, H, pointPerSide);
    
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
    estimate = sum(particles.*weights, 1);
    est_x = estimate(1);
    est_y = estimate(2);
    prev_estimate = estimate;

    %resample based on updated center
    indices = systematicResampling(weights);
    particles = particles(indices,:);

    weights = ones(N, 1)/N;

    % Create frame for saving
    bbox = [est_x - W/2, est_y - H/2, W, H];
    
    outputFrame = frame;
    outputFrame = insertShape(outputFrame, 'Rectangle', bbox, ...
        'Color', 'green', 'LineWidth', 2);
    
    outputFrame = insertMarker(outputFrame, [est_x, est_y], ...
        'o', 'Color', 'blue', 'Size', 1);
    
    outputFrame = insertMarker(outputFrame, particles, ...
    'circle', 'Color', 'red', 'Size', 1); %draw particles
    
    imshow(outputFrame);
    title(sprintf('Frame %d', frameIndex));
    drawnow;

    writeVideo(out, outputFrame);
    if hasFrame(v)
        frame = readFrame(v);
        frameIndex = frameIndex + 1;
    else
        break;
    end
end

close(out);
disp('Done. Output saved as result1.mp4');

function [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(cx, cy, W, H, n)

    x1 = cx - W/2;
    x2 = cx + W/2;
    y1 = cy - H/2;
    y2 = cy + H/2;

    topPts    = [linspace(x1, x2, n)', y1 * ones(n,1)];
    bottomPts = [linspace(x1, x2, n)', y2 * ones(n,1)];
    leftPts   = [x1 * ones(n,1), linspace(y1, y2, n)'];
    rightPts  = [x2 * ones(n,1), linspace(y1, y2, n)'];

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

    d = median(distances);

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

