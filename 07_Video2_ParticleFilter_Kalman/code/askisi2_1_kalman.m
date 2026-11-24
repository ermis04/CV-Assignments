clear; clc; close all;

v = VideoReader('sevenup.mp4');
frame = readFrame(v);

figure;
imshow(frame);
title('Select initial bounding box around the object');
rect = getrect;

W = rect(3);
Hbox = rect(4);
cx = rect(1) + W/2;
cy = rect(2) + Hbox/2;

pointPerSide = 40;
paddingOverBBox = 30;

dt = 1;
% state: [x; y; vx; vy]
x = [cx; cy; 0; 0];
P = diag([25 25 100 100]);

A = [1 0 dt 0;
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];

% Measurement is only the observed bbox center [x; y]
C = [1 0 0 0;
     0 1 0 0];

% Larger Q = follows fast motion more; larger R = trusts detector less.
Q = diag([1 1 6 6]);
Rk = diag([4^2 4^2]);ç

searchRadius = 25;    % pixels around Kalman prediction for measurement search
searchStep = 2;       % smaller = slower but more accurate
maxAcceptedDistance = 20;  % reject very bad edge matches

out = VideoWriter('result1_kalman.mp4', 'MPEG-4');
out.FrameRate = v.FrameRate;
open(out);

frameIndex = 1;
while true
    gray = rgb2gray(frame);

    %predict
    xPred = A * x;
    PPred = A * P * A' + Q;
    predCenter = xPred(1:2)';

    % Edge detector in local window around Kalman prediction
    x1 = round(predCenter(1) - W/2 - paddingOverBBox - searchRadius);
    x2 = round(predCenter(1) + W/2 + paddingOverBBox + searchRadius);
    y1 = round(predCenter(2) - Hbox/2 - paddingOverBBox - searchRadius);
    y2 = round(predCenter(2) + Hbox/2 + paddingOverBBox + searchRadius);

    x1 = max(1, x1); x2 = min(size(gray,2), x2);
    y1 = max(1, y1); y2 = min(size(gray,1), y2);

    edgesFull = false(size(gray));
    roi = gray(y1:y2, x1:x2);
    edgesROI = edge(roi, 'canny');
    edgesFull(y1:y2, x1:x2) = edgesROI;
    D = bwdist(edgesFull);

    [z, bestD, found] = detectCenterByEdges(D, predCenter, W, Hbox, ...
        searchRadius, searchStep, pointPerSide, maxAcceptedDistance);

    if found
        innovation = z(:) - C * xPred;
        S = C * PPred * C' + Rk;
        K = PPred * C' / S;
        x = xPred + K * innovation;
        P = (eye(4) - K * C) * PPred;
    else
        x = xPred;
        P = PPred;
    end

    % Keep state inside image
    x(1) = min(max(x(1), 1), size(frame,2));
    x(2) = min(max(x(2), 1), size(frame,1));

    est_x = x(1);
    est_y = x(2);
    bbox = [est_x - W/2, est_y - Hbox/2, W, Hbox];

    outputFrame = frame;
    outputFrame = insertShape(outputFrame, 'Rectangle', bbox, ...
        'Color', 'green', 'LineWidth', 2);
    outputFrame = insertMarker(outputFrame, [est_x, est_y], ...
        'o', 'Color', 'blue', 'Size', 3);

    %show raw measurement in magenta
    if found
        outputFrame = insertMarker(outputFrame, z(:)', ...
            'x', 'Color', 'magenta', 'Size', 5);
    end

    imshow(outputFrame);
    title(sprintf('Kalman Frame %d, edge distance %.2f', frameIndex, bestD));
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
disp('Done. Output saved as result1_kalman.mp4');

function [z, bestD, found] = detectCenterByEdges(D, predCenter, W, Hbox, radius, step, n, maxAcceptedDistance)
    xs = (predCenter(1)-radius):step:(predCenter(1)+radius);
    ys = (predCenter(2)-radius):step:(predCenter(2)+radius);

    bestD = inf;
    z = predCenter(:);

    for px = xs
        for py = ys
            [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(px, py, W, Hbox, n);

            dTop    = sideDistance(topPts, D);
            dBottom = sideDistance(bottomPts, D);
            dLeft   = sideDistance(leftPts, D);
            dRight  = sideDistance(rightPts, D);

            if isinf(dTop) || isinf(dBottom) || isinf(dLeft) || isinf(dRight)
                continue;
            end

            d = mean([dTop, dBottom, dLeft, dRight]);
            if d < bestD
                bestD = d;
                z = [px; py];
            end
        end
    end

    found = isfinite(bestD) && bestD <= maxAcceptedDistance;
end

function [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(cx, cy, W, Hbox, n)
    x1 = cx - W/2;
    x2 = cx + W/2;
    y1 = cy - Hbox/2;
    y2 = cy + Hbox/2;

    topPts    = [linspace(x1, x2, n)', y1 * ones(n,1)];
    bottomPts = [linspace(x1, x2, n)', y2 * ones(n,1)];
    leftPts   = [x1 * ones(n,1), linspace(y1, y2, n)'];
    rightPts  = [x2 * ones(n,1), linspace(y1, y2, n)'];
end

function d = sideDistance(pts, D)
    xs = round(pts(:,1));
    ys = round(pts(:,2));

    valid = xs >= 1 & xs <= size(D,2) & ys >= 1 & ys <= size(D,1);

    if sum(valid) < 0.8 * size(pts,1)
        d = inf;
        return;
    end

    xs = xs(valid);
    ys = ys(valid);

    idx = sub2ind(size(D), ys, xs);
    distances = D(idx);
    d = median(distances);
end
