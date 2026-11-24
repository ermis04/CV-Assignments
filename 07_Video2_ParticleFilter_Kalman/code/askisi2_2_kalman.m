clear; clc; close all;

files = dir(fullfile('Coke/img', '*.jpg'));
firstImg = imread(fullfile('Coke/img', files(1).name));

figure;
imshow(firstImg);
title(files(1).name, 'Interpreter', 'none');
rect = getrect;

W = rect(3);
Hbox = rect(4);
cx = rect(1) + W/2;
cy = rect(2) + Hbox/2;
theta0 = 0;

pointPerSide = 50;
paddingOverBBox = 50;

dt = 1;
x = [cx; cy; theta0; 0; 0; 0];
P = diag([25 25 deg2rad(8)^2 100 100 deg2rad(10)^2]);

A = [1 0 0 dt 0  0;
     0 1 0 0  dt 0;
     0 0 1 0  0  dt;
     0 0 0 1  0  0;
     0 0 0 0  1  0;
     0 0 0 0  0  1];

% Measurement is observed pose [x; y; theta]
C = [1 0 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 0 0 0];

Q = diag([2 2 deg2rad(1.5)^2 8 8 deg2rad(2)^2]);
Rk = diag([8^2 8^2 deg2rad(6)^2]);

searchRadius = 18;
searchStep = 3;
thetaRadius = deg2rad(15);
thetaStep = deg2rad(3);
maxAcceptedDistance = 25;

out = VideoWriter('result2_kalman.mp4', 'MPEG-4');
out.FrameRate = 24;
open(out);

for k = 2:length(files)
    frame = imread(fullfile('Coke/img', files(k).name));
    gray = rgb2gray(frame);

    %prediction
    xPred = A * x;
    xPred(3) = angleWrap(xPred(3));
    PPred = A * P * A' + Q;

    pred_x = xPred(1);
    pred_y = xPred(2);
    pred_theta = xPred(3);

    % Rotated search window around predicted pose
    searchW = W + 2 * (paddingOverBBox + searchRadius);
    searchH = Hbox + 2 * (paddingOverBBox + searchRadius);
    searchCorners = rotatedRectangleCorners(pred_x, pred_y, searchW, searchH, pred_theta);

    searchMask = poly2mask(searchCorners(:,1), searchCorners(:,2), size(gray,1), size(gray,2));

    x1 = round(min(searchCorners(:,1))); x2 = round(max(searchCorners(:,1)));
    y1 = round(min(searchCorners(:,2))); y2 = round(max(searchCorners(:,2)));

    x1 = max(1, x1); x2 = min(size(gray,2), x2);
    y1 = max(1, y1); y2 = min(size(gray,1), y2);

    edgesFull = false(size(gray));
    roi = gray(y1:y2, x1:x2);
    edgesROI = edge(roi, 'canny');
    edgesFull(y1:y2, x1:x2) = edgesROI;
    edgesFull = edgesFull & searchMask;
    D = bwdist(edgesFull);

    % Measurement from image: best rotated rectangle pose according to edge distance
    [z, bestD, found] = detectPoseByEdges(D, [pred_x; pred_y; pred_theta], ...
        W, Hbox, searchRadius, searchStep, thetaRadius, thetaStep, ...
        pointPerSide, maxAcceptedDistance);

    %update the 
    if found
        innovation = z(:) - C * xPred;
        innovation(3) = angleWrap(innovation(3));

        S = C * PPred * C' + Rk;
        K = PPred * C' / S;
        x = xPred + K * innovation;
        P = (eye(6) - K * C) * PPred;
        x(3) = angleWrap(x(3));
    else
        x = xPred;
        P = PPred;
    end

    x(1) = min(max(x(1), 1), size(frame,2));
    x(2) = min(max(x(2), 1), size(frame,1));
    x(3) = angleWrap(x(3));

    est_x = x(1);
    est_y = x(2);
    est_theta = x(3);

    corners = rotatedRectangleCorners(est_x, est_y, W, Hbox, est_theta);
    poly = reshape(corners', 1, []);
    searchPoly = reshape(searchCorners', 1, []);

    outputFrame = frame;
    outputFrame = insertShape(outputFrame, 'Polygon', searchPoly, ...
        'Color', 'yellow', 'LineWidth', 2);
    outputFrame = insertShape(outputFrame, 'Polygon', poly, ...
        'Color', 'green', 'LineWidth', 2);
    outputFrame = insertMarker(outputFrame, [est_x, est_y], ...
        'o', 'Color', 'blue', 'Size', 3);

    %show raw measurement in magenta
    if found
        measCorners = rotatedRectangleCorners(z(1), z(2), W, Hbox, z(3));
        measPoly = reshape(measCorners', 1, []);
        outputFrame = insertShape(outputFrame, 'Polygon', measPoly, ...
            'Color', 'magenta', 'LineWidth', 1);
    end

    imshow(outputFrame);
    title(sprintf('Kalman Frame %d, edge distance %.2f', k, bestD));
    drawnow;
    writeVideo(out, outputFrame);
end

close(out);
disp('Done. Output saved as result2_kalman.mp4');

function [z, bestD, found] = detectPoseByEdges(D, predPose, W, Hbox, radius, step, thetaRadius, thetaStep, n, maxAcceptedDistance)
    xs = (predPose(1)-radius):step:(predPose(1)+radius);
    ys = (predPose(2)-radius):step:(predPose(2)+radius);
    thetas = (predPose(3)-thetaRadius):thetaStep:(predPose(3)+thetaRadius);

    bestD = inf;
    z = predPose(:);

    for px = xs
        for py = ys
            for theta = thetas
                theta = angleWrap(theta);
                [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(px, py, W, Hbox, theta, n);

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
                    z = [px; py; theta];
                end
            end
        end
    end

    found = isfinite(bestD) && bestD <= maxAcceptedDistance;
end

function [topPts, bottomPts, leftPts, rightPts] = rectangleFourSides(cx, cy, W, Hbox, theta, n)
    corners = rotatedRectangleCorners(cx, cy, W, Hbox, theta);

    c1 = corners(1,:);
    c2 = corners(2,:);
    c3 = corners(3,:);
    c4 = corners(4,:);

    topPts = [linspace(c1(1), c2(1), n)', linspace(c1(2), c2(2), n)'];
    rightPts = [linspace(c2(1), c3(1), n)', linspace(c2(2), c3(2), n)'];
    bottomPts = [linspace(c3(1), c4(1), n)', linspace(c3(2), c4(2), n)'];
    leftPts = [linspace(c4(1), c1(1), n)', linspace(c4(2), c1(2), n)'];
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
    d = mean(distances);
end

function corners = rotatedRectangleCorners(cx, cy, W, Hbox, theta)
    localCorners = [
        -W/2, -Hbox/2;
         W/2, -Hbox/2;
         W/2,  Hbox/2;
        -W/2,  Hbox/2
    ];

    Rot = [cos(theta), -sin(theta);
           sin(theta),  cos(theta)];

    corners = localCorners * Rot';
    corners(:,1) = corners(:,1) + cx;
    corners(:,2) = corners(:,2) + cy;
end

function a = angleWrap(a)
    a = atan2(sin(a), cos(a));
end
