function res = videoStabilization1(invideo, outvideo)

vidObj = VideoReader(invideo);
writerObj = VideoWriter(outvideo, 'MPEG-4');
open(writerObj);

windowSize = 25;
thres = 0.5;

% Πρώτο frame
prevFrame = readFrame(vidObj);
prevGray = im2double(rgb2gray(prevFrame));

[sz1, sz2] = size(prevGray);
[X, Y] = meshgrid(1:sz2, 1:sz1);

% Γράφω το πρώτο frame όπως είναι
writeVideo(writerObj, prevFrame);

u_tot = zeros(sz1, sz2);
v_tot = zeros(sz1, sz2);

while hasFrame(vidObj)

    currFrame = readFrame(vidObj);
    currGray = im2double(rgb2gray(currFrame));

    [u, v] = calculate_of(prevGray, currGray, windowSize, thres);
    u_tot = u_tot + u;
    v_tot = v_tot + v;
    currFrameDouble = im2double(currFrame);
    stabilizedRGB = zeros(size(currFrameDouble));
    for c = 1:3
        stabilizedRGB(:,:,c) = interp2( ...
            X, Y, currFrameDouble(:,:,c), ...
            X + u_tot, Y + v_tot, ...
            'linear', 0);
    end
    writeVideo(writerObj, stabilizedRGB);
    prevGray = currGray;
end

close(writerObj);
res = 1;

end