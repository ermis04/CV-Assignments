function res = videoStabilization2(invideo, outvideo)
 vidObj = VideoReader(invideo);
 writerObj = VideoWriter(outvideo, 'MPEG-4');
open(writerObj);

refFrame = readFrame(vidObj);
refGray = im2double(rgb2gray(refFrame));

[sz1, sz2] = size(refGray);
[X, Y] = meshgrid(1:sz2, 1:sz1);
window = 25;

writeVideo(writerObj, refFrame);

while hasFrame(vidObj)
    currFrame = readFrame(vidObj);
    currGray = im2double(rgb2gray(currFrame));

    [i_corners, j_corners] = cornerDetection(refGray, window, 0.05);
    x = j_corners;
    y = i_corners;
    N = length(x);
    
    [u, v] = calculate_of(refGray, currGray, window, 0.05);

    indices = sub2ind(size(u), y, x);
    u_sparse = u(indices);
    v_sparse = v(indices);
    
    x_prime = x + u_sparse;
    y_prime = y + v_sparse;
    
    A = [x, y, ones(N,1), zeros(N,3); 
         zeros(N,3), x, y, ones(N,1)];
    b = [x_prime; y_prime];

    p = A \ b;
    T_matrix = [p(1) p(4) 0;
                p(2) p(5) 0;
                p(3) p(6) 1];

    disp(T_matrix);
    Tform = affine2d(T_matrix);
    Tform_inv = invert(Tform);

    stabilizedFrame = imwarp(currFrame, Tform_inv, 'OutputView', imref2d(size(currFrame)));
    stabilizedFrameFirst = imwarp(refFrame, Tform_inv, 'OutputView', imref2d(size(currFrame)));
    writeVideo(writerObj, stabilizedFrame);


    
    refGray = im2double(rgb2gray(stabilizedFrameFirst));
end
close(writerObj);
res = 1;

end