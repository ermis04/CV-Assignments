function res = videoStabilization3(invideo, outvideo)
 vidObj = VideoReader(invideo);
 writerObj = VideoWriter(outvideo, 'MPEG-4');
open(writerObj);

refFrame = readFrame(vidObj);
refGray = im2double(rgb2gray(refFrame));

[sz1, sz2] = size(refGray);
[X, Y] = meshgrid(1:sz2, 1:sz1);
window = 25;

writeVideo(writerObj, refFrame);
i = 0;
while hasFrame(vidObj)
    if mod(i,100) == 0
    fprintf("frame %d ", i);
    end
    i = i + 1;

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

        matchedPts1 = [x, y];
        matchedPts2 = [x_prime, y_prime];
        
        try
            % Το σύμβολο ~ σημαίνει "πέτα τον έτοιμο πίνακα, θέλω ΜΟΝΟ το φίλτρο inlierIdx"
            [~, inlierIdx] = estimateGeometricTransform2D(matchedPts1, matchedPts2, 'affine');
            
            x = x(inlierIdx);
            y = y(inlierIdx);
            x_prime = x_prime(inlierIdx);
            y_prime = y_prime(inlierIdx);
            N = length(x);
        catch
            % Αν για κάποιο λόγο τα σημεία είναι πολύ λίγα, απλά προχωράει
        end
    
    A = [x, y, ones(N,1), zeros(N,3); 
         zeros(N,3), x, y, ones(N,1)];
    b = [x_prime; y_prime];

    p = A \ b;
    T_matrix = [p(1) p(4) 0;
                p(2) p(5) 0;
                p(3) p(6) 1];


    Tform = affine2d(T_matrix);
    Tform_inv = invert(Tform);

    stabilizedFrame = imwarp(currFrame, Tform_inv, 'OutputView', imref2d(size(currFrame)));
    writeVideo(writerObj, stabilizedFrame);
end
close(writerObj);
res = 1;

end