function [i,j] = cornerDetection(im, windowSize, thres)

Ix = conv2(im, [-1, 1], 'same');
Iy = conv2(im, [-1; 1], 'same');

%φίλτρο
Ix = conv2(Ix, ones(3, 3)/9, 'same');
Iy = conv2(Iy, ones(3, 3)/9, 'same');

Ixx = Ix.^2;
Iyy = Iy.^2;
Ixy = Ix.*Iy;

weight = ones(windowSize, windowSize);


M11 = conv2(Ixx, weight, 'same');
M22 = conv2(Iyy, weight, 'same');
M12 = conv2(Ixy, weight, 'same');        

Det = M11.*M22 -M12.^2;
lambda_min = ((M11+M22) - sqrt((M11+M22).^2 - 4*Det))/2;
mask = lambda_min >= thres;
[i, j] = find(mask);

end