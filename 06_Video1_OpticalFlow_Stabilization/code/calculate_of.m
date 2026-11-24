
function [u, v] = calculate_of(I1,I2,window, thres)


Ix = conv2(I1, [-1, 1], 'same');
Iy = conv2(I1, [-1; 1], 'same');
It = I2-I1;

%φίλτρο
Ix = conv2(Ix, ones(3, 3)/9, 'same');
Iy = conv2(Iy, ones(3, 3)/9, 'same');
It = conv2(It, ones(3, 3)/9, 'same');

Ixx = Ix.^2;
Iyy = Iy.^2;
Ixy = Ix.*Iy;

Ixt = Ix.*It;
Iyt = Iy.*It;

weight = ones(window, window);


M11 = conv2(Ixx, weight, 'same');
M22 = conv2(Iyy, weight, 'same');
M12 = conv2(Ixy, weight, 'same');

b1 = conv2(Ixt, -weight, 'same');
b2 = -conv2(Iyt, weight, 'same');
        

Det = M11.*M22 -M12.^2;
u = (M22.*b1 -M12.*b2)./Det;
v = (-M12.*b1 +M11.*b2)./Det;

lambda_min = ((M11+M22) - sqrt((M11+M22).^2 - 4*Det))/2;
mask = lambda_min >= thres;
u = u .* mask;
v = v .* mask;

end