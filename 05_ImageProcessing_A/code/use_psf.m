clear; clc; close;

addpath("./images/Ασκηση 4")
X = im2single(imread("new_york.png"));
Y = psf(X);
save("result.mat", "Y")