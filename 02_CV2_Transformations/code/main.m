clear
clc
close all

%% 2 - DIfferent Scales
ball = imread('ball.jpg');
imshow(ball)

bg_height = 2000;
bg_width = 4000;
bg = 255 * ones(bg_height, bg_width, 3, 'uint8');

scales = [0.2, 0.4, 0.6, 0.8, 1, 1.5];

current_x_position = 1;

for i = 1:length(scales)
    scale = scales(i);
    T_temp = [scale 0 0;
              0 scale 0;
                0 0 1];
    T = affine2d(T_temp);
    resizedBall = imwarp(ball, T);
    [new_height, new_width, ~] = size(resizedBall);

    y_start = 1;
    y_end = new_height;

    x_start = current_x_position;
    x_end = current_x_position + new_width - 1;

    if x_end <= bg_width && y_end <= bg_height
        bg(y_start:y_end, x_start:x_end, :) = resizedBall;
        current_x_position = x_end + 20;
    end
end

figure;
imshow(bg);

%% 3 - Shearing video

pudding = imread('pudding.png');
[height, width, ~] = size(pudding);

videoW = VideoWriter('sheared_pudding.avi', 'Motion JPEG AVI');
videoW.FrameRate = 30;
open(videoW);

frame_padding = 200;
xLimits = [-frame_padding , width+frame_padding ];
yLimits = [0, height];

ref = imref2d([height, width + 2*frame_padding], xLimits, yLimits);

duration = 4; % in seconds
frames = duration*30;
max_shear = 0.5;

for t = 1:frames
    sh = max_shear*sin(2*2*pi*t/frames);
    T = [1 0 0;
        sh 1 0;
        0 0 1];
    tform = affine2d(T);

    shearedImg = imwarp(pudding, tform, 'OutputView',ref);
    writeVideo(videoW, shearedImg);
end

close(videoW);
implay('sheared_pudding.avi');

%% 4 - Shearing video - stable base

pudding = imread('pudding.png');
[height, width, ~] = size(pudding);

videoW = VideoWriter('sheared_pudding.avi', 'Motion JPEG AVI');
videoW.FrameRate = 30;
open(videoW);

frame_padding = 200;
xLimits = [-frame_padding , width+frame_padding ];
yLimits = [0, height];

ref = imref2d([height, width + 2*frame_padding], xLimits, yLimits);

duration = 4; % in seconds
frames = duration*30;
max_shear = 0.5;

for t = 1:frames
    sh = max_shear*sin(2*2*pi*t/frames);
    T = [1 0 0;
        sh 1 0;
        -sh*height 0 1];
    tform = affine2d(T);

    shearedImg = imwarp(pudding, tform, 'OutputView',ref);
    writeVideo(videoW, shearedImg);
end

videoW = VideoWriter('sheared_pudding.avi', 'Motion JPEG AVI');
close(videoW);
implay('sheared_pudding.avi');

%% 5 - Windmill

clear clc
close all

windmill = im2double(imread('windmill.png'));
windmill_mask = im2double(imread('windmill_mask.png'));
windmill_back = im2double(imread('windmill_back.jpeg'));

if size(windmill_mask, 3) == 3
    windmill_mask = rgb2gray(windmill_mask);
end
windmill_mask = windmill_mask > 0.5;

windmill_mask = 1 - windmill_mask;
[windmill_h, windmill_w, ~] = size(windmill);
center_x = windmill_w/2;
center_y = windmill_h/2;
[windmill_back_h, windmill_back_w, ~] = size(windmill_back);

frames = 90;
target_x = windmill_back_w / 2;
target_y = windmill_back_h / 2;

videoW = VideoWriter('transf_windmill.avi', 'Motion JPEG AVI');
open(videoW);

ref_bg = imref2d([windmill_back_h, windmill_back_w]);

for t = 1:frames
    theta = -(2*pi*t/frames);

    T_to_origin = [1 0 0; 0 1 0; -center_x -center_y 1];
    T_rotate = [cos(theta)  sin(theta)  0;
                      -sin(theta) cos(theta)  0;
                      0                 0                 1];
    T_to_target = [1 0 0; 0 1 0; target_x target_y 1];
    
    T_final = T_to_origin * T_rotate* T_to_target;
    tform = affine2d(T_final);

    curr_blade = imwarp(windmill, tform, 'OutputView', ref_bg, 'FillValues', 0); 
    curr_mask = imwarp(windmill_mask, tform, 'OutputView', ref_bg, 'FillValues', 0);

    final_frame = windmill_back;
    
    mask_3ch = repmat(curr_mask, [1 1 3]);
    final_frame = final_frame .* (1 - mask_3ch) + curr_blade .* mask_3ch;
    
    writeVideo(videoW, final_frame);
end
close(videoW);
implay('transf_windmill.avi');

%% 6 - Interpolation

clear;
clc;
close all;

windmill = im2double(imread('windmill.png'));
windmill_mask = im2double(imread('windmill_mask.png'));
windmill_back = im2double(imread('windmill_back.jpeg'));

if size(windmill_mask, 3) == 3
    windmill_mask = rgb2gray(windmill_mask);
end
windmill_mask = windmill_mask > 0.5;

windmill_mask = 1 - windmill_mask;
[windmill_h, windmill_w, ~] = size(windmill);
center_x = windmill_w/2;
center_y = windmill_h/2;
[windmill_back_h, windmill_back_w, ~] = size(windmill_back);

frames = 90;
target_x = windmill_back_w / 2;
target_y = windmill_back_h / 2;

videoW = VideoWriter('transf_windmill.avi', 'Motion JPEG AVI');
open(videoW);

ref_bg = imref2d([windmill_back_h, windmill_back_w]);

for t = 1:frames
    theta = -(2*pi*t/frames);

    scale = 0.5;
    T_to_origin = [1 0 0; 0 1 0; -center_x -center_y 1];
    T_rotate = [cos(theta)  sin(theta)  0;
                      -sin(theta) cos(theta)  0;
                      0                 0                 1];
    T_to_target = [1 0 0; 0 1 0; target_x target_y 1];
    
    T_final = T_to_origin * T_rotate* T_to_target;
    tform = affine2d(T_final);

    method = 'cubic';
    curr_blade = imwarp(windmill, tform, method, 'OutputView', ref_bg, 'FillValues', 0);
    curr_mask = imwarp(windmill_mask, tform, method, 'OutputView', ref_bg, 'FillValues', 0);

    final_frame = windmill_back;
    
    mask_3ch = repmat(curr_mask, [1 1 3]);
    final_frame = final_frame .* (1 - mask_3ch) + curr_blade .* mask_3ch;
    
    % to not have type error
    final_frame(final_frame < 0) = 0;
    final_frame(final_frame > 1) = 1;
    writeVideo(videoW, final_frame);
end
close(videoW);
implay('transf_windmill.avi');

%% 7 - ball at the beach

clear;
clc;
close all;

ball = im2double(imread('ball.jpg'));
ball_mask_raw = im2double(imread('ball_mask.jpg'));
beach = im2double(imread('beach.jpg'));

if size(ball_mask_raw, 3) == 3
    ball_mask_raw = rgb2gray(ball_mask_raw);
end
ball_mask = ball_mask_raw < 0.5;

[h_ball, w_ball, ~] = size(ball);
center_ball_x = w_ball / 2;
center_ball_y = h_ball / 2;

[h_bg, w_bg, ~] = size(beach);
frames = 120;

start_x = -100;
start_y = h_bg - 150;
start_scale = 0.8;

end_x = w_bg + 100;
end_y = h_bg / 2.8;
end_scale = 0.02;

%linspace για ομαλή μετάβαση
path_x = linspace(start_x, end_x, frames);
path_y = linspace(start_y, end_y, frames);
path_scale = linspace(start_scale, end_scale, frames);

videoFileName = 'transf_beach.avi';
videoW = VideoWriter(videoFileName, 'Motion JPEG AVI');
videoW.FrameRate = 30; % Ρύθμιση των fps
open(videoW);

ref_bg = imref2d([h_bg, w_bg]);

base_y = h_bg - 150;
jump_height = 250;
freq = 3;
amp_y  = 80;
fixed_scale = 0.4;

for t = 1:frames
    

    wave = abs(sin(2 * pi * freq * (t / frames)));
    curr_x = path_x(t);
    curr_y = base_y - (jump_height * wave);
    curr_s = fixed_scale - (0.1 * wave);

    theta = -2*pi*t/15;
    T_origin = [1 0 0; 0 1 0; -center_ball_x -center_ball_y 1];
    T_rot_scale = [curr_s*cos(theta)  curr_s*sin(theta) 0;
                   -curr_s*sin(theta) curr_s*cos(theta) 0;
                   0                  0                 1];
    T_target = [1 0 0; 0 1 0; curr_x curr_y 1];
    T_final = T_origin * T_rot_scale * T_target;
    tform = affine2d(T_final);

    curr_ball = imwarp(ball, tform, 'cubic', 'OutputView', ref_bg, 'FillValues', 0);
    curr_mask = imwarp(ball_mask, tform, 'linear', 'OutputView', ref_bg, 'FillValues', 0);

    % bad values could come from cubic
    curr_ball(curr_ball < 0) = 0; 
    curr_ball(curr_ball > 1) = 1;

    mask_3ch = repmat(curr_mask, [1 1 3]);
    final_frame = beach .* (1 - mask_3ch) + curr_ball .* mask_3ch;
    writeVideo(videoW, final_frame);
end

close(videoW);
implay(videoFileName);


%% 8 - ball to the sea

clear;
clc;
close all;

ball = im2double(imread('ball.jpg'));
ball_mask_raw = im2double(imread('ball_mask.jpg'));
beach = im2double(imread('beach.jpg'));

if size(ball_mask_raw, 3) == 3
    ball_mask_raw = rgb2gray(ball_mask_raw);
end
ball_mask = ball_mask_raw < 0.5;

[h_ball, w_ball, ~] = size(ball);
center_ball_x = w_ball / 2;
center_ball_y = h_ball / 2;

[h_bg, w_bg, ~] = size(beach);
frames = 120;

start_x = w_bg / 2;
start_y = h_bg - 150;
start_scale = 0.8;

end_x = w_bg / 2 + 100;
end_y = h_bg / 2.8;
end_scale = 0.02;

%linspace για ομαλή μετάβαση
path_x = linspace(start_x, end_x, frames);
path_y = linspace(start_y, end_y, frames);
path_scale = linspace(start_scale, end_scale, frames);

videoFileName = 'transf_beach.avi';
videoW = VideoWriter(videoFileName, 'Motion JPEG AVI');
videoW.FrameRate = 30; % Ρύθμιση των fps
open(videoW);

ref_bg = imref2d([h_bg, w_bg]);

for t = 1:frames
    curr_x = path_x(t);
    curr_y = path_y(t);
    curr_s = path_scale(t);

    theta = -2*pi*t/15;
    T_origin = [1 0 0; 0 1 0; -center_ball_x -center_ball_y 1];
    T_rot_scale = [curr_s*cos(theta)  curr_s*sin(theta) 0;
                   -curr_s*sin(theta) curr_s*cos(theta) 0;
                   0                  0                 1];
    T_target = [1 0 0; 0 1 0; curr_x curr_y 1];
    T_final = T_origin * T_rot_scale * T_target;
    tform = affine2d(T_final);

    curr_ball = imwarp(ball, tform, 'cubic', 'OutputView', ref_bg, 'FillValues', 0);
    curr_mask = imwarp(ball_mask, tform, 'linear', 'OutputView', ref_bg, 'FillValues', 0);

    % bad values could come from cubic
    curr_ball(curr_ball < 0) = 0; 
    curr_ball(curr_ball > 1) = 1;

    mask_3ch = repmat(curr_mask, [1 1 3]);
    final_frame = beach .* (1 - mask_3ch) + curr_ball .* mask_3ch;
    writeVideo(videoW, final_frame);
end

close(videoW);
implay(videoFileName);

