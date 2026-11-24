close all

%Read a video
vidObj = VideoReader('video1.mp4');

vidFrames = read(vidObj);

% Get the number of frames.
numFrames = get(vidObj, 'NumFrames');

window = 25;
STEP = 10;

f1 = figure;

for i=1:numFrames-1
    
    frame1 = vidFrames(:,:,:,i);
    frame2 = vidFrames(:,:,:,i+1);
    

    frame1 = imresize(frame1,0.5);
    frame2 = imresize(frame2,0.5);
    
    [u, v] = calculate_of(im2double(rgb2gray(frame1)),im2double(rgb2gray(frame2)),window,0.0001);


    u_deci = u(1:STEP:end, 1:STEP:end);
    v_deci = v(1:STEP:end, 1:STEP:end);
    
    off = floor(window/2)+1;
    [sz1,sz2, sz3]=size(frame1);
    f = frame1(1:sz1-off,1:sz2-off);
    
    u_tot = zeros(sz1, sz2);
    v_tot = zeros(sz1, sz2);

    [X,Y] = meshgrid(1:sz2, 1:sz1); 
    X_deci = X(1:STEP:end, 1:STEP:end);
    Y_deci = Y(1:STEP:end, 1:STEP:end);
    
      % figure(f1);
      % imshow(frame1);
      % hold on;
      % quiver(X_deci, Y_deci, u_deci, v_deci, 2, 'y', 'LineWidth', 1.5); 
      % hold off;
     
      u_tot = u_tot + u;
      v_tot = v_tot + v;
     
     [x, y] = meshgrid(1:size(f,2), 1:size(f,1));
    
    
    pause(0.1);

    % αραιή οπιτκή ροή:

    [i_corners, j_corners] = cornerDetection(im2double(rgb2gray(frame1)), 10, 0.05);
    i_corners = i_corners(1:10:end);
    j_corners = j_corners(1:10:end);

    indices = sub2ind(size(u), i_corners, j_corners);
    u_sparse = u(indices);
    v_sparse = v(indices);

     % figure(f1);
     % imshow(frame1);
     %  hold on;
     %  quiver(j_corners, i_corners, u_sparse, v_sparse, 2, 'r', 'LineWidth', 1.5);
     %  hold off;
     % 
     %  u_tot = u_tot + u;
     %  v_tot = v_tot + v;
  
      opticFlow = opticalFlowLK;
     flow = estimateFlow(opticFlow, im2double(rgb2gray(frame1)));
     plot(flow, 'DecimationFactor', [10 10], 'ScaleFactor', 10);
end



