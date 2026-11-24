% Initialization
clear; clc;

% Parameters
num_steps = 100;
true_radius = 3;
initial_center = [50; 50];
process_noise_range = 1;      % Uniform noise [-1, 1]
measurement_noise_std = 2;    % Gaussian measurement noise

% Kalman Filter setup
x = [initial_center; 0; 0];   % [x; y; vx; vy]
P = eye(4) * 10;              % Initial uncertainty

dt = 1;
A = [1 0 dt 0;
     0 1 0 dt;
     0 0 1 0;
     0 0 0 1];                % State transition
H = [1 0 0 0;
     0 1 0 0];                % Measurement matrix

Q = 0.1 * eye(4);             % Process noise covariance
R = measurement_noise_std^2 * eye(2); % Measurement noise

% Storage for plotting
true_positions = zeros(2, num_steps);
measured_positions = zeros(2, num_steps);
estimated_positions = zeros(2, num_steps);

% Simulation loop
center = initial_center;
for t = 1:num_steps
    % True motion (random walk with uniform noise)
    center = center + (rand(2,1)*2 - 1) * process_noise_range;
    true_positions(:,t) = center;

    % Measurement (noisy observation)
    measurement = center + randn(2,1) * measurement_noise_std;
    measured_positions(:,t) = measurement;

    % --- Kalman Filter ---
    % Predict
    x = A * x;
    P = A * P * A' + Q;

    % Update
    y = measurement - H * x;
    S = H * P * H' + R;
    K = P * H' / S;
    x = x + K * y;
    P = (eye(4) - K * H) * P;

    estimated_positions(:,t) = x(1:2);
end

% Plotting
figure;
for t = 1:num_steps
    clf;
    viscircles(true_positions(:,t)', true_radius, 'Color', 'k', 'LineStyle', '--');
    viscircles(measured_positions(:,t)', true_radius, 'Color', 'r', 'LineStyle', ':');
    viscircles(estimated_positions(:,t)', true_radius, 'Color', 'b');

    legend('True', 'Measured', 'Estimated');
    axis([30 70 30 70]);
    axis equal;
    title(sprintf('Time Step %d', t));
    drawnow;
    pause(0.1);
end
