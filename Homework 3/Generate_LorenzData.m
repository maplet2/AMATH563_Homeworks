clear all; close all; clc
% How to use neural nets in solving/predicting dynamical systems 
% Simulate Lorenz system
input = [];
output = [];
rs = [10, 28, 40]; 
for k = 1:3
    dt=0.01; T=8; t=0:dt:T; % defining one trajectory
    b=8/3; sig=10; r=rs(k);

    Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                      r * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    % Get really accurate trajectories from this

    % The goal is to find a mapping that will take our system from time t to
    % time t + dt. This will give us an idea about the future state of the
    % system.
    
    for j=1:150 % training trajectories
        x0=30*(rand(3,1)-0.5);
        [t,y] = ode45(Lorenz,t,x0);
        input=[input; [y(1:end-1,:),ones(800,1)*r]];
        output=[output; [y(2:end,:)]];
    end
end

save('lorenz.mat', 'input', 'output')
%%
close all;
input2 = [];
output2 = [];
rs2 = [17, 35];
for k=1:2 % training trajectories
        dt=0.01; T=8; t=0:dt:T; % defining one trajectory
    b=8/3; sig=10; r=rs2(k);

    Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                      r * x(1)-x(1) * x(3) - x(2) ; ...
                      x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    input2=[input2; [y(1:end-1,:),ones(800,1)*r]];
    output2=[output2; [y(2:end,:)]];
end
save('lorenz_test.mat', 'input2', 'output2')

plot3(input2(1:800,1), input2(1:800,2), input2(1:800,3))
hold on
plot3(input2(801:end,1), input2(801:end,2), input2(801:end,3), 'r')
legend('17','35')