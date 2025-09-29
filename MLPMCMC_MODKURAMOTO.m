% This code is MLPMCMC for the modified kuramoto model 
close all;
clear;
clc;
format long


particle_count = 50;
x0 = 1;
X0 = ones(particle_count, 1);
kuramoto_obs = readmatrix('Modkuramoto_obs_T_100.txt');
T = 100;
Lmin = 4;
LP = 5;
sigma_obs = 1;
% discretization size
hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%number of iterations for each level
Nl = [5000,5000];
%store the acceptance rate
Aln = zeros(1, 1);
Theta_trace = cell(LP - Lmin + 1, 1);
Theta_trace_1 = cell(LP - Lmin,1);
Theta_trace_2 = cell(LP - Lmin,1);

Theta_traceN = cell(LP - Lmin + 1, 1);
Theta_trace_1N = cell(LP - Lmin,1);
Theta_trace_2N = cell(LP - Lmin,1);

%mean of theta over iterations
ML_Theta_trace = cell(LP - Lmin + 1, 1);
%weights for finer and corse level        
H1_trace = cell(LP - Lmin, 1);
H2_trace = cell(LP - Lmin, 1);

for k = 1 : LP - Lmin + 1
    Theta_trace{k, 1} = zeros(Nl(k),3);
    Theta_traceN{k,1} = zeros(Nl(k),3);
    ML_Theta_trace{k, 1} = zeros(Nl(k),3);
end

for i = 1:LP - Lmin
    Theta_trace_1{i,1} = zeros(Nl(i+1),3);
    Theta_trace_2{i,1} = zeros(Nl(i+1),3);

    Theta_trace_1N{i,1} = zeros(Nl(i+1),3);
    Theta_trace_2N{i,1} = zeros(Nl(i+1),3);

    H1_trace{i,1} = zeros(Nl(i+1),1);
    H2_trace{i,1} = zeros(Nl(i+1),1);
end

delta = 2^(-Lmin);
Theta_A = [0,-1,0];
tic;
Theta_A_p = Theta_A;
Theta_SIG_p = [Theta_A_p(1), exp(Theta_A_p(2)), exp(Theta_A_p(3))];
X_measure = simulate_discrete_modkuramoto_p(delta, T, X0, Theta_SIG_p(1), Theta_SIG_p(2));
Z =  particle_filter(kuramoto_obs, X_measure, delta, T, X0, Theta_SIG_p(1), Theta_SIG_p(2), Theta_SIG_p(3));
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace{1,1}(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;
Sigma_A1 = 0.1*diag([1,3.8,3.5]);
Sigma_A = 0.1*diag([1,3.8,3.5]);
for iter = 1:Nl(1)
 
    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)),', ', num2str(Theta_A_p(3)),  ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1 = Theta_A_p;
    Theta_A_prime = mvnrnd(Theta_A_prime_1, Sigma_A1*Sigma_A1');
    Theta_SIG_prime = [Theta_A_prime(1),exp(Theta_A_prime(2)), exp(Theta_A_prime(3))];

    X_measure_prime = simulate_discrete_modkuramoto_p(delta, T, X0, Theta_SIG_prime(1), Theta_SIG_prime(2));
    Z_prime =  particle_filter(kuramoto_obs, X_measure_prime, delta, T, X0, Theta_SIG_prime(1), Theta_SIG_prime(2), Theta_SIG_prime(3));
    lZ_prime = Z_prime;
    l_pos_Theta_A_prime = l_posterior(Theta_A_prime, lZ_prime);
 
    alpha_U = min(0, l_pos_Theta_A_prime - l_pos_Theta_A_p);
    U = log(rand);
    
    if U < alpha_U
        Theta_A_p = Theta_A_prime;
        Theta_SIG_p = Theta_SIG_prime;
        X_measure = X_measure_prime;
        lZ = lZ_prime;
        l_pos_Theta_A_p = l_pos_Theta_A_prime;
        Theta_trace{1, 1}(iter,:) = Theta_A_prime; 
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace{1, 1}(iter,:) = Theta_A_p; 
        lZ = particle_filter(kuramoto_obs, X_measure, delta, T, X0, Theta_SIG_p(1), Theta_SIG_p(2), Theta_SIG_p(3));
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 

end

Aln = N_count_1 / Nl(1);

H1_sum = 0;
H2_sum = 0;
tic;

%mlpmmh
for l = 1:LP - Lmin 

    level = l + Lmin;
    fprintf('level = %f\n', level);
    delta_t = 1/2^(level);
    X0_m = x0 * ones(50, 1);
    X0_pf = x0 * ones(50, 1);

   Theta_l = mean(Theta_trace{1,1});
   %Theta_l = [-0.02, -1.5, 0.7];
   Theta_SIG_l = [Theta_l(1),exp(Theta_l(2)),exp(Theta_l(3))];

    [X_1_measure, X_2_measure] = simulate_coupled_discrete_modkuramoto(delta_t, T, X0_m, X0_m, Theta_SIG_l(1), Theta_SIG_l(2));
    [H1_l, H2_l, G_l] = particle_filter_coupled(kuramoto_obs, X_1_measure, X_2_measure, delta_t, T, X0_pf, X0_pf, Theta_SIG_l(1), Theta_SIG_l(2), Theta_SIG_l(3));
    lG_l = G_l;
    l_pos_theta_l = l_posterior(Theta_l, lG_l);
    
    N_count_l = 0;

    for iter = 1:Nl(l+1)

        if mod(iter, 50) == 0
            fprintf('iter = %f\n', iter);
            fprintf('AR = %f\n', N_count_l/iter);
            fprintf('H1 average = %f\n', H1_sum/50);
            fprintf('H2 average = %f\n', H2_sum/50);
            disp(['current estimate = [', num2str(Theta_l(1)), ', ', num2str(Theta_l(2)), ', ', num2str(Theta_l(3)),  ']']);
            H1_sum = 0;
            H2_sum = 0;
            toc;
            tic;
        end
        
        Theta_l_prime1 = Theta_l;
        Theta_l_prime = mvnrnd(Theta_l_prime1,Sigma_A*Sigma_A');
        Theta_l_SIG_prime = [Theta_l_prime(1),exp(Theta_l_prime(2)),exp(Theta_l_prime(3))];
        
        [X_1_measure_p, X_2_measure_p] = simulate_coupled_discrete_modkuramoto(delta_t, T, X0_m, X0_m, Theta_l_SIG_prime(1), Theta_l_SIG_prime(2));
        [H1_lp, H2_lp, lG_lp] = particle_filter_coupled(kuramoto_obs, X_1_measure_p, X_2_measure_p, delta_t, T, X0_pf, X0_pf, Theta_l_SIG_prime(1), Theta_l_SIG_prime(2), Theta_l_SIG_prime(3));
        l_pos_theta_l_prime = l_posterior(Theta_l_prime, lG_lp);
        alpha_l = min(0, l_pos_theta_l_prime - l_pos_theta_l);

        Ul = log(rand);
        if Ul < alpha_l
            Theta_l = Theta_l_prime;
            Theta_SIG_l = Theta_l_SIG_prime;
            Theta_trace{l+1, 1}(iter,:) = Theta_l_prime;
            X_1_measure = X_1_measure_p;
            X_2_measure = X_2_measure_p;
            lG_l = lG_lp;
            l_pos_theta_l = l_pos_theta_l_prime;
            H1_l = H1_lp;
            H2_l = H2_lp;
            H1_trace{l, 1}(iter,1) = H1_lp;
            H2_trace{l, 1}(iter,1) = H2_lp;
            N_count_l= N_count_l + 1;
            H1_sum = H1_sum + H1_lp;
            H2_sum = H2_sum + H2_lp;
            
        else
            Theta_trace{l+1, 1}(iter,:) = Theta_l; 
            [X_1_measure, X_2_measure] = simulate_coupled_discrete_modkuramoto(delta_t, T, X0_m, X0_m, Theta_SIG_l(1), Theta_SIG_l(2));
            [H1_l, H2_l, lG_l] = particle_filter_coupled(kuramoto_obs, X_1_measure, X_2_measure, delta_t, T, X0_pf, X0_pf, Theta_SIG_l(1), Theta_SIG_l(2), Theta_SIG_l(3));
            l_pos_theta_l = l_posterior(Theta_l, lG_l);
            H1_trace{l, 1}(iter,1) = H1_l;
            H2_trace{l, 1}(iter,1) = H2_l;
            H1_sum = H1_sum + H1_l;
            H2_sum = H2_sum + H2_l;
        end   
    end
        Aln(l+1,1) = N_count_l/ Nl(l+1);        
end

toc;

burnin = 1;
for ll = 1:LP - Lmin
    for i = 1:3      
        Theta_trace_1N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))));
        Theta_trace_2N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H2_trace{ll,1}(:,1)) / sum(exp(H2_trace{ll,1}(:,1))));
        Theta_traceN{ll+1,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))) - exp(H2_trace{ll,1}(:,1))/ sum(exp(H2_trace{ll,1}(:,1))));
        ML_Theta_trace{ll+1,1}(:,i) = cumsum(Theta_traceN{ll+1, 1}(:,i)) ./ (1:Nl(ll+1))';
    end
end


final_theta =  mean(Theta_trace{1,1}(burnin:end,:));
level_means = zeros(LP-Lmin, 3);

for i=1:3
    for j = 1:LP - Lmin
        final_theta(i) = final_theta(i) + sum(Theta_traceN{j+1,1}(burnin:end,i));
        level_means(j,i) = sum(Theta_traceN{j+1,1}(burnin:end,i));
    end
end




Theta_iters = Theta_trace{1,1} +  Theta_traceN{2,1} ;
burnin = 1;
niter = 5000;
desired_height = 0.12;
figure_distance = 400;
f = figure;
f.Position = f.Position+[0 -figure_distance 0 figure_distance];

ax = subplot(3,1,1);
plot(burnin:3:niter,Theta_iters(burnin:3:end,1), 'r-',LineWidth=1);
title('\theta');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(3,1,2);
plot(burnin:3:niter,Theta_iters(burnin:3:end,2), 'r-',LineWidth=1);
title('log(\sigma)');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');


ax = subplot(3,1,3);
plot(burnin:3:niter,Theta_iters(burnin:3:end,3), 'r-',LineWidth=1);
title('log(\sigma_{obs})')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');




function X = simulate_discrete_modkuramoto_p(delta_t, T, X0, theta, sigma)
    particle_count = length(X0);     
    steps_count = round(T/delta_t);
    X = zeros(particle_count, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count);
    for i = 1:steps_count
        X(:,i+1) = X(:,i) + (theta + mean(sin(X(:,i) - X(:,i)'),2)) * delta_t + ...
                          + sigma/(1+ X(:,i).^2) * delta_W(:,i);
    end
end

function [X_1, X_2] = simulate_coupled_discrete_modkuramoto(delta_t, T, X0_1, X0_2, theta, sigma)
    particle_count = length(X0_1);     
    steps_count_2 = round(T/delta_t/2);      
    steps_count_1 = 2*steps_count_2;
    X_1 = zeros(particle_count, steps_count_1+1);
    X_2 = zeros(particle_count, steps_count_2+1);
    X_1(:,1) = X0_1;
    X_2(:,1) = X0_2;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count_1);
    for i = 2:steps_count_1+1
        X_1(:,i) = X_1(:,i-1) + (theta + mean(sin(X_1(:,i-1) - X_1(:,i-1)'),2)) * delta_t + ...
                          + sigma /(1+ X_1(:,i).^2)* delta_W(:,i-1);
        if mod(i,2) == 1
            j = ceil(i/2);
            X_2(:,j) = X_2(:,j-1) + (theta + mean(sin(X_2(:,j-1) - X_2(:,j-1)'),2)) * 2*delta_t + ...
                          + sigma/(1+ X_2(:,j).^2) * (delta_W(:,i-1) + delta_W(:,i-2));         
        end
    end
end



 
function z = particle_filter(Y, X_measure, delta_t, T, X0, theta, sigma, sigma_obs)
    particle_count = length(X0);    
    steps_count = round(T/delta_t);
    X = zeros(particle_count, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count);
    log_w = zeros(particle_count, 1);
    k = 1;
    lGL_star = zeros(1,T);
    for i = 2:steps_count+1
        X(:,i) = X(:,i-1) + (theta + mean(sin(X(:,i-1) - X_measure(:,i-1)'),2)) * delta_t + ...
                          + sigma /(1+ X(:,i).^2)* delta_W(:,i-1);       
        if i == round(k/delta_t)+1
            log_w = log_normpdf(X(:,i), Y(k), sigma_obs);
            W = exp(log_w - max(log_w));
            lGL_star(1,k)= log(sum(W)) + max(log_w);
            W = W / sum(W);
            if 1/sum(W.^2) <= particle_count
                I = resampleSystematic(W);
                X = X(I,:);
            end
            k = k + 1;
        end
    end
    z = T * log(1/particle_count) + sum(lGL_star);
end


function [lwf,lwc,cz] = particle_filter_coupled(Y, X_1_measure, X_2_measure, delta_t, T, X0_1, X0_2, theta, sigma, sigma_obs)
    particle_count = length(X0_1);     
    steps_count_2 = T/delta_t/2;       
    steps_count_1 = T/delta_t;
    X_1 = zeros(particle_count, steps_count_1+1);
    X_2 = zeros(particle_count, steps_count_2+1);
    X_1(:,1) = X0_1;
    X_2(:,1) = X0_2;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count_1);
    k = 1;
    log_w = zeros(particle_count, 1);
    lGL_star = zeros(1,T);
    lwf = 0;
    lwc = 0;
    for i = 1:steps_count_1
        X_1(:,i+1) = X_1(:,i) + (theta + mean(sin(X_1(:,i) - X_1_measure(:,i)'),2)) * delta_t + ...
                          + sigma/(1+ X_1(:,i).^2) * delta_W(:,i);
        if mod(i,2) == 0
            j = i/2;
            X_2(:,j+1) = X_2(:,j) + (theta + mean(sin(X_2(:,j) - X_2_measure(:,j)'),2)) * 2*delta_t + ...
                          + sigma /(1+ X_2(:,j).^2)* (delta_W(:,i-1) + delta_W(:,i));           
        end

        if mod(i, 1/delta_t)==0
            j = i/2;
            log_w_1 = log_normpdf(X_1(:,i+1), Y(i*delta_t), sigma_obs);
            log_w_2 = log_normpdf(X_2(:,j+1), Y(i*delta_t), sigma_obs);
            
            log_w = max(log_w_1, log_w_2);
            lwf = lwf + mean((log_w_1 - log_w));
            lwc = lwc + mean((log_w_2 - log_w));
            if lwf == inf
                 disp('inf lwf');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end
            
            if lwc == inf
                 disp('inf lwc');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end

            GL0 = exp(log_w - max(log_w));
            lGL_star(1,k)= log(sum(GL0)) + max(log_w);
           GLL = GL0 / sum(GL0);
        
            if isnan(sum(GLL)) 
                disp('ANNOYING NAN ERROR! GLL');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end
            if  sum(GLL) == 0
                disp(' GLL = 0');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end

           I = resampleSystematic( GLL);
           X_1 = X_1(I,:);
           X_2 = X_2(I,:);
           k = k + 1;
        end
    end
    %fprintf('lwf = = %f\n', lwf);
    %fprintf('lwc= = %f\n', lwc);
     cz = T * log(1/particle_count) + sum(lGL_star);
end

function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end

function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior = lG(Theta(1),0,0.1) + lG(Theta(2), -1.6, 0.001) + lG(Theta(2), 0, 0.01) ;
    lpos_p = log_lik + log_prior;
    
end

function  indx  = resampleSystematic(w)
    N = length(w);
    Q = cumsum(w);
    indx = zeros(1,N);
    T = linspace(0,1-1/N,N) + rand(1)/N;
    T(N+1) = 1;
    i=1;
    j=1;
    while (i<=N)
        if (T(i)<Q(j))
            indx(i)=j;
            i=i+1;
        else
            j=j+1;        
        end
    end
end

function a = log_normpdf(x,m,s)
    a = -log(2*pi)/2 - log(s) - (x-m).^2/(2*s^2);
    %a = -(x-m).^2/(2*s^2);
end
