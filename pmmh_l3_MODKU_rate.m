LP = 3;

job_id = getenv('SLURM_JOB_ID');
proc_id = getenv('SLURM_PROCID');
folder_read = '';
folder_write = sprintf('%s%s', job_id, '/');
results_filename = sprintf('%sL_%i_%s_%s.txt', folder_write, LP, job_id,proc_id);
rng_seed = sum(clock)*mod(str2num(job_id),10000)*(str2num(proc_id)+1);
%rng(rng_seed);
format long

kuramoto_obs = readmatrix(sprintf('%s%s', folder_read,'Modkuramoto_obs_T_100.txt'));
particle_count = 100;
T = 100;
%sigma_obs = 1;
M = floor(0.2*2^(2*LP)+40);
Nl =  floor(0.5 * 2^(2*LP)+20);
x0 = 1;
X0_M = x0*ones(M, 1);
X0_N = x0*ones(particle_count, 1);
Theta_trace = zeros(Nl,3);
delta = 2^(-LP);
Theta_A = [0,-2,0.7];
tic;

Theta_A_p = Theta_A;
Theta_SIG_p = [Theta_A_p(1), exp(Theta_A_p(2)), exp(Theta_A_p(3))];
X_measure = simulate_discrete_modkuramoto_p(delta, T, X0_M, Theta_SIG_p(1), Theta_SIG_p(2));
Z =  particle_filter(kuramoto_obs, X_measure, delta, T, X0_N, Theta_SIG_p(1), Theta_SIG_p(2), Theta_SIG_p(3));
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;


Sigma_A1 = 0.1*diag([1,3,3.5]);
for iter = 1:Nl
 
    if mod(iter, 50) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        %disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)), ', ', num2str(Theta_A_p(3)), ', ', num2str(Theta_A_p(4)), ', ', num2str(Theta_A_p(5)), ', ', num2str(Theta_A_p(6)),  ', ', num2str(Theta_A_p(7)), ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 50) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1 = Theta_A_p;
    Theta_A_prime = mvnrnd(Theta_A_prime_1, Sigma_A1*Sigma_A1');
    Theta_SIG_prime = [Theta_A_prime(1),exp(Theta_A_prime(2)), exp(Theta_A_prime(3))];

    X_measure_prime = simulate_discrete_modkuramoto_p(delta, T, X0_M, Theta_SIG_prime(1), Theta_SIG_prime(2));
    Z_prime =  particle_filter(kuramoto_obs, X_measure_prime, delta, T, X0_N, Theta_SIG_prime(1), Theta_SIG_prime(2), Theta_SIG_prime(3));
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
        Theta_trace(iter,:) = Theta_A_prime; 
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace(iter,:) = Theta_A_p; 
        lZ = particle_filter(kuramoto_obs, X_measure, delta, T, X0_N, Theta_SIG_p(1), Theta_SIG_p(2), Theta_SIG_p(3));
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 

   
end

Aln = N_count_1 / Nl;
toc;

burnin = 1;
final_theta = mean(Theta_trace(burnin:end,:),1);
writematrix(final_theta, results_filename);



%particle filter using diffusion bridge
function X = simulate_discrete_modkuramoto_p(delta_t, T, X0, theta, sigma)
    particle_count = length(X0);     
    steps_count = round(T/delta_t);
    X = zeros(particle_count, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count);
    for i = 1:steps_count
        %{
        if mod(i,100) == 0
            disp(['i = ', num2str(i)])
        end
        %}
        X(:,i+1) = X(:,i) + (theta + mean(sin(X(:,i) - X(:,i)'),2)) * delta_t + ...
                          + sigma/(1+ X(:,i).^2) * delta_W(:,i);
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
                          + sigma/(1+ X(:,i).^2) * delta_W(:,i-1);       
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

function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end

function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior = lG(Theta(1),0,0.1) + lG(Theta(2), -1.6, 0.001) + lG(Theta(3), 0, 0.01) ;
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

