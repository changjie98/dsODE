function [res] = run_LIF_model(params,start)
res = struct;
flag = true;
if nargin > 2
    error('too many variables！');
elseif nargin == 1
    flag = false; 
end
%% preparation
dt = params.dt;
ne = params.ne;
ni = params.ni;
t_end = params.duration_time/dt;
V_e = zeros(t_end, ne);
% V_e(1,1:300) = 3;
V_i = zeros(t_end, ni);
H_ie = zeros(t_end, ni);
H_ii = zeros(t_end, ni);
H_ee = zeros(t_end, ne);
H_ei = zeros(t_end, ne);
H_e_adapt = zeros(t_end, ne);
H_i_adapt = zeros(t_end, ni);
nf_e = zeros(t_end, 1);
nf_i = zeros(t_end, 1);
ref_e = zeros(t_end, ne);
ref_i = zeros(t_end, ni);
res.I_ee = zeros(t_end, ne);
res.I_ei = zeros(t_end, ne);
res.I_ie = zeros(t_end, ni);
res.I_ii = zeros(t_end, ni);
res.ft = zeros(t_end, ne+ni);

J_eex = random('normal',params.Ex_Poisson_lambda*dt ,sqrt(params.Ex_Poisson_lambda*dt),t_end,ne);
J_iex = random('normal',params.Ex_Poisson_lambda*dt ,sqrt(params.Ex_Poisson_lambda*dt),t_end,ni);
if flag
    V_e(1,:) = start.V_e;
    V_i(1,:) = start.V_i;
    H_ee(1,:) = start.H_ee;
    H_ie(1,:) = start.H_ie;
    H_ei(1,:) = start.H_ei;
    H_ii(1,:) = start.H_ii; 
    nf_e(1) = start.fr_e*dt;
    nf_i(1) = start.fr_i*dt;
    ref_e(1,:) = start.ref_e;
    ref_i(1,:) = start.ref_i;
end

connection_matrix_e = zeros(ne,ne+ni);
connection_matrix_i = zeros(ni,ne+ni);
flag2 = 4;
switch flag2
    case 1 % Connectionless
    connection_matrix_e(:,1:ne)=zeros(ne,ne);
    connection_matrix_i(:,1:ne)=zeros(ni,ne);
    connection_matrix_e(:,ne+1:ne+ni)=zeros(ne,ni);
    connection_matrix_i(:,ne+1:ne+ni)=zeros(ni,ni);
    case 2 % Excitatory Connection Only
    connection_matrix_e(:,1:ne)=binornd(1,params.p_ee,ne,ne);
    connection_matrix_i(:,1:ne)=zeros(ni,ne);
    connection_matrix_e(:,ne+1:ne+ni)=zeros(ne,ni);
    connection_matrix_i(:,ne+1:ne+ni)=zeros(ni,ni);
    case 3 % Only inhibitory connections
    connection_matrix_e(:,1:ne)=zeros(ne,ne);
    connection_matrix_i(:,1:ne)=zeros(ni,ne);
    connection_matrix_e(:,ne+1:ne+ni)=zeros(ne,ni);
    connection_matrix_i(:,ne+1:ne+ni)=binornd(1,0.1,ni,ni);
    case 4 % normal connection
    connection_matrix_e(:,1:ne)=binornd(1,params.p_ee,ne,ne);
    connection_matrix_i(:,1:ne)=binornd(1,params.p_ei,ni,ne);
    connection_matrix_e(:,ne+1:ne+ni)=binornd(1,params.p_ie,ne,ni);
    connection_matrix_i(:,ne+1:ne+ni)=binornd(1,params.p_ii,ni,ni);
    case 5  % Two Exciting Groups
    connection_matrix_e(1:ne/2,1:ne/2)=binornd(1,params.p_ee,ne/2,ne/2);
    connection_matrix_e(ne/2+1:ne,ne/2+1:ne)=binornd(1,params.p_ee,ne/2,ne/2);
    connection_matrix_e(1:ne/2,ne+1:ne+ni)=binornd(1,params.p_ie,ne/2,ni);
    connection_matrix_e(ne/2+1:ne,ne+1:ne+ni)=binornd(1,params.p_ie,ne/2,ni);
    connection_matrix_i(:,1:ne)=binornd(1,params.p_ei,ni,ne);
    connection_matrix_i(:,ne+1:ne+ni)=binornd(1,params.p_ii,ni,ni);
end
connection_mat = [connection_matrix_e; connection_matrix_i];
% connection_mat(logical(eye(ne+ni))) = 0; % 去掉自身的连接


%% run model
for i = 2:t_end % Calculate from 'this moment'
    %% Calculate the current at the previous moment
    J_ee = params.s_ee;
    J_ie = params.s_ie;
%     J_ei = params.s_ei;
%     J_ii = params.s_ii;
    J_ei = (V_e(i-1,:)+params.Mr)*params.s_ei/(params.M+params.Mr);    
    J_ii = (V_i(i-1,:)+params.Mr)*params.s_ii/(params.M+params.Mr);

    I_eex = J_eex(i,:);
    I_iex = J_iex(i,:);
    I_ee = J_ee.*H_ee(i-1,:)/params.tau_ee*dt;
    I_ei = J_ei.*H_ei(i-1,:)/params.tau_i*dt;    
    I_ie = J_ie.*H_ie(i-1,:)/params.tau_ie*dt;
    I_ii = J_ii.*H_ii(i-1,:)/params.tau_i*dt;

%     I_ee = J_ee.*H_ee(i-1,:)/params.tau_ee;
%     I_ei = J_ei.*H_ei(i-1,:)/params.tau_i;    
%     I_ie = J_ie.*H_ie(i-1,:)/params.tau_ie;
%     I_ii = J_ii.*H_ii(i-1,:)/params.tau_i;
%     I_ee = (I_ee-mean(I_ee))*sqrt(dt) + mean(I_ee)*dt;
%     I_ei = (I_ei-mean(I_ei))*sqrt(dt) + mean(I_ei)*dt;
%     I_ie = (I_ie-mean(I_ie))*sqrt(dt) + mean(I_ie)*dt;
%     I_ii = (I_ii-mean(I_ii))*sqrt(dt) + mean(I_ii)*dt;
    
    res.I_ee(i,:) = I_ee;
    res.I_ei(i,:) = I_ei;
    res.I_ie(i,:) = I_ie;
    res.I_ii(i,:) = I_ii;

    %% The voltage at the previous moment is obtained by adding current to the voltage. 
    %The voltage above the threshold is reset to zero and added to the nf array
    e_index = find(~ref_e(i,:));
    i_index = find(~ref_i(i,:));

    % Leakage current
    if params.tau_m~=0
        I_leak_e = -V_e(i-1,:).*(1/params.tau_m)*dt;
        I_leak_i = -V_i(i-1,:).*(1/params.tau_m)*dt;
        I_eex = I_eex/params.tau_m;
        I_iex = I_iex/params.tau_m;
    else
        I_leak_e = zeros(1, ne);
        I_leak_i = zeros(1, ni);
    end
    % Adaptive current
    if params.tau_adapt~=0
        I_adapt_e = -H_e_adapt(i-1,:)/params.tau_adapt;
        I_adapt_i = -H_i_adapt(i-1,:)/params.tau_adapt;
    else
        I_adapt_e = zeros(1, ne);
        I_adapt_i = zeros(1, ni);
    end
    
    V_e(i,e_index) =  I_eex(e_index) + (I_leak_e(e_index) + I_adapt_e(e_index) + I_ee(e_index) - I_ei(e_index)) + V_e(i-1,e_index);
    V_i(i,i_index) =  I_iex(i_index) + (I_leak_i(i_index) + I_adapt_i(i_index) + I_ie(i_index) - I_ii(i_index)) + V_i(i-1,i_index);
%     V_i(i,i_index) = zeros(1,ni);
    nf_e(i) = sum(V_e(i,:) > params.M);
    nf_i(i) = sum(V_i(i,:) > params.M);
    res.ft(i,:) = ([V_e(i,:) > params.M V_i(i,:) > params.M]);
    
    if params.tau_adapt~=0
        H_e_adapt_generate = 100*res.nft(i,1:ne);
        H_e_adapt_consume = H_e_adapt(i-1,:).*(1/params.tau_adapt)*dt;
        H_e_adapt(i,:) = H_e_adapt(i-1,:) + H_e_adapt_generate - H_e_adapt_consume;    
        H_i_adapt_generate = 100*res.nft(i,ne+1:ne+ni);
        H_i_adapt_consume = H_i_adapt(i-1,:).*(1/params.tau_adapt)*dt;
        H_i_adapt(i,:) = H_i_adapt(i-1,:) + H_i_adapt_generate - H_i_adapt_consume;   
    end
    if params.tau_r ~= 0
        ref_e(i,V_e(i,:) > params.M) = 1;
        ref_i(i,V_i(i,:) > params.M) = 1;
    end

    V_e(i,V_e(i,:) > params.M) = 0;
    V_i(i,V_i(i,:) > params.M) = 0;
    
    %% Calculate H at this moment
    connect = 'nofixed'; %Deciding whether it is a random connection or a fixed connection
    if strcmp(connect, 'fixed')
        fire_index = find(res.ft(i,:));
        efire_index = fire_index(fire_index<=ne);
        ifire_index = fire_index(fire_index>ne);
        Hee_generate = sum(connection_mat(efire_index,1:ne),1);
        Hie_generate = sum(connection_mat(efire_index,(ne+1):(ne+ni)),1);
        Hei_generate = sum(connection_mat(ifire_index,1:ne),1);
        Hii_generate = sum(connection_mat(ifire_index,(ne+1):(ne+ni)),1);
    else        
        Hee_generate = binornd(nf_e(i)*ones(1,ne), params.p_ee);
        Hei_generate = binornd(nf_i(i)*ones(1,ne), params.p_ei);
        Hie_generate = binornd(nf_e(i)*ones(1,ni), params.p_ie);
        Hii_generate = binornd(nf_i(i)*ones(1,ni), params.p_ii);
        
%         Hee_generate(1:ne/2) = binornd(sum(res.ft(i,1:ne/2))*ones(1,ne/2), params.p_ee);
%         Hee_generate(ne/2+1:ne) = binornd(sum(res.ft(i,ne/2+1:ne))*ones(1,ne/2), params.p_ee);
%         Hei_generate = binornd(nf_i(i)*ones(1,ne), params.p_ei);
%         Hie_generate = binornd(nf_e(i)*ones(1,ni), params.p_ie);
%         Hii_generate = binornd(nf_i(i)*ones(1,ni), params.p_ii);
    end

    Hee_consume = random('Poisson',H_ee(i-1,:)./params.tau_ee*dt);
    Hei_consume = random('Poisson',H_ei(i-1,:)./params.tau_i*dt);    
    Hie_consume = random('Poisson',H_ie(i-1,:)./params.tau_ie*dt);
    Hii_consume = random('Poisson',H_ii(i-1,:)./params.tau_i*dt);    
        
%     Hee_consume = H_ee(i-1,:).*(1/params.tau_ee)*dt;
%     Hei_consume = H_ei(i-1,:).*(1/params.tau_i)*dt;
%     Hie_consume = H_ie(i-1,:).*(1/params.tau_ie)*dt;
%     Hii_consume = H_ii(i-1,:).*(1/params.tau_i)*dt;
   
    
    H_ee(i,:) = H_ee(i-1,:) + Hee_generate - Hee_consume;    
    H_ei(i,:) = H_ei(i-1,:) + Hei_generate - Hei_consume;
    H_ie(i,:) = H_ie(i-1,:) + Hie_generate - Hie_consume;
    H_ii(i,:) = H_ii(i-1,:) + Hii_generate - Hii_consume;

    H_ee(i,(H_ee(i,:)<0)) = 0; 
    H_ei(i,(H_ei(i,:)<0)) = 0;
    H_ie(i,(H_ie(i,:)<0)) = 0;
    H_ii(i,(H_ii(i,:)<0)) = 0;
    
    if params.tau_r ~= 0
        % If there is a refractory period, you can choose whether it is a 
        % Poisson random refractory period or a fixed duration refractory period. 
        % Here, we choose a Poisson random refractory period
        ref_e_consume = random('Poisson',ref_e(i,:)./params.tau_r*dt);
        ref_i_consume = random('Poisson',ref_i(i,:)./params.tau_r*dt);    
%         ref_e_consume = ref_e(i,:).*(1-exp(-dt/params.tau_r));
%         ref_i_consume = ref_i(i,:).*(1-exp(-dt/params.tau_r));
        ref_e(i+1,:) = ref_e(i,:) - ref_e_consume;    
        ref_i(i+1,:) = ref_i(i,:) - ref_i_consume;
        ref_e(i+1,(ref_e(i+1,:)<0.37)) = 0;
        ref_i(i+1,(ref_i(i+1,:)<0.37)) = 0;
    end
end

res.V_e = V_e;
res.V_i = V_i;
res.H_ie = H_ie;
res.H_ii = H_ii;
res.H_ee = H_ee;
res.H_ei = H_ei;

res.nf_e = nf_e;
res.nf_i = nf_i;
res.fr_e = nf_e/dt;
res.fr_i = nf_i/dt;
res.ref_e = ref_e(1:end,:);
res.ref_i = ref_i(1:end,:);
res.I_eex = J_eex;
res.I_iex = J_iex;
res.I_e_mean = mean(I_eex,2)+mean(res.I_ee,2)-mean(res.I_ei,2);
res.I_i_mean = mean(I_iex,2)+mean(res.I_ie,2)-mean(res.I_ii,2);
% res.t = dt:dt:params.duration_time;
res.H_e_adapt = H_e_adapt;
res.H_i_adapt = H_i_adapt;
end
