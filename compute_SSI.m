function SSI = compute_SSI(fr_list,n)
% n is the number of neurons, fr_ist is the neuronal firing sequence, dt=0.1ms
w = 10;
t_end = length(fr_list);
dt = 0.1;
SSI = [];
weight = [];
res.fr_e = fr_list*dt;
for j = (w/2+1):(t_end-w/2-1)
    n_t = res.fr_e(j);
    num_s = sum(res.fr_e(j-w/2:j+w/2));
    if n_t ~= 0
        SSI = [SSI num_s];
        weight = [weight n_t];
    end
end
SSI = (sum(SSI.*weight)/sum(weight))/(n);
end
