data = res_lif.ft;
data = data';
data = flipud(data);
[row1, col1] = find(data(1:params.ni,:) == 1);
[row2, col2] = find(data(params.ni+1:params.ni+params.ne/2,:) == 1);
%[row3, col3] = find(data(params.ni+params.ne/2+1:params.ni+params.ne,:) == 1);
% figure
hold on
scatter(col1*0.1, row1, 20, 'b.');
hold on
scatter(col2*0.1, row2+params.ni, 20, 'r.');
% scatter(col3*0.1, row3+params.ni+params.ne/2, 20, 'g.');
% xlim([1700,2000])
ylim([0,700])
