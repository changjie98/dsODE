flag = 1;
if flag == 1 % RDM
    x = (1:51)*0.06+1-0.06;
    y1 = abs(taue60_dsODE_error);
    y2 = abs(taue200_dsODE_error);
    y3 = abs(taue800_dsODE_error);
    y4 = abs(taue60_RD_error);
    y5 = abs(taue200_RD_error);
    y6 = abs(taue800_RD_error);

    color1 = [0.2, 0.6, 1]; % Light blue for Y1
    color2 = [0.2, 0.8, 0.8]; % Aqua for Y2
    color3 = [0.6, 0.8, 1]; % Lighter blue for Y3

    color4 = [1, 0.4, 0.4]; % Light red for Y4
    color5 = [1, 0.6, 0.6]; % Pinkish red for Y5
    color6 = [1, 0.8, 0.8]; % Very light red for Y6

    s1 = shadedErrorBar(x, y1, {@mean,@std}, 'lineprops', {':','Color',color1,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;
    s2 = shadedErrorBar(x, y2, {@mean,@std}, 'lineprops', {'--','Color',color2,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;
    s3 = shadedErrorBar(x, y3, {@mean,@std}, 'lineprops', {'-','Color',color3,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;

    s4 = shadedErrorBar(x, y4, {@mean,@std}, 'lineprops', {':','Color',color4,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;
    s5 = shadedErrorBar(x, y5, {@mean,@std}, 'lineprops', {'--','Color',color5,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;
    s6 = shadedErrorBar(x, y6, {@mean,@std}, 'lineprops', {'-','Color',color6,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;

    legend({'dsODE 60','dsODE 200','dsODE 800','refractory density 60','refractory density 200','refractory density 800'}, 'Location', 'northeast','fontsize',12);
    xlim([21,24])
    ylim([0,0.6])
    set([s1.patch, s2.patch, s3.patch, s4.patch, s5.patch, s6.patch],'FaceAlpha',0.2);
    txt = xlabel('$\tau^{E}$','fontsize',16);
    set(txt, 'Interpreter', 'latex');
    grid on;
    set(gca,'GridColor',[0.9 0.9 0.9]); % Light gray grid
    set(gca,'Color','white'); % White background

    
else % FP
    figure;
    x = (1:51)*0.005+0.05-0.005;
    y1 = abs(s1000_dsODE_error);
    y2 = abs(s10000_dsODE_error);
    y4 = abs(s1000_FP_error);
    y5 = abs(s10000_FP_error);

    color1 = [0.2, 0.6, 1]; % Light blue for Y1
    color2 = [0.2, 0.8, 0.8]; % Aqua for Y2

    color4 = [1, 0.4, 0.4]; % Light red for Y4
    color5 = [1, 0.6, 0.6]; % Pinkish red for Y5

    s1 = shadedErrorBar(x, y1, {@mean,@std}, 'lineprops', {'-','Color',color1,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;
    s2 = shadedErrorBar(x, y2, {@mean,@std}, 'lineprops', {'--','Color',color2,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;

    s4 = shadedErrorBar(x, y4, {@mean,@std}, 'lineprops', {'-','Color',color4,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;
    s5 = shadedErrorBar(x, y5, {@mean,@std}, 'lineprops', {'--','Color',color5,'LineWidth',2}, 'patchSaturation', 0.1);
    hold on;



    legend({'dsODE 1000','dsODE 10000','fokker-planck 1000','fokker-planck 10000'}, 'Location', 'northeast','fontsize',12);
    txt = xlabel('$S^{EE}$','fontsize',16);
    set(txt, 'Interpreter', 'latex');

    set([s1.patch, s2.patch, s4.patch, s5.patch],'FaceAlpha',0.2);
    xlim([1,6])
    ylim([0,0.4])
    yticks(0.1:0.1:0.4);
    grid on;
    set(gca,'GridColor',[0.9 0.9 0.9]); % Light gray grid
    set(gca,'Color','white'); % White background

end
