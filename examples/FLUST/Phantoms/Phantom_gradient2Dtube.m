function [flowField, GT, p] = Phantom_gradient2Dtube( setup, X, Z ) 

%% small 2D tube phantom, run and check signal integrity in the middle of the tube
p.btf = 60;
p.npoints = 10;
p.flowlength = 0.0024; %0.005; %0.03; %0.005;
p.tubedepth = 0.015; %0.03;
p.depthstep = 0.0001; %0.00015; %lambda/2 for 5 MHz
p.noFlowLines = 3;
p.vel_1 = 0.4;
p.vel_2 = 2;

fields = fieldnames(setup);
for k=1:size(fields,1)
    if(isfield(p,fields{k}))
        p.(fields{k}) = setup.(fields{k});
    else
        disp([ fields{k} ' is not a valid parameter for this phantom type']);
    end
end


veltab = linspace( p.vel_1, p.vel_2, p.noFlowLines);

depthtab = linspace( p.tubedepth-p.depthstep*(p.noFlowLines-1)/2, p.tubedepth+p.depthstep*(p.noFlowLines-1)/2, p.noFlowLines);
for kk = 1:p.noFlowLines
    time_max = p.flowlength/veltab(kk);
    currtubedepth = depthtab(kk);
    flowField(kk).timetab = linspace(0, time_max, p.npoints);
    flowField(kk).postab = veltab(kk)*(flowField(kk).timetab-time_max/2).*[sind(p.btf); 0; cosd(p.btf)]+[0; 0; currtubedepth];
    flowField(kk).timetab = flowField(kk).timetab.'; 
    flowField(kk).postab = flowField(kk).postab.';
end
if nargin > 1
    projZ = Z-X/tand(p.btf);
    projDist = X/sind(p.btf);
    GT = interp1( depthtab, veltab, projZ);
    GT( abs( projDist) > p.flowlength/2 ) = NaN;
end