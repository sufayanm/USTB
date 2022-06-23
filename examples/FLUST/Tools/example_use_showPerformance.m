%% 2D velocity vector estimates
% To use: 
% - Run FLUST simulation. 
% - Estimate 2D velocity vectors using estimator of choice
% - Estimates are in the form vxEst [Nx,Nz,Nreals], vzEst [Nx,Nz,Nreals]

% Insert estimates from 2D VV estimator of choice
vxEst = squeeze(vectorVel(:,:,1,:));
vzEst = squeeze(vectorVel(:,:,2,:));

% Mask accounting for valid scan region.vUseful if phantom is only partly
% covered by valid scan zone (due to steering etc.). If not set, all pixels
% within phantom region is considered

maxXVal = 0.003;
scanMask = ones(size(X));
scanMask(abs(X(:))>maxXVal) = NaN;

% Insert required fields into struct
inStruct.GT = GT; % From FLUST
inStruct.vxEst = vxEst; % From estimator
inStruct.vzEst = vzEst; % From estimator
inStruct.X = X; % From FLUST
inStruct.Z = Z; % From FLUST
inStruct.scanMask = scanMask;
inStruct.scatterDecimationFac = 10; % Controlling the number of dots in scatterplots
inStruct.SNR = 10; % SNR used in FLUST simulations
inStruct.biasPercentage = 10; % Percentage of maximum velocity in Vx/Vz direction
inStruct.stdPercentage = 10;  % Percentage of maximum velocity in Vx/Vz direction

% Run performance analysis
showPerformance_VV(inStruct);

%% Autocorrelation estimates
% Get autocorrelation estimates from FLUST realizations
Na = length(s.PSF_params.acq.alphaTx);
PRF = s.firing_rate/Na;
f_demod = s.PSF_params.trans.f0;
c = 1540;

R1 = squeeze(mean(conj(realTab(:,:,1:end-1,:,:)).*realTab(:,:,2:end,:,:),3));
vAxEst = angle(R1)*PRF*c/(4*f_demod*pi);

% Mask accounting for valid scan region
maxXVal = 0.003;
scanMask = ones(size(X));
scanMask(abs(X(:))>maxXVal) = NaN;

% Insert required fields into struct
inStruct.GT = GT;
inStruct.vAxEst = vAxEst;
inStruct.vNyq = PRF*c/(4*f_demod);
inStruct.PSF_params = s.PSF_params;
inStruct.X = X;
inStruct.Z = Z;
inStruct.scanMask = scanMask;
inStruct.scatterDecimationFac = 10;
inStruct.SNR = 10;
inStruct.biasPercentage = 500; % Percentage of Nyquist velocity
inStruct.stdPercentage = 100;  % Percentage of Nyquist velocity

% Run performance analysis
showPerformance_autocorr(inStruct);

