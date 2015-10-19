%CS 289 - HW3, P3
plotsDir = 'C:\Users\Jesus\Documents\School\2015 - Fall\CS 189 - Machine Learning\hw3\plots';


%Load the data (Xtrain, Xtest, Ytrain)
load('C:\Users\Jesus\Documents\School\2015 - Fall\CS 189 - Machine Learning\hw3\data\spam.mat')

[mX, nX] = size(Xtrain);

%Standardize each column so they each have mean 0 and unit variace.  This
%is the same thing as getting the "z" value, which can be computed as
%follows 
%
%       zi = (xi - xm)/SD, 
%                           where xi is the ith sample, xm is the mean, and
%                           SD is the standard deviation
Xm = repmat(mean(Xtrain), mX,1);
Xsd = repmat(std(Xtrain), mX,1);
Xz = (Xtrain - Xm)./Xsd;

%Transform the features using x(i,j) <==   log(x(i,j) + 1)
Xlog = log(Xtrain + 1);

% Binarize the features x(i,j) <==  I(x(i,j) > 0) I denotes an indicator
% variable.  Let's make the condition the mean of each feature.  
XB = Xtrain;
XB(Xtrain>= Xm) = 1;
XB(Xtrain<Xm) = 0;



Xcell = {Xz; Xlog; XB};
RWcell = cell(size(Xcell));

%% 3.1
%{
Y = Ytrain;


for ii = 1:1:numel(Xcell);
    X = Xcell{ii};
    
    nIts = 2000;
    [ wc, u, RWcell{ii} ] = log_reg_batch(X, Y, nIts );

end

inds = 1:1:nIts;
h = figure('visible', 'on','units', 'normalized','outerposition',[0 0 1 1]);
plot(inds, RWcell{1}, inds, RWcell{2}, inds, RWcell{3}),
title('Risk for Batch Gradient Descent Logistic Regression');
xlabel('ith Iteration');
ylabel('Rw value');
legend('Normalized', 'Log Transformed', 'Binarized');
saveas(h, [plotsDir '\BatchGradDescent.jpg']);
%}


%% 3.2
%
Y = Ytrain;
nablaBool = false;

for ii = 1:1:numel(Xcell);
    X = Xcell{ii};
    
    nIts = 2000;
    [ wc, u, Rwtemp ] = log_reg_stoic(X, Y, nIts, nablaBool);
    RWcell{ii} = Rwtemp;

end

inds = 1:1:nIts;
h = figure('visible', 'on','units', 'normalized','outerposition',[0 0 1 1]);
plot(inds, RWcell{1}, inds, RWcell{2}, inds, RWcell{3}),
title('Risk for Stoichastic Gradient Descent Logistic Regression');
xlabel('ith Iteration');
ylabel('Rw value');
legend('Normalized', 'Log Transformed', 'Binarized');
saveas(h, [plotsDir '\StoichGradDescent.jpg']);
%}

%% 3.3 
Y = Ytrain;
nablaBool = true;

for ii = 1:1:numel(Xcell);
    X = Xcell{ii};
    
    nIts = 2000;
    [ wc, u, Rwtemp ] = log_reg_stoic(X, Y, nIts, nablaBool);
    RWcell{ii} = Rwtemp;

end

inds = 1:1:nIts;
h = figure('visible', 'on','units', 'normalized','outerposition',[0 0 1 1]);
plot(inds, RWcell{1}, inds, RWcell{2}, inds, RWcell{3}),
title('Risk for Stoichastic Gradient Descent Logistic Regression');
xlabel('ith Iteration');
ylabel('Rw value');
legend('Normalized', 'Log Transformed', 'Binarized');
saveas(h, [plotsDir '\StoichNabCorrGradDescent.jpg']);




