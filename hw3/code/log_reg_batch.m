function [ wc, u, Rw ] = log_reg_batch(X, Y, nIts )
%This function uses batch gradient descent to obtain parameters to the
%logistic regression model: 
%
%p(Y = 1 | X = x) = 1./(1 + exp(-z))

numSamples = size(X,1);
numFeats = size(X, 2);

%make sure that our variables are doubles
X = double(X);
Y = double(Y);

%Make w0 the mean of our features
w0 = mean(X)';%rand(numFeats,1); %

%Define our variables
Q = diag(Y)*X;

%compute u0
u0 = 1./(1 + exp(-Q*w0));


wc = w0;

zThreshLow = -100;
zThreshHigh = 200;

%Preallocate variables
Rw = zeros(nIts,1);
OneT = ones(1, numSamples);
for ii = 1:1:nIts
    
    %Compute the current z
    zc = Q*wc;
    zc(zc<=zThreshLow) = zThreshLow; %Limit the range 
    zc(zc>=zThreshHigh) = zThreshHigh;
    
    %make sure to restrict the values of z
    
    %compute the next set of w's
    n = 1; %nabla
    wn = wc - n*(- Q'*(1./(1 + exp(zc))));
    
    %Compute the probabilities u
    zn = Q*wn;
    zn(zn<=zThreshLow) = zThreshLow;
    zn(zn>=zThreshHigh) = zThreshHigh;
    
    u = 1./(1 + exp(-zn));
    
    %{
    if (ii == 1) || (ii == 2)
        %display the weights w
       disp(['w' num2str(ii) ' = ' ])
       disp(num2str(wn));
       disp(' ');
       
       %display the probabilities u
       disp(['u' num2str(ii) ' = ' ]);
       disp(num2str(u));
       disp(' ');
    end
    %}
    
    %Compute the Risk for each iteration and store in Rw
    Rw(ii) = OneT*log(1 + exp(-zn));
    
    %set wc
    wc = wn;
end


end

