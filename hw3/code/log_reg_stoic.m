function [ wc, u, Rw  ] = log_reg_stoic(X, Y, nIts, nablaBool )
%This function uses stoichastic gradient descent to obtain parameters to
%the logistic regression model:
%
%p(Y = 1 | X = x) = 1./(1 + exp(-z))
%
%The only difference between this and batch gradient descent is that we use
%only one learning example for each iteration, chosen at random at each
%iteration as well.
%
%nablaBool if this is high, then nabla = 1/t, where t is the ith iteration
numExamples = size(X,1);
numFeats = size(X, 2);

%make sure that our variables are doubles
X = double(X);
Y = double(Y);

%Make w0 the mean of our features.  
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
OneT = ones(1, size(Q,1));

numSamples = 1; %floor(numExamples/100);
for ii = 1:1:nIts
    
    %Randomly sample learning examples from Q

    Qt = Q(randsample(numExamples,numSamples), :);
    
    
    %Compute the current z
    zc = Qt*wc;
    zc(zc<=zThreshLow) = zThreshLow; %Limit the range 
    zc(zc>=zThreshHigh) = zThreshHigh;
    
    %make sure to restrict the values of z
    
    %Compute nabla
    if nablaBool
        n = .001/ii;
    else
        n = .001; 
    end
    
    %compute the next set of w's
    
    
    wn = wc - n*(- Qt'*(1./(1 + exp(zc))));
    
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

