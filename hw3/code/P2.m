%% 2.3 (a)

%Define our variables
X = [0 3 1; 1 3 1; 0 1 1; 1 1 1]; %make sure to append a third dimension to X
y = [1; 1; -1; -1];
w0 = [-2; 1; 0];
Q = diag(y)*X;

%compute u0
u0 = 1./(1 + exp(-Q*w0));
disp('u0 = ');
disp(u0);

wc = w0;

for ii = 1:1:10
    %compute the next set of w's
    n = 1; %nabla
    wn = wc - n*(- Q'*(1./(1 + exp(Q*wc))));
    
    %Compute the probabilities u
    u = 1./(1 + exp(-Q*wn));
    
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
    
    
    
    %set wc
    wc = wn;
end