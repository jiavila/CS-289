%Problem 1: Centering and Ridge Regression

%get the current directory
currDir = cd;
I = strfind(currDir, '\');
I = I(end);
parentDir = currDir(1:I-1);
plotsDir = [parentDir '\plots'];
dataDir = [parentDir '\data'];

%assign the plots directory
if ~isdir(plotsDir)
    mkdir(plotsDir);
end

%Load the data (Xtrain, Xvalidate, Ytrain, Yvalidate)
load([dataDir '\housing_data']);






%% 1.1 Ridge Regression

%assign the data to our equation variables and Center our data
n = size(Xtrain,1);
%Xtrain = [ones(n,1) Xtrain]; %add a constant term
X = center_data(Xtrain); 
y = Ytrain;


L = 1;

%compute the weights 
w0 = 1/n*sum(y);
w = inv(X'*X + L)*X'*y;


%% 1.2 Cross-validation and Residual sum-of-squares (RSS)
%for each value of L,
k = 10;
L = [1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4 1e5];
mean_errs = zeros(1, numel(L));
for ii = 1:1:numel(L)
    %
    disp('************************************************');
    disp(['Training for L = ' num2str(L(ii))]);
    %Randomly split the data into k parts
    [split_data_cell, split_labels_cell, rand_inds_cell] = rand_split_data(Xtrain, Ytrain, 1, k);
    
    err_vec = zeros(1, k);
    for jj = 1:1:k
        %grab the jth dataset and labels to separate for validation
        XVal = center_data(split_data_cell{jj});
        yV = split_labels_cell{jj};
        
        %grab the rest for use as the training data
        ind_vec = 1:1:k;
        ind_vec(jj) = [];
        X = split_data_cell(ind_vec); %get rest of data (class cell)
        X = center_data(cat(1,X{:})); %center and concatenate along 1st dimension
        y = split_labels_cell(ind_vec);
        y = cat(1,y{:}); %concatenate along first dimension
        
        %compute the weights
        w0 = 1/n*sum(y);
        w = inv(X'*X + L(ii))*X'*y;
        
    
        %compute the predicted value
        yp = XVal*w + w0;
        
        %compute the RSS
        RSS = (yV - yp)'*(yV-yp);
        
        %save the error of the jth iteration
        err_vec(jj) = RSS;
        
        
    end
    
    %store the mean of the errors
    mean_errs(ii) = mean(err_vec);
    
    
    
end


%plot the mean error as a function of L
%plot the accuracy vs. number of samples selected
h0 = figure('visible', 'on', 'units', 'normalized','outerposition',[0 0 1 1]);
semilogx(L, mean_errs), title('P1bii 10-fold cross-validation: Mean Error vs. Lambda'),
xlabel('L'), ylabel('RSS'), grid('on');
saveas(h0, strcat(plotsDir, '\P1bii - 10-fold Err vs Lambda.jpg'));
%}

%%%%% Get the RSS for L = 1.0x10^2
%get our 
X = center_data(Xtrain); 
y = Ytrain;
yV = Yvalidate;
XVal = Xvalidate;

L = 10^2;

%compute the weights 
w0 = 1/n*sum(y);
w = inv(X'*X + L)*X'*y;



%compute the predicted value
yp = XVal*w + w0;

%compute the RSS
RSS = (yV - yp)'*(yV-yp);



%plot the predicted values and get the range
ypMax = max(yp);
ypMin = min(yp);
ypRange = [ypMin ypMax];
disp(ypRange);
h = figure('visible', 'on','units', 'normalized','outerposition',[0 0 1 1]);
subplot(2,1,1), plot(yp); 
title({'Predicted Median Home Value'; ['Range: [ ' num2str(ypRange) ' ]']; ...
    ['L = ' num2str(L) ', RSS = ' num2str(RSS, 4)]});
xlabel('Predicted Home No.');
ylabel('Median Home Value');

%% 1.3 Plot w as function of its index
subplot(2,1,2), plot(w); 
title('Ridge regression weights/coefficients (excluding constant)')
xlabel('Coefficient Index');
ylabel('Weight Values');
saveas(h, [plotsDir '\1biii Ridge Regression Coefficients.jpg']);
%}


