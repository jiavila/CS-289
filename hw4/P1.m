%Problem 1: Centering and Ridge Regression

%get the current directory
currDir = cd;
plotsDir = [currDir '\plots'];
dataDir = [currDir '\data'];

%assign the plots directory
if ~isdir(plotsDir)
    mkdir(plotsDir);
end

%Load the data (Xtrain, Xvalidate, Ytrain, Yvalidate)
load([dataDir '\housing_data']);






%% 1.1 Ridge Regression

%assign the data to our equation variables and Center our data
n = size(X,1);
X = center_data([ones(n,1) Xtrain]); %add a constant term
y = Ytrain;


L = 1;

%compute the weights 
w0 = 1/n*sum(y);
w = inv(X'*X + L)*X'*y;

%% 1.2 Cross-validation and Residual sum-of-squares (RSS)
%for each value of C,
k = 10;
mean_errs = zeros(1, numel(C));
for ii = 1:1:numel(C)
    %
    disp('************************************************');
    disp(['Training for C = ' num2str(C(ii))]);
    %Randomly split the data into k parts
    [split_data_cell, split_labels_cell, rand_inds_cell] = rand_split_data(train_sam_data, train_sam_labels, 3, k);
    
    err_vec = zeros(1, k);
    for jj = 1:1:k
        %grab the jth dataset and labels to separate for validation
        val_data = split_data_cell{jj};
        val_labels = split_labels_cell{jj};
        
        %grab the rest for use as the training data
        ind_vec = 1:1:k;
        ind_vec(jj) = [];
        train_data = split_data_cell(ind_vec);
        train_labels_jj = split_labels_cell(ind_vec);
        
        %Test the model
        [pred_labels, accuracy, decision_values] = predict(val_labels, val_data, myModel);
        
        %save the error of the jth iteration
        err_vec(jj) = 100 - accuracy(1);
        
        
    end
    
    %store the mean of the errors
    mean_errs(ii) = mean(err_vec);
    
    
    
end



%assign data to our equation variables
b = ones(numel(Yvalidate),1);
Xv = [b Xvalidate]; %
yv = Yvalidate;

%compute the predicted value 
yp = Xv*w;

%compute the RSS
RSS = (yv - yp)'*(yv-yp);

%plot the predicted values and get the range
ypMax = max(yp);
ypMin = min(yp);
ypRange = [ypMin ypMax];
disp(ypRange);
h = figure('visible', 'on','units', 'normalized','outerposition',[0 0 1 1]);
subplot(3,1,1), plot(yp); 
title({'Predicted Median Home Value'; ['Range: [ ' num2str(ypRange) ' ]']});
xlabel('Predicted Home No.');
ylabel('Median Home Value');

%% 1.3 Plot w as function of its index
subplot(3,1,2), plot(w(2:end)); 
title('Linear regression weights/coefficients (excluding constant)')
xlabel('Coefficient Index');
ylabel('Weight Values');