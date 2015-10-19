%CS 289 - hw1.
addpath('C:\Users\Jesus\Documents\MATLAB\liblinear-2.01\matlab');
%clear



%% Problem 1 & 2
%Create a simple SVM

%load the data
my_dir = cd;
load(strcat(my_dir, '\data\digit-dataset\train.mat'));

%create a directory of where our plots will be stored
plots_dir = strcat(my_dir, '\Plots');
plots_dir_P4 = strcat(plots_dir, '\P4');
if ~isdir(plots_dir)
    mkdir(plots_dir);
end

if ~isdir(plots_dir_P4)
    mkdir(plots_dir_P4);
end

%
%separate 10k images for validation and organize them for processing
[val_images, val_labels, remain_data, remain_labels, I_val, uniq_labels] = ...
    sample_by_label(train_images, train_labels, 10e+3);


val_images = reshape(val_images, 1, [], size(val_images, 3)); %Turn images into row vectors
val_images = permute(val_images, [3 2 1]);

val_images = norm_row(val_images);  %normalize
val_images = sparse(val_images);    %make it sparse



N = [100; 200; 500; 1e+3; 2e+3; 5e+3; 10e+3];
acc_vec = zeros(numel(N), 1);
%conf_mats = cell(N,1);
h = figure('visible', 'off', 'units', 'normalized','outerposition',[0 0 1 1]);
for ii = 1:1:numel(N)
    %get a sample from the remaining training data
    [sam_data, sam_labels, remain_data1, remain_labels1, I_val1, uniq_labels] = ...
        sample_by_label(remain_data, remain_labels, N(ii));
    
    %Turn the images into row vectors to make these the features
    sam_feat_row = reshape(sam_data, 1, [], size(sam_data, 3)); %turn the vectors into rows. This is still 3-d
    sam_feat_row = permute(sam_feat_row, [3 2 1]); %This reorganizes the order of the 3rd dimension to be the first, second stay the same, etc.
    
    %normalize my matrix
    sam_feat_row = norm_row(sam_feat_row);
    
    %make the matrix sparse
    sam_feat_row = sparse(sam_feat_row);
    
    %Create the SVM
    myModel = train(sam_labels, sam_feat_row);
    
    %Test the model
    [pred_labels, accuracy, decision_values] = predict(val_labels, val_images, myModel);
    
    %store the accuracy
    acc_vec(ii) = accuracy(1);
    
    %Get the confusion matrix and save it as an image
    CM = confusionmat(val_labels, pred_labels);

    subplot(3,3,ii), imagesc(uniq_labels,uniq_labels,CM), colorbar,
    title({['CM N = ' num2str(N(ii)) ]; ...
        ['Accuracy: ' num2str(accuracy(1)) ' %']});
    
    
    
end

saveas(h, strcat(plots_dir, '\P2 - digits_CM.jpg'));
close(h);
    

%plot the accuracy vs. number of samples selected
h0 = figure('visible', 'on', 'units', 'normalized','outerposition',[0 0 1 1]);
semilogx(N, 100-acc_vec), title('P1: Error vs. Number of Training Samples'),
xlabel('log(N)'), ylabel('Error (%)'), grid('on');
saveas(h0, strcat(plots_dir, '\P1 - digits_err.jpg'));
%}


%% Problem 3
%
%Create a range of values for C
%C = [.0001 .001 .01 .1 1 10 100];
%C = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1 2 3 4 5 6 7 8 9 10];
C = .8;

%separate 10k samples for training
[train_sam_data, train_sam_labels, remain_data, remain_labels, I_val, uniq_labels] = ...
    sample_by_label(train_images, train_labels, 10e+3);

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
        
        %stack the training data along 3rd dimension and the training
        %labels along the first
        train_data = cat(3, train_data{:});     
        train_labels_jj = cat(1, train_labels_jj{:});
        
        %Turn the images into row vectors to make these the features
        train_data = reshape(train_data, 1, [], size(train_data, 3)); %turn the vectors into rows. This is still 3-d
        train_data = permute(train_data, [3 2 1]); %This reorganizes the order of the 3rd dimension to be the first, second stay the same, etc.
        
        val_data = reshape(val_data, 1, [], size(val_data, 3)); %turn the vectors into rows. This is still 3-d
        val_data = permute(val_data, [3 2 1]);
        
        %normalize the matrices
        train_data = norm_row(train_data);
        
        val_data = norm_row(val_data);
        
        %make the matrices sparse
        train_data = sparse(train_data);
        
        val_data = sparse(val_data);
        
        %Create the SVM
        myModel = train(train_labels_jj, train_data,['-c ', num2str(C(ii))]);
        
        %Test the model
        [pred_labels, accuracy, decision_values] = predict(val_labels, val_data, myModel);
        
        
        %save the error of the jth iteration
        err_vec(jj) = 100 - accuracy(1);
        
        
    end
    
    %store the mean of the errors
    mean_errs(ii) = mean(err_vec);
    
    
    
end

%plot the mean error as a function of C
%plot the accuracy vs. number of samples selected
h0 = figure('visible', 'on', 'units', 'normalized','outerposition',[0 0 1 1]);
semilogx(C, mean_errs), title('P3 - 10-fold cross-validation: Mean Error vs. C'),
xlabel('C'), ylabel('Error (%)'), grid('on');
saveas(h0, strcat(plots_dir, '\P3 - 10-fold Err vs C.jpg'));
%}

%Use the optimized value of C with our training data to create a model make
%the matrices sparse
C =.8;
train_data = reshape(train_images, 1, [], size(train_images, 3)); %Turn images into row vectors
train_data = permute(train_data, [3 2 1]);

train_data = norm_row(train_data);  %normalize
train_data = sparse(train_data);
%train_labels = double(training_labels');
myModel = train(train_labels, train_data,['-c ', num2str(C)]);

%Perform against the test set
load(strcat(my_dir, '\data\digit-dataset\test.mat'));
num_tests = size(test_images,3);
ids = (1:1:num_tests)';%Turn the images into row vectors to make these the features
test_images = reshape(test_images, 1, [], size(test_images, 3)); %turn the vectors into rows. This is still 3-d
test_images = permute(test_images, [3 2 1]);    %This reorganizes the order of the 3rd dimension to be the first, second stay the same, etc.
test_images = norm_row(test_images);            %normalize the matrices
test_images = sparse(test_images);              %make the matrices sparse
[pred_test_labels, accuracy, decision_values] = predict(zeros(num_tests,1), test_images, myModel);
out_data = dataset(ids, pred_test_labels);
out_data.Properties.VarNames = {'Id', 'Category'};
export(out_data, 'file', strcat(my_dir,'\kaggle_submission - Digits.csv'), 'Delimiter', ',');

%}

%% Problem 4
%
%load the dataset
load(strcat(my_dir, '\data\spam-dataset\spam_data.mat'));
%Create a range of values for C
%C = [.0001 .001 .01 .1 1 10 100 1e+3 1e+4];
%C = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1 2 3 4 5 6 7 8 9 10];
C = 1;

%separate 10k samples for training
%[train_sam_data, train_sam_labels, remain_data, remain_labels, I_val, uniq_labels] = ...
%    sample_by_label(train_images, train_labels, 10e+3);


%for each value of C,
k = 12;
mean_errs = zeros(1, numel(C));
for ii = 1:1:numel(C)
    
    %get the number of features
    num_feats = size(training_data,2);
    
    
    %get the unique labels
    uniq_labels = unique(training_labels);
    num_uniq_labels = numel(uniq_labels);
    jj_vec = 1:1:num_feats;
    
    feat_errs = zeros(1, num_feats);
    for jj = jj_vec
        
        %grab one of the features only to see how it predicts
        new_train_data = training_data(:,jj);
        
        %Randomly split the data into k parts.  Make sure to turn the
        %training_labels row vector into column form and from int to double
        [split_data_cell, split_labels_cell, rand_inds_cell] = rand_split_data(new_train_data, double(training_labels'), 1, k);
        
        err_vec = zeros(1, k);
        
        
        
        %preallocate a confusion matrix
        
        CM = zeros(num_uniq_labels, num_uniq_labels);
        
        for kk = 1:1:k
            %grab the jth dataset and labels to separate for validation
            val_data = split_data_cell{kk};
            val_labels = split_labels_cell{kk};
            
            %grab the rest for use as the training data
            ind_vec = 1:1:k;
            ind_vec(kk) = [];
            train_data = split_data_cell(ind_vec);
            train_labels_jj = split_labels_cell(ind_vec);
            
            %
            
            %stack the training data along 1st dimension and the training
            %labels along the first
            train_data = cat(1, train_data{:});
            train_labels_jj = cat(1, train_labels_jj{:});
            
            %{
        
        %Turn the images into row vectors to make these the features
        train_data = reshape(train_data, 1, [], size(train_data, 3)); %turn the vectors into rows. This is still 3-d
        train_data = permute(train_data, [3 2 1]); %This reorganizes the order of the 3rd dimension to be the first, second stay the same, etc.
        
        val_data = reshape(val_data, 1, [], size(val_data, 3)); %turn the vectors into rows. This is still 3-d
        val_data = permute(val_data, [3 2 1]);
            %}
            
            %normalize the matrices
            %train_data = norm_row(train_data);
            
            %val_data = norm_row(val_data);
            
            %make the matrices sparse
            train_data = sparse(train_data);
            
            val_data = sparse(val_data);
            
            %Create the SVM
            myModel = train(train_labels_jj, train_data,['-c ', num2str(C(ii))]);
            
            %Test the model
            [pred_labels, accuracy, decision_values] = predict(val_labels, val_data, myModel);
            
            
            %save the error of the jth iteration
            err_vec(kk) = 100 - accuracy(1);
            
            %Compute the confusion matrix, and sum it to previous estimated CM
            CM = CM + confusionmat(val_labels, pred_labels);
            
        end
        
        %store the mean of the errors
        mean_err = mean(err_vec);
        mean_errs(ii) = mean_err;
        feat_errs(jj) = mean_err;
        
        %Plot the average confusion matrix
        CM = CM/k;
        h = figure('visible', 'off', 'units', 'normalized','outerposition',[0 0 1 1]);
        imagesc(uniq_labels,uniq_labels,CM), colorbar,
        title({['Average CM for Feat # = ' num2str(jj) ]; ...
            ['Mean Error: ' num2str(mean_err) ' %']});
        
        saveas(h, strcat(plots_dir_P4, '\P4 - CM - Feat_num ', num2str(jj), '.jpg'));
        close(h);
        
        
    end
end

%for the number of features, estimate the average error for each feature
%make a barplot of the kkth feature


%plot the mean error as a function of of feature number
%plot the accuracy vs. number of samples selected
h0 = figure('visible', 'on', 'units', 'normalized','outerposition',[0 0 1 1]);
bar(jj_vec, feat_errs), title('P4 - Mean Error vs. Feature #'),
xlabel('Feature #'), ylabel('Error (%)'), grid('on');
saveas(h0, strcat(plots_dir_P4, '\P4 - Mean Error vs. Feature No.jpg'));
%}



%
%load the dataset
load(strcat(my_dir, '\data\spam-dataset\spam_data.mat'));

%{
%Remove the features that scored above 70% error rate
%rem_feats = find(feat_errs > 60);
%rem_feats = [8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32];
rem_feats = [];
new_training_data = training_data;
new_training_data(:, rem_feats) = [];

%Create a range of values for C
%C = [1e-6 1e-5 1e-4 .001 .01 .1 1 10 100];
%C = [.000001 .00001 .0001 .001 .01 .1 1 10 100];
%C = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1 2 3 4 5 6 7 8 9 10];
%C = .00001;
C = 1;

k = 12;
mean_errs = zeros(1, numel(C));
for ii = 1:1:numel(C)
    
    %Remove selected features

    
    
    
    %Randomly split the data into k parts.  Make sure to turn the
    %training_labels row vector into column form and from int to double
    [split_data_cell, split_labels_cell, rand_inds_cell] = rand_split_data(new_training_data, double(training_labels'), 1, k);
    
    err_vec = zeros(1, k);
    

    
    %preallocate a confusion matrix
    uniq_labels = unique(training_labels);
    num_uniq_labels = numel(uniq_labels);
    CM = zeros(num_uniq_labels, num_uniq_labels);
    
    %get the current C
    currC = C(ii);
    
    for jj = 1:1:k
        %grab the jth dataset and labels to separate for validation
        rand_inds = rand_inds_cell{jj};
        val_data = split_data_cell{jj};
        val_labels = split_labels_cell{jj};
        
        %grab the rest for use as the training data
        ind_vec = 1:1:k;
        ind_vec(jj) = [];
        train_data = split_data_cell(ind_vec);
        train_labels_jj = split_labels_cell(ind_vec);
        
        %
        
        %stack the training data along 1st dimension and the training
        %labels along the first
        train_data = cat(1, train_data{:});     
        train_labels_jj = cat(1, train_labels_jj{:});
        
        %{
        
        %Turn the images into row vectors to make these the features
        train_data = reshape(train_data, 1, [], size(train_data, 3)); %turn the vectors into rows. This is still 3-d
        train_data = permute(train_data, [3 2 1]); %This reorganizes the order of the 3rd dimension to be the first, second stay the same, etc.
        
        val_data = reshape(val_data, 1, [], size(val_data, 3)); %turn the vectors into rows. This is still 3-d
        val_data = permute(val_data, [3 2 1]);
        %}
        
        %normalize the matrices
        %train_data = norm_row(train_data);
        
        %val_data = norm_row(val_data);
        
        %make the matrices sparse
        train_data = sparse(train_data);
        
        val_data = sparse(val_data);
        
        %Create the SVM
        
        myModel = train(train_labels_jj, train_data,['-c ', num2str(currC)]);
        
        %Test the model
        [pred_labels, accuracy, decision_values] = predict(val_labels, val_data, myModel);
        
        
        %save the error of the jth iteration
        err_vec(jj) = 100 - accuracy(1);
        
        %Compute the confusion matrix, and sum it to previous estimated CM
        CM = CM + confusionmat(val_labels, pred_labels);
        
    end
    
    %store the mean of the errors
    mean_errs(ii) = mean(err_vec);
    
    %Plot the average confusion matrix
    CM = CM/k;
    h = figure('visible', 'off', 'units', 'normalized','outerposition',[0 0 1 1]);
    imagesc(uniq_labels,uniq_labels,CM), colorbar,
        title({['Average CM for C = ' num2str(currC) ]; ...
            ['Mean Error: ' num2str(mean_errs(ii)) ' %']});
    
    saveas(h, strcat(plots_dir_P4, '\P4 - CM - C = ', num2str(currC), '.jpg')); 
    close(h);
end

%for the number of features, estimate the average error for each feature
%make a barplot of the 

%
%plot the mean error as a function of C
%plot the accuracy vs. number of samples selected
%
h0 = figure('visible', 'on', 'units', 'normalized','outerposition',[0 0 1 1]);
semilogx(C, mean_errs), title('P4 - Mean Error vs. C'),
xlabel('C'), ylabel('Error (%)'), grid('on');
saveas(h0, strcat(plots_dir_P4, '\P4 - 10-fold Err vs C.jpg'));
%}
%Use the optimized value of C with our training data to create a model make
%the matrices sparse
C =1;
train_data = sparse(training_data);
train_labels = double(training_labels');
myModel = train(train_labels, train_data,['-c ', num2str(C)]);

%Perform against the test set
num_tests = size(test_data,1);
ids = (1:1:num_tests)';
test_data = sparse(test_data);              %make the matrices sparse
[pred_test_labels, accuracy, decision_values] = predict(zeros(num_tests,1), test_data, myModel);
out_data = dataset(ids, pred_test_labels);
out_data.Properties.VarNames = {'Id', 'Category'};
export(out_data, 'file', strcat(my_dir,'\kaggle_submission - Spam-ham.csv'), 'Delimiter', ',');

%close(h0);
%}

%}

