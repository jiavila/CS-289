%CS 289 - HW3, P1
plotsDir = 'C:\Users\Jesus\Documents\School\2015 - Fall\CS 189 - Machine Learning\hw3\plots';

%Load the data (Xtrain, Xvalidate, Ytrain, Yvalidate)
load('C:\Users\Jesus\Documents\School\2015 - Fall\CS 189 - Machine Learning\hw3\data\housing_data.mat')


%% 1.1 Linear Regression

%assign the data to our equation variables
a = ones(numel(Ytrain),1);
X = [a Xtrain];
y = Ytrain;

%compute the weights using Equation 1.1
w = inv(X'*X)*X'*y;

%% 1.2 Residual sum-of-squares (RSS)

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
title({'Predicted Median Home Value'; ['Range: [ ' num2str(ypRange) ' ]']; ... 
        ['RSS = ' num2str(RSS, 4)]});
xlabel('Predicted Home No.');
ylabel('Median Home Value');

%% 1.3 Plot w as function of its index
subplot(3,1,2), plot(w(2:end)); 
title('Linear regression weights/coefficients (excluding constant)')
xlabel('Coefficient Index');
ylabel('Weight Values');

%% Plot a histogram of the residuals (f(x) - y)
rezids = yp - yv; 
nBins = numel(rezids)/10;

%get the range of the residuals for plotting
rezMax = max(rezids);
rezMin = min(rezids);
rezRangeDiff = rezMax - rezMin;

%Create a vector for the x dimension of the histogram
xHistStep = rezRangeDiff/(nBins-1);
xHist = rezMin:xHistStep:rezMax;

[myHist, histCen] = hist(rezids, nBins);
subplot(3,1,3), plot(xHist, myHist);
title(['Histogram of Residuals (f(x) - y), No. Bins: ' num2str(nBins)]);
xlabel('Residual Values');
ylabel('No. of Residuals');


saveas(h, [plotsDir '\P1.jpg']);




