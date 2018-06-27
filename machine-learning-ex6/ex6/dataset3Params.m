function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec=[ 0.01,0.03,0.1,0.3,1,3,10,30];
sigma_vec=[ 0.01,0.03,0.1,0.3,1,3,10,30];
error=zeros(length(C_vec),length(sigma_vec));
for i=1:length(C_vec)
    C1=C_vec(i);
    for j=1:length(sigma_vec)
       sigma1=sigma_vec(j);
       model=svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
       predictions = svmPredict(model, Xval);
       error(i,j)=mean(double(predictions~=yval));
    end
end
[err,ind]=min(error(:));
[I,J]=ind2sub([length(C_vec) length(sigma_vec)],ind);
C=C_vec(I);
sigma=sigma_vec(J);




% =========================================================================

end
