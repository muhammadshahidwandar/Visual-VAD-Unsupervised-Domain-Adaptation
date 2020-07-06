%% VAD_Domain_Adaptation - This matlab script implements unsuperwised 
%   domain adaptation using 2 layer auto encoder and PCA based subspace 
%   alignment proposed in [1] and [2]
%
% INPUT
% Featurs : d-by-n matrix representing the d dimentional fc features vector 
% extracted from Resnet50 for n number of images,
% label : Binary VAD label (0: not Speaking, 1: Speaking)
%
%
% OUTPUT
% Mat_AE2_SAPCA_SVM : A mat file containing cross validation prediction f1score with 
% different parametrs will be saved 

% Muhammad Shahid
% Copyright 2020 Muhammad Shahid [ shahid.muhammad-at-iit.it ]
% Please, email me if you have any question.

% [1] C. Beyan, M. Shahid and V. Murino,
%      "RealVAD: A Real-world Dataset and A Method for Voice Activity Detection by Body Motion Analysis"
%      in IEEE Transactions on Multimedia, IN PRESS., 2020
%[2] M. Shahid, C. Beyan and V. Murino
%      Voice Activity Detection by Upper Body Motion Analysis and Unsupervised Domain Adaptation
%      in ICCVW 2019
%
clear all
close all
clc
test =[1 2 3 4 5];% 5 cross validation folds
C = [0.0001,0.001,.01,.1,1,10,100,1000];% Best C values search for SVM
EignV_d = [5,10,15,20,25,30];%Number of Eigen Vectors for PCA based SubSpace part
Mat{1,1}='Svm_C';%C values being used
Mat{1,2}='Eign_D';% Eigen value being used 

%%%%%%%%%%%%%Validation Results Columns%%%%%%%%%%%%%%%
Mat{1,3}='ValPerson1';
Mat{1,4}='ValPerson2';
Mat{1,5}='ValPerson3';
Mat{1,6}='ValPerson4';
Mat{1,7}='ValPerson5';
Mat{1,8}='ValAvg_F_Scor';
m= 8
%%%%%%%%%%%%%Test Results Columns%%%%%%%%%%%%%%%%%%
Mat{1,1+m}='TestPerson1';
Mat{1,2+m}='TestPerson2';
Mat{1,3+m}='TestPerson3';
Mat{1,4+m}='TestPerson4';
Mat{1,5+m}='TestPerson5';
Mat{1,6+m}='Avg_F_Scor';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for t=1:length(test)%%%%Leave one person out cross validation loop
            Row = 1;
            BaseFolderPath = './CNNfcFeatures/Fold'; % Base folder containing Leave One out Subfolders
            subFolder = strcat(BaseFolderPath,num2str(test(t)),'/');
            Mat_Files = dir(fullfile(subFolder,'*.mat')); 
            Xtrain = []; Xval = []; Xtest = []; Ytrain =[]; Yval =[]; Ytest =[];
            Total_MatFiles = length( Mat_Files )
            for i=1:Total_MatFiles
                fulPath = strcat(subFolder,Mat_Files(i).name);
                load(fulPath); %matfiles containing resnet50 features(Featurs) and VAD lab(label)    
                FeatureTrain=Featurs;
                LabelTrain    = label;
                MatLen = length(LabelTrain);
                VLdSampl = ceil(MatLen*1/10); % 10 percent training data is considerd as validation set  
                Vld_RandIndx = randperm(MatLen,VLdSampl);
                %%%%%%%%%Separating out Training, Test and Validation Data part%%
                if(i==test(t))
                        Xtest=[Xtest,FeatureTrain'];
                        Ytest=[Ytest,LabelTrain'];
                else
                    %enable this for loop to take validation data from training set
                    %for j=1:MatLen 
                    %    if(find(Vld_RandIndx==j)>0) %%% comment this if statment when you are using full training Data/no validation
                     %       Xval=[Xval,FeatureTrain(j,:)'];
                      %      Yval=[Yval,LabelTrain(j)'];
                      %  else
                         Xtrain=[Xtrain,FeatureTrain'];%(j,:)'];
                         Ytrain=[Ytrain,LabelTrain'];%(j)'];
                       %end
                    %end
                end
            end
			%%%%Select your best parameters values using Validation data as a target set,
			%%%%Afterward train using complete training set with test data as a target set
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Auto Encoder%%%%%%%%%%%%%%%%%%%
            Tar_len = length(Ytest);%Yval);  
            Src_len = length(Ytrain);
            randIdx =  randi(Src_len,Tar_len,1);
            Target = Xtest;%Xval;
            Target_lbl = Ytest;%Yval;%

            Source_lbl = Ytrain(randIdx);

            Source = Xtrain(:,randIdx);
            hiddenSize1 = 512;
            hiddenSize2A = 128;
            xTrainfc1 = [Source, Target];
			%%%%%%%%%%%%%%%%%%auto Encoder 1%%%%%%%%%%%%%%%%%%%%
            autoenc1 = trainAutoencoder(xTrainfc1,hiddenSize1, ...
                'MaxEpochs',300, ...
                'L2WeightRegularization',0.0020, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',0.15, ...
                'ScaleData', false);
            TrainFcEncod1 = encode(autoenc1,Xtrain);
            TargetFcEncod1 = encode(autoenc1,Xtest); %Xval
            FullTrainfc1 = encode(autoenc1,xTrainfc1);
            %%%%%%%%%%%%%%%%%%auto Encoder 2%%%%%%%%%%%%%%%%%%%%
           Sparsty = [0.40];%different values for sparsity ratio  
            for LS = 1 :size(Sparsty,2)%
                VarSparty = Sparsty(LS)
                hiddenSize2 = hiddenSize2A;
            autoenc2 = trainAutoencoder(FullTrainfc1,hiddenSize2, ...
                'MaxEpochs',300, ...
                'L2WeightRegularization',0.0020, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',VarSparty,...
                'ScaleData', false);
			TrainFcEncod2 = encode(autoenc2,TrainFcEncod1);
            TargetFcEncod2 = encode(autoenc2,TargetFcEncod1);
            FullTrainfc2 = encode(autoenc2,FullTrainfc1);
            
%%%%%%%%Subspace Aligment%%%%%%%%%%%%%%%%%%%%%%%%%
            SourceFc   =  TrainFcEncod2';
			SourceLbl  =  Ytrain;
		    TargetFc   =  TargetFcEncod2';
			TargetLbl  =  Target_lbl;
    
%%%%%%%%%%%%%%%%%%%%SubSpace Alignment Part%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%SubSpce Alignment using PCA based%%%%%%%%%%%%%%%
%%% The Source code for Subspace Alignment part is taken from https://basurafernando.github.io/DA_SA/
            [SourcePCA,~,~] = princomp(SourceFc);
            [TargetPCA,~,~] = princomp(TargetFc);
           
%%%%%%%%%%%%%%%%%%%%%%%%%


              for j = 1:length(EignV_d) %% enabel this loop for best parameter search
                  for k = 1 :size(C,2)
                       C_val = C(k); 
                       subspace_dim_d = EignV_d(j);
                       Row = Row+1;
                       Mat{Row,1} = C_val;
            %%%%%%%%%%%%%%%%%%%%%%Apply Subspace Alignment Part%%%%%%%%
                    total = length(SourceLbl);
                    cls1  = sum(SourceLbl(:));
                    cls0  = total-cls1;
                    cls1Prcnt = cls1/total; % class 1 ratio
                    cls0Prcnt = cls0/total; % class 0 ratio
                    %%%%%%%%%%%%%%%%Target Subspace Aligned Component%%%%
                    Xs = SourcePCA(:,1:subspace_dim_d);
                    Xt = TargetPCA(:,1:subspace_dim_d);
                    Source_SA  = SourceFc*(Xs*Xs'*Xt);
                    Target_SA  = TargetFc*Xt; 
                    %%%%%%%%%%%%%%%%%%using SVM from Libsvm library%%%%%%%%%%
                    model = svmtrain(SourceLbl', Source_SA, sprintf('-t 0 -c %d -q -wi %d %d',C_val,cls1Prcnt, cls0Prcnt));
                    y_pred_targt = svmpredict(TargetLbl', Target_SA, model);%labels are input just as reference to measure performance
                    Con_val = confusionmat(TargetLbl,y_pred_targt');
                    %%%%%%%%%%%validation Confusion%%%%%%%%%%
                    TP = Con_val(2,2);
                    TN = Con_val(1,1);
                    FP  = Con_val(1,2);
                    FN = Con_val(2,1);
                    PRCN = TP/(TP+FP);
                    RCAL  = TP/(TP+FN);
                    F_score_val = (2*PRCN*RCAL)/(PRCN+RCAL);
                    Mat{Row,2+t} = F_score_val;          
                   %%%%%%%%%%%%%%%%%%%%%%%Test Data Confusion Matrix%%%%%                    
                    Con = confusionmat(Ytest,y_pred_targt');
                    TP = Con(2,2);
                    TN = Con(1,1);
                    FP  = Con(1,2);
                    FN = Con(2,1);
                    PRCN = TP/(TP+FP);
                    RCAL  = TP/(TP+FN);
                    F_score = (2*PRCN*RCAL)/(PRCN+RCAL);
                    Mat{Row,m+t} = F_score;
              end
              end
            end
       end


save Mat_AE2_SAPCA_SVM.mat Mat