%function [] = predict()
    clear all; close all;
    
%parameters:
    dataset = 'aerosol_full'; %fixed
    %parameters of the randomization in the experiments:
        CV = 5; %CV-fold cross validation: (CV-2),1,1 part used for training, validation, testing. 
        v_v = [1:1]; %[1:10]; indices of the random runs; number of random runs = length(v_v); '_v': vector
    %distribution embedding:
        cost_name = 'expected'; %name of the used embedding
            kernel = 'Cauchy'; %'RBF', 'exponential', 'Cauchy', 'student', 'Matern3p2', 'Matern5p2','poly2', 'poly3','ratquadr', 'invmquadr'; see 'Kexpected_initialization.m' if cost_name = 'expected'
            %kernel parameter:
                base_kp = 2; %fine-grainedness of the parameter scan; smaller base_kp = finer scan, 'kp': kernel parameter
                kp_v = base_kp.^[2:4]; %candidate kernel parameters ('RBF widths'); '_v': vector
    %regularization parameter (lambda):
        base_rp = 2;  %fine-grainedness of the parameter scan; smaller base_rp = finer scan; 'rp': regularization parameter
        rp_v = base_rp.^[-35:-3]; %candidate regularization parameters (lambda)
    %experiment generation, precomputing, plot:
        load_generated_experiment = 1; %if you would like to generate a new experiment (and study the efficiency of multiply kernels, ...), then load_generated_experiment = 0; else you just load the already generated dataset (=1).
        load_precomputed_Gram_matrices = 1; %0/1; if you have already precomputed the Gram matrices set precompute_Gram_matrices = 1; else =0.
        plot_needed = 1; %plot the results: 0/1
    
%matlabpool open; %use this line with 'Matlab: parallel computing'
%-------------------------------
%generate dataset:
   if ~load_generated_experiment %If you are experimenting with multiple kernels, to be able to study the efficiency of different kernels under the same conditions, call this generating function only _once_:
        [X,Y] = generate_AOD_dataset(dataset,v_v);
   else %if you have already generated the dataset just load 'X' and 'Y'):
        FN = strcat(dataset,'_dataset.mat');
        load(FN,'X','Y');
   end
    
 %create Gram matrices (given the AOD random runs, we precompute the Gram matrices):
    if ~load_precomputed_Gram_matrices
        precomputation_of_Gram_matrices_k(cost_name,kernel,kp_v,base_kp,dataset,X);%the function supports 'Matlab: parallel computing' if you activate 'parfor' in it
    end    
    
 %load idx_train,idx_val,idx_test-s:
    %indices = cross_validation_indices(length(Y),CV);
    indices = prediction_indices(980, length(Y), CV);
    
%for v = v_v
    v = 1;
    %p:
        % Generate index (in good order)
        %p = 1:length(Y);
        %FN = strcat('aerosol_index.mat');
        %save(FN,'p')
        
        FN = 'aerosol_index.mat';
        load(FN,'p');
        
    %if nCV = 1, divide first three folds in train,
    %fourth validation and last test
    nCV = 1;
    %for nCV = 1 : CV
        %idx_train,idx_val,idx_test): nCV-dependent, see 'p'
            idx_train = p(indices{nCV}.train);
            idx_val = p(indices{nCV}.val);
            idx_test = p(indices{nCV}.test);
        %output values (Y_train, Y_val, Y_test), number of training bags (L_train):
            Y_train = Y(idx_train);
            Y_val = Y(idx_val);
            Y_test = Y(idx_test); 
            L_train = length(idx_train);  
        %validation surface, test surface:
            %initialization:
                validation_surface = zeros(length(rp_v),length(kp_v));
                %test_surface = zeros(length(rp_v),length(kp_v));
            for nkp = 1 : length(kp_v)
                %G_train:
                    kp = kp_v(nkp);
                    %load G => G_train:
                        FN = FN_Gram_matrix_k(dataset,cost_name,kernel,kp,base_kp);
                        load(FN,'G');
                        G_train = G(idx_train,idx_train); %Gram matrix of the training distributions  
                %validation_slice, test_slice:
                    [validation_slice,test_slice] = compute_slices(rp_v,G,G_train,L_train,Y_train,Y_val,Y_test,idx_train,idx_val,idx_test); %the function supports 'Matlab: parallel computing' if you activate 'parfor' in it              
                    validation_surface(:,nkp) = validation_slice;   
                    
                    %%%comment out when predicting
                    %test_surface(:,nkp) = test_slice;    
                    %%%
            end
        %optimal parameters (regularization-, kernel parameter):
            [rp_idx,kp_idx,minA] = min_2D(validation_surface);
            kp_opt = kp_v(kp_idx);
            rp_opt = rp_v(rp_idx);
            RMSE_opt = validation_surface(rp_idx,kp_idx); %validation error for the chosen parameters
                disp(strcat(['Experiment:', num2str(v),', fold:',num2str(nCV),' -> RMSE-optimal for validation: ',num2str(RMSE_opt),'.']));
                % print optimal parameters
                disp(strcat(['Optimal theta: ', num2str(kp_opt),', lambda: ', num2str(rp_opt)]));
        %plot (validaton_surface, test_surface):                
            if plot_needed
                plot_surf(validation_surface,kp_v,rp_v,base_kp,base_rp,'log(validation surface)');
                %plot_surf(test_surface,kp_v,rp_v,base_kp,base_rp,'log(test surface)');
            end
        %Y_test_predicted_test:
            %load G => G_train:
                FN = FN_Gram_matrix_k(dataset,cost_name,kernel,kp_opt,base_kp);
                load(FN,'G');
                G_train = G(idx_train,idx_train); %Gram matrix of the training distributions  
            %left:
                A = real(inv(G_train + L_train * rp_opt * eye(L_train))); %real(): 'eps' imaginary values may appear
                left = Y_train.' * A; %left hand side of the inner product; row vector            
            Y_predicted_test = ( left * G(idx_train,idx_test) ).';%column vector
            
            % Save predicted y as a csv
            result = zeros(length(Y_predicted_test), 2);
            result(:, 1) = idx_test;
            result(:, 2) = Y_predicted_test;
            %csvwrite('Prediction_10.csv', result);
            % Decimals precision
            dlmwrite('Prediction_10.csv', result, 'delimiter', ',', 'precision', 15); 
            
            
            % Compute test RMSE
            %RMSE_test = RMSE(Y_test, Y_predicted_test);
            if plot_needed
                %figure; plot([1:length(Y_test)],Y_test,'b',[1:length(Y_test)],Y_predicted_test,'g'); legend({'true','predicted'});
            end
        %save:
            %save_dir = strcat(dataset,'_results_k');
            %dir_present = create_and_cd_dir(save_dir);
            %FN = FN_surfs_optspars_k(dataset,cost_name,kernel,CV,nCV,v,base_kp,base_rp);
            %save(FN,'validation_surface','test_surface','kp_v','rp_v','kp_idx','rp_idx','kp_opt','rp_opt','RMSE_opt','base_kp','base_rp', 'Y_predicted_test','Y_test'); 
            %cd(dir_present);
    %end%nCV
%end%v
%-------------------------------
%matlabpool close; %use this line with 'Matlab: parallel computing'
