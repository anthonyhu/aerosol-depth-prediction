## Gradient boosting
cols_lr0 = cols_orig
cols_dnn = cols_orig

models_weights = {"lr0": 1.0}
models_weights_agg = {"lr0": 0.2, "dnn": 0.8}
models_cols = {"lr0": cols_lr0, "dnn": cols_dnn}

# Scoring function in the hyperopt hyperparameters tuning.
def scoring_function(parameters):
    print("Training the model with parameters: ")
    print(parameters)
    average_RMSE = 0.0
    n_splits = 5

    kf = KFold(n_splits=n_splits)
    for train_index, validation_index in kf.split(train):
        train_fold, validation_fold = train.loc[train_index], train.loc[validation_index] 

        model_lr0 = LinearRegression()
        model_lr0.fit(train_fold[cols_lr0], train_fold["y"])

        models = {"lr0": model_lr0}

        train_pred = train_fold[["id"]].assign(y_hat=0)
        for i, m in models.items():
            train_pred["y_hat"] += models_weights[i] * m.predict(train_fold[models_cols[i]])

        # Use median value by id
        y_hat_med = train_pred.groupby("id").median()["y_hat"].to_dict()

        RMSE = np.sqrt(mean_squared_error(train_pred["id"].replace(y_hat_med).values, train_fold["y"]))
        
        # Prune outliers
        RMSE_decreasing = True
        count = 0
        while (RMSE_decreasing):
            count +=1
            if ((count % 5) == 0):
                print(count)
            train_pred["y_med"] = train_pred["id"].replace(y_hat_med)

            # Distance from the median for each bag
            train_pred["score"] = (train_pred["y_hat"] - train_pred["y_med"])**2
            # Rank of each instance by bag
            train_pred["rank"] = train_pred.groupby("id")["score"].rank()
            bag_size_dict = train_pred.groupby("id")["score"].count().to_dict()
            train_pred["bag_size"] = train_pred["id"].replace(bag_size_dict)
            train_pred["rank"] = train_pred["rank"] / train_pred["bag_size"]

            # Remove outliers
            outliers_index = train_pred["rank"] > (1 - outliers_threshold)
            train_fold = train_fold.loc[~outliers_index, :].reset_index(drop=True)

            model_lr0 = LinearRegression()
            model_lr0.fit(train_fold[cols_lr0], train_fold["y"])

            models = {"lr0": model_lr0}

            # Compute new RMSE
            train_pred = train_fold[["id"]].assign(y_hat=0)
            
            for i, m in models.items():
                train_pred["y_hat"] += models_weights[i] * m.predict(train_fold[models_cols[i]])

            # Use median value by id
            y_hat_med = train_pred.groupby("id").median()["y_hat"].to_dict()

            new_RMSE = np.sqrt(mean_squared_error(train_pred["id"].replace(y_hat_med), train_fold["y"]))
            if ((count % 5) == 0):
                print(new_RMSE)

            if (new_RMSE < RMSE):
                RMSE = new_RMSE
            else:
                RMSE_decreasing = False
        
        # Aggregated prediction
        #agg_train_fold = train_fold.groupby("id").median().reset_index()
        #agg_validation = validation_fold.groupby("id").median().reset_index()
    
        #models_agg = {"lr1": model_lr1, "xgb": model_xgb}
        feature_cols = [tf.contrib.layers.real_valued_column(k) for k in cols_dnn]
        
        # Tune number of layers
        model_dir = ("./temp_log_"
                     + str(parameters["steps"]) + "_"
                     + str(parameters["nb_neurons_1"]) #+ "_"
                     #+ str(parameters["nb_neurons_2"])
                    )
        model_dnn = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                                  hidden_units=[parameters["nb_neurons_1"]],
                                                                #parameters["nb_neurons_2"]],
                                                  optimizer=tf.train.ProximalAdagradOptimizer(
                                                      learning_rate=0.1,
                                                      l1_regularization_strength=0.001),
                                                  model_dir=model_dir)

        def input_fn(data_set):
            feature_cols = {k: tf.constant(data_set[k].values) for k in cols_dnn}
            labels = tf.constant(data_set["y"].values)
            return feature_cols, labels

        model_dnn.fit(input_fn=lambda: input_fn(train_fold), steps=parameters["steps"])
        
        models = {"lr0": model_lr0, "dnn": model_dnn}

        # Compute RMSE on validation set
        validation_pred = validation_fold[["id"]].assign(y_hat=0).reset_index(drop=True)
        for i, m in models.items():
            if (i == "dnn"):
                temp = m.predict(input_fn=lambda: input_fn(validation_fold))
                # .predict() returns an iterator; convert to an array
                y_hat = np.array(list(itertools.islice(temp, 0, None)))
                validation_pred["y_hat"] += models_weights_agg[i] * y_hat
            else:
                validation_pred["y_hat"] += models_weights_agg[i] * m.predict(validation_fold[models_cols[i]])
            
        #agg_validation["agg_y_hat"] = 0.0
        # Aggregated prediction
        #for i, m in models_agg.items():
            #if (i == "xgb"):
             #   agg_validation["agg_y_hat"] += models_weights_agg[i] * m.predict(D_validation,
              #                                                            ntree_limit=m.best_iteration)
            #else:
             #   agg_validation["agg_y_hat"] += models_weights_agg[i] * m.predict(agg_validation[models_cols[i]])
                
        # Add aggregated prediction
        #agg_dict = dict(zip(agg_validation["id"], agg_validation["agg_y_hat"]))
        #validation_pred["y_hat"] += validation_pred["id"].replace(agg_dict)
        
        # Use median value by id
        y_hat_med = validation_pred.groupby("id").median()["y_hat"].to_dict()
        
        RMSE = np.sqrt(mean_squared_error(validation_pred["id"].replace(y_hat_med).values, validation_fold["y"]))
        average_RMSE += RMSE
        print("Current validation RMSE: {0}".format(RMSE))

    average_RMSE /= n_splits

    print("Cross-validation score: {0}\n".format(average_RMSE))
    
    return {"loss": average_RMSE, "status": STATUS_OK}