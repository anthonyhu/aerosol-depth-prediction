cols_excl = ["id", "y"]
cols_orig = [c for c in train.columns if c not in cols_excl]

## Gradient boosting
cols_xgb = cols_orig
cols_lr0 = cols_orig
cols_lr1 = cols_orig

n_trees = 200
models_weights = {"lr0": 1.0}
models_weights_agg = {"lr0": 0.6, "lr1": 0.4, "xgb": 0}
models_cols = {"lr0": cols_lr0, "lr1": cols_lr1, "xgb": cols_xgb}

# Scoring function in the hyperopt hyperparameters tuning.
def scoring_function(parameters):
    print("Training the model with parameters: ")
    print(parameters)
    early_stopping_rounds = 5
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
        agg_train_fold = train_fold.groupby("id").median().reset_index()
        agg_validation = validation_fold.groupby("id").median().reset_index()
        
        # Linear Model
        model_lr1 = LinearRegression()
        model_lr1.fit(agg_train_fold[cols_lr1], agg_train_fold["y"])
        
        # Gradient boosting
        D_train = xgb.DMatrix(agg_train_fold[cols_xgb], label=agg_train_fold["y"])
        D_validation = xgb.DMatrix(agg_validation[cols_xgb], label=agg_validation["y"])
        
        # Two eval metrics to print. Validation-rmse will be used for early stopping.
        watchlist = [(D_train, "Training"), (D_validation, "Validation")]
        model_xgb = xgb.train(parameters, D_train, n_trees, watchlist, 
                              early_stopping_rounds=early_stopping_rounds,
                              verbose_eval=5)
        
        models_agg = {"lr1": model_lr1, "xgb": model_xgb}
        
        # Compute RMSE on validation set
        validation_pred = validation_fold[["id"]].assign(y_hat=0).reset_index(drop=True)
        for i, m in models.items():
            validation_pred["y_hat"] += models_weights_agg[i] * m.predict(validation_fold[models_cols[i]])
        
        agg_validation["agg_y_hat"] = 0.0
        # Aggregated prediction
        for i, m in models_agg.items():
            if (i == "xgb"):
                agg_validation["agg_y_hat"] += models_weights_agg[i] * m.predict(D_validation,
                                                                          ntree_limit=m.best_iteration)
            else:
                agg_validation["agg_y_hat"] += models_weights_agg[i] * m.predict(agg_validation[models_cols[i]])
                
        # Add aggregated prediction
        agg_dict = dict(zip(agg_validation["id"], agg_validation["agg_y_hat"]))
        validation_pred["y_hat"] += validation_pred["id"].replace(agg_dict)
        
        # Use median value by id
        y_hat_med = validation_pred.groupby("id").median()["y_hat"].to_dict()
        
        RMSE = np.sqrt(mean_squared_error(validation_pred["id"].replace(y_hat_med).values, validation_fold["y"]))
        average_RMSE += RMSE
        print("Current validation RMSE: {0}".format(RMSE))

    average_RMSE /= n_splits

    print("Cross-validation score: {0}\n".format(average_RMSE))
    
    return {"loss": average_RMSE, "status": STATUS_OK}


t0 = time()

# Grid to pick parameters from.
parameters_grid = {"eta": hp.quniform("eta", 0.2, 0.4, 0.1),
                   "max_depth": hp.choice("max_depth", np.arange(4, 8, dtype=int)),
                   #"min_child_weight": hp.quniform("min_child_weight", 1, 4, 1),
                   #"gamma": hp.quniform("gamma", 0, 1, 0.2),
                   "subsample": hp.quniform("subsample", 0.7, 0.9, 0.1),
                   "lambda": hp.quniform("lambda", 0.2, 0.6, 0.2),
                   "objective": "reg:linear",
                   "eval_metric": "rmse",
                   "seed": random_seed,
                   "nthread": n_threads,
                   "silent": 0
                  }
# Record the information about the cross-validation.
trials = Trials()

best = fmin(scoring_function, parameters_grid, algo=tpe.suggest, max_evals=1, 
            trials=trials) 

computing_time = time() - t0