# Random Forest
cols_rfr = [c for c in train.columns if c not in cols_excl]
cols_lr0 = ["solar_1"]
cols_lr1 = ["reflectance_2", "reflectance_3"]

n_trees = 50
max_depth = 7
models_weights = {"rfr": 0.5, "lr0": 0.25, "lr1": 0.25}
models_cols = {"rfr": cols_rfr, "lr0": cols_lr0, "lr1": cols_lr1}

kf = KFold(n_splits=5)
RMSE_values = []

current_fold = 0
for train_index, test_index in kf.split(train):
    current_fold += 1
    print("Training with fold {0}".format(current_fold))
    train_fold, test_fold = train.loc[train_index], train.loc[test_index] 

    model_rfr = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, 
                                      n_jobs=n_threads, random_state=random_seed)
    model_rfr.fit(train_fold[cols_rfr], train_fold["y"])

    model_lr0 = Ridge()
    model_lr0.fit(train_fold[cols_lr0], train_fold["y"])

    model_lr1 = Ridge()
    model_lr1.fit(train_fold[cols_lr1], train_fold["y"])

    models = {"rfr": model_rfr, "lr0": model_lr0, "lr1": model_lr1}

    # Initialisation (need to prune outliers)
    test_pred = test_fold[["id"]].assign(y_hat=0).reset_index(drop=True)
    for i, m in models.items():
        test_pred["y_hat"] += models_weights[i] * m.predict(test_fold[models_cols[i]])

    # Use median value by id
    y_hat_med = test_pred.groupby("id").median()["y_hat"].to_dict()

    RMSE = np.sqrt(mean_squared_error(test_pred["id"].replace(y_hat_med).values, test_fold["y"]))
    RMSE_values.append(RMSE)
    print("RMSE: {0}".format(RMSE))
    
print ("Finished! Mean RMSE is: {0}".format(np.mean(RMSE_values)))

# Gradient boosting
cols_xgb = [c for c in train.columns if c not in cols_excl]
cols_lr0 = ["solar_1"]
cols_lr1 = ["reflectance_2", "reflectance_3"]

n_trees = 100
parameters = {"eta": 0.1,
              "ntread": n_threads,
              "max_depth": 7,
              "lambda": 0.5,
              "objective": "reg:linear",
              "eval_metric": "rmse",
              "seed": random_seed
             }
models_weights = {"xgb": 0.9, "lr0": 0.05, "lr1": 0.05}
models_cols = {"xgb": cols_xgb, "lr0": cols_lr0, "lr1": cols_lr1}

kf = KFold(n_splits=5)
RMSE_values = []

current_fold = 0
for train_index, test_index in kf.split(train):
    current_fold += 1
    print("Training with fold {0}".format(current_fold))
    train_fold, test_fold = train.loc[train_index], train.loc[test_index] 

    xgb_train = xgb.DMatrix(train_fold[cols_xgb], label=train_fold["y"])
    model_xgb = xgb.train(parameters, xgb_train, n_trees)

    model_lr0 = Ridge()
    model_lr0.fit(train_fold[cols_lr0], train_fold["y"])

    model_lr1 = Ridge()
    model_lr1.fit(train_fold[cols_lr1], train_fold["y"])

    models = {"xgb": model_xgb, "lr0": model_lr0, "lr1": model_lr1}

    # Initialisation (need to prune outliers)
    test_pred = test_fold[["id"]].assign(y_hat=0).reset_index(drop=True)
    for i, m in models.items():
        if (i == "xgb"):
            xgb_test = xgb.DMatrix(test_fold[models_cols[i]])
            test_pred["y_hat"] += models_weights[i] * m.predict(xgb_test)
        else:
            test_pred["y_hat"] += models_weights[i] * m.predict(test_fold[models_cols[i]])

    # Use median value by id
    y_hat_med = test_pred.groupby("id").median()["y_hat"].to_dict()

    RMSE = np.sqrt(mean_squared_error(test_pred["id"].replace(y_hat_med).values, test_fold["y"]))
    RMSE_values.append(RMSE)
    print("RMSE: {0}".format(RMSE))
    
print ("\nFinished! Mean RMSE is: {0}".format(np.mean(RMSE_values)))