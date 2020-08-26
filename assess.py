
# Import Packages
import pandas as pd
import os
import sklearn as sk
import numpy as np
import pickle
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_curve, auc

# Global Vars
pathto_data = '/app_io'
pathto_spacefeats = os.path.join(pathto_data, 'spatial_features_model', 'output-spatial_features')
pathto_damdata = os.path.join(pathto_data, 'phase_1_optimization', 'input', 'MA_U.csv')
pathto_deployidx = os.path.join(pathto_data, 'phase_1_optimization', 'input', 'deploy_idx.pkl')
pathto_phase1_results = os.path.join(pathto_data, 'phase_1_results_convert', 'output', 'results.csv')
pathto_solution_classifications = os.path.join(pathto_data, 'phase_2_assessment', 'output', 'solution_classifications')
pathto_assessment_objectives = os.path.join(pathto_data, 'phase_2_assessment', 'output', 'assessment_objectives')
parameter_names = ['N_length', 'N_width', 'n_estimators', 'min_samples_split', 'min_samples_leaf',
                   'min_weight_fraction_leaf', 'max_depth', 'max_features', 'max_leaf_nodes']
objective_names = ['P2_accuracy', 'P2_FPR', 'P2_TPR', 'P1_AUROCC']
feature_names = ['Dam Height (ft)', 'Dam Length (ft)', 'Reservoir Size (acre-ft)', 'Maximum Downstream Slope (%)',
                 'Downstream Houses', 'Downstream Population', 'Building Exposure ($1000)',
                 'Building Footprint (1000 sq. ft.)', 'Content Exposure ($1000)']
predicted_name = 'Hazard'
positive_lab = 'NH'


def parameter_converter(params):
    """
    Convert parameter to valid types
    :param params: tuple
                    current parameters of default types
    :return: dict
                    All the corresponding parameters in required types
    """
    # Parse Ints
    for i, val in enumerate(params):
        if val.is_integer():
            params[i] = int(val)
    # Convert to Dictionary
    param_dict = dict(zip(parameter_names, params))
    return param_dict


def get_features(param_dict):
    """
    Retrive the corresponding spatial and non-spatial feature values
    :param param_dict: dict
                        All the corresponding simulation parameters
    :return: DataFrame
                        Spatial and non-spatial dam hazard feature values
    """
    # Import Spatial Features
    df_name = 'N_length_' + str(param_dict['N_length']) + '_N_width_' + str(param_dict['N_width'])
    space_feats = pd.read_hdf(os.path.join(pathto_spacefeats, 'spatial_feats.h5'), df_name)
    # Import Non-Spatial Features
    data = pd.read_csv(pathto_damdata)
    # Merge Features
    data = space_feats.join(data)
    data.index = data['RECORDID']
    # Rename Columns
    data = data.rename(index=str, columns={'HAZARD': predicted_name, 'DAM_HEIGHT': feature_names[0],
                                           'DAM_LENGTH': feature_names[1], 'NORMAL_STORAGE': feature_names[2],
                                           'Slope_max': feature_names[3], 'hous_sum': feature_names[4],
                                           'pop_sum': feature_names[5], 'buil_sum': feature_names[6],
                                           'foot_sum': feature_names[7], 'cont_sum': feature_names[8]})
    # Extract Features
    data = data[feature_names+[predicted_name]]
    # Export
    return data


def preprocessor(df):
    """
    Processing the feature values before classification
    :param df: DataFrame
                    Feature values
    :return: DataFrame
                    Processed feature values
    """
    # Combine Categories
    df = df.replace(to_replace=['L', 'S', 'H'], value=['NH', 'NH', 'H'])
    # Replace nans with median
    df = df.fillna(df.median())
    # Specify Objective
    y = df[predicted_name]
    # Shape Data
    X = np.array(df[feature_names])
    y = np.array(y)
    return X, y


def train_model(ml_params, data):
    """
    Train the random forest to the current set of hyperparameters (no cross-validation)
    :param ml_params: dict
                        Current set of hyperparameters
    :param data: DataFrame
                        The current set of dams with features and true hazard classifications
    :return: RandomForestClassifier
                        Trained random forest
    """
    # Initialized Vars
    random_state = 1008
    # Process Data
    X, y = preprocessor(data)
    # Resample the training data to deal with class imbalance
    method = ADASYN(random_state=random_state)
    X_res, y_res = method.fit_sample(X, y)
    # Create Model
    clf = sk.ensemble.RandomForestClassifier(n_jobs=-1, random_state=random_state,
                                             n_estimators=ml_params['n_estimators'],
                                             min_samples_split=ml_params['min_samples_split'],
                                             min_samples_leaf=ml_params['min_samples_leaf'],
                                             min_weight_fraction_leaf=ml_params['min_weight_fraction_leaf'],
                                             max_depth=ml_params['max_depth'],
                                             max_features=ml_params['max_features'],
                                             max_leaf_nodes=ml_params['max_leaf_nodes'])
    # Fit model to train data
    clf.fit(X_res, y_res)
    # Export
    return clf


def predict_values(model, data):
    """
    Predict values based on a trained random forest
    :param model: RandomForestClassifier
                    Trained random forest
    :param data: DataFrame
                    The current set of dams with features and true hazard classifications
    :return: DataFrame
                    The current set of dams with features, true hazard classifications, and predicted hazard
                     classifications
    """
    # Process Data
    X, y = preprocessor(data)
    # Predicted Values
    y_pred = model.predict(X)
    # Append Predicted Value
    data['True Hazard Class'] = y
    data['Predicted Hazard Class'] = y_pred
    # Area Under ROC Curve
    y_score = model.predict_proba(X)[:, 1]
    false_positive, true_positive, _ = roc_curve(y, y_score, pos_label=positive_lab)
    AUROCC = auc(false_positive, true_positive)
    data['AUROCC'] = AUROCC
    return data


def CM(row):
    """
    Confusion matrix function to classify true positive, false positive, false negative, or true negative
    classifications
    :param row: Series
                Predicted and true classification of the current dam being evaluated
    :return: str
                Classification type
    """
    if row['True Hazard Class'] == 'H' and row['Predicted Hazard Class'] == 'H':
        return 'TN'
    elif row['True Hazard Class'] == 'NH' and row['Predicted Hazard Class'] == 'NH':
        return 'TP'
    elif row['True Hazard Class'] == 'H' and row['Predicted Hazard Class'] == 'NH':
        return 'FP'
    elif row['True Hazard Class'] == 'NH' and row['Predicted Hazard Class'] == 'H':
        return 'FN'


def get_obj(df):
    """
    Calculate objective values
    :param df: dataframe
                Phase 2 classifications of current solution
    :return:
                Phase 2 objective values
    """
    # Extract Errors
    TP = df['error'].value_counts()['TP']
    TN = df['error'].value_counts()['TN']
    FP = df['error'].value_counts()['FP']
    FN = df['error'].value_counts()['FN']
    # Calculate Objectives
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    AUROCC = df['AUROCC'][0]
    return pd.Series([FN, FP, TN, TP, accuracy, FPR, TPR, AUROCC], index=['P2_FN', 'P2_FP', 'P2_TN', 'P2_TP']+objective_names)


def simulation(vars, name):
    """
    Evaluate a dam hazard potential 'simulation' with a given set of spatial parameters and random forest
    hyperparameters
    :param vars: tuple
                     set of spatial and nonspatial parameters
    :param name: str
                    Name of current solution
    :return: Series
                    Phase 2 objective values
    """
    # Convert Parameters
    param_dict = parameter_converter(vars)
    # Get Features
    data = get_features(param_dict)
    # Get Deployment Indexes
    with open(pathto_deployidx, 'rb') as f:
        deploy_idx = pickle.load(f)
    # Train Model on All But Deployment Features
    model = train_model(param_dict, data.drop(deploy_idx))
    # Predict Deployment Features
    df = predict_values(model, data.loc[deploy_idx])
    # Compute Confusion Matrix
    df['error'] = df.apply(CM, axis=1)
    # Export Classifications
    df.to_csv(os.path.join(pathto_solution_classifications, 'solution_'+str(int(name)) + '.csv'), index=False)
    # Compute Objectives
    objs = get_obj(df)
    print(objs)
    return objs


def main():
    # Import Reference Set
    df = pd.read_table(pathto_phase1_results, sep=',').infer_objects()
    # Use All Solutions
    df['solution_num'] = list(df.index)
    # Run Simulation
    objs_df = df.apply(lambda row: simulation(row[parameter_names].tolist(), row['solution_num']), axis=1)
    rep_df = pd.concat([df, objs_df], axis=1)
    # Export Representative Solution
    rep_df.to_csv(os.path.join(pathto_assessment_objectives, 'assessment_results.csv'), index=False, header=True, sep=',')
    return 0


if __name__ == '__main__':
    main()
