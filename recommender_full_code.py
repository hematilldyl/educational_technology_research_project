import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds


def process_data(student_data,metadata):
    course_survey_basic = pd.read_csv(student_data,index_col=0)
    course_metadata = pd.read_csv(metadata)
    course_survey_basic['Q3'] = course_survey_basic['Q3'].str.split(',')
    course_survey_exploded = course_survey_basic.explode('Q3')
    X_user = course_survey_exploded.iloc[:,:1].reset_index()
    X_user=X_user.rename(columns={'Q3':'course_code'})
    X_model = pd.merge(X_user,course_metadata,on='course_code')
    return X_model

def cluster_data(data,n_clusters=3,categorical_columns=[0,1,2,6,7,8]):
    kprototypes = KPrototypes(n_clusters=n_clusters, init='random',verbose=1)
    data['Predicted']= kprototypes.fit_predict(data.values,categorical=categorical_columns)+1
    return data 

def build_cluster_features(data):
    data['num_courses']=data['Predicted'].groupby(data['response']).transform('count')
    data['score1'] = data[data['Predicted'].eq(1)].groupby(['response'])['Predicted'].transform('count')
    data['score2'] = data[data['Predicted'].eq(2)].groupby(['response'])['Predicted'].transform('count')
    data['score3'] = data[data['Predicted'].eq(3)].groupby(['response'])['Predicted'].transform('count')
    data['average_pred'] = data.groupby(['response'])['Predicted'].transform('mean')
    data=data.fillna(0).groupby(['response','num_courses']).max().reset_index()
    return data

def peer_match(data):
    X_peer_match = data[['response','num_courses','score1','score2','score3','average_pred']]
    model = NMF(n_components=len(X_peer_match)//5, random_state=0)
    nmf_features = model.fit_transform(X_peer_match.set_index('response'))
    normalized = normalize(nmf_features)
    df = pd.DataFrame(data=normalized, index=X_peer_match["response"])
    matches = []
    for i in range(0,len(df)):
        similarities = df.dot(df.iloc[i])
        sims = pd.DataFrame(similarities.nlargest(5))
        matches.append(list(sims.index))
    X_peer_match['matches'] = matches 
    return X_peer_match

def recommend_course(predictions_df, response_number, courses_df, num_recommendations=5):
    # Get and sort the user's predictions
    sorted_user_predictions = predictions_df.loc[response_number].sort_values(ascending=False)
    taken = courses_df[courses_df['response']==response_number]['course_name']
    recs = sorted_user_predictions.loc[~sorted_user_predictions.index.isin(taken.values)].iloc[:num_recommendations]
    return taken, recs

def generate_course_recommender(users,data,k=7):
    pivoted = users.pivot_table(index = 'response', columns ='course_name', values = 'Predicted',aggfunc='mean').fillna(0)
    R = pivoted.to_numpy()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(pivoted, k = k)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivoted.columns)
    preds_df['response'] = pivoted.index
    preds_df = preds_df.set_index('response')
    preds = []
    taken = []
    for response in preds_df.index:
        already_taken, predictions = recommend_course(preds_df,response,users)
        preds.append(list(predictions.index))
        taken.append(list(already_taken))
    X_full = data.copy(deep=True)
    X_full['taken_courses'] = taken
    X_full['recommended_courses'] = preds
    return X_full

def main():
    X_data = process_data('C:/Users/Dylan/Documents/EdTechResearch/edtech_first_survey_processed.csv','C:/Users/Dylan/Documents/EdTechResearch/edtech_course_metadata.csv')
    X_data_with_clusters = cluster_data(X_data)
    X_with_features = build_cluster_features(X_data_with_clusters)
    X_peer_matched = peer_match(X_with_features)
    X_course_recs = generate_course_recommender(X_data_with_clusters,X_with_features)
    results = X_peer_matched.copy(deep=True)
    results['course_recs'] = X_course_recs['recommended_courses']
    results['taken_courses'] = X_course_recs['taken_courses']
    results.to_csv('C:/Users/Dylan/Documents/EdTechResearch/final_output.csv')
    return

if __name__ == "__main__":
    main()