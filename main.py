import streamlit as st
import pickle
import pandas as pd
from xgboost import XGBClassifier



def main():

    # Set project
    project = 'OASIS'

    # Load UI Config file
    ui_df = pd.read_csv('ui~target_transfertohospice.csv')

    # Create Form Title Header
    st.title('Prediction Classification')
    st.header('Hospice Care')

    # Iterate through the DataFrame and create Streamlit form elements
    ui_question_order_prev = 0
    form_values = {}

    for index, row in ui_df.iterrows():
        #
        ui_question_order = row['ui_question_order']
        ui_type = row['ui_type']
        question = row['question']
        score_mean = row['score_mean']
        score_min = row['score_min']
        score_max = row['score_max']
        question_type = row['question_type']

        # Create Question if question is different from previous
        if ui_question_order_prev != ui_question_order:

            # Get the Options and Var for each question
            filtered_df = ui_df[ui_df['ui_question_order'] == ui_question_order]
            sorted_df = filtered_df.sort_values(by=['ui_option_order'])
            options_df = sorted_df[['question_option']]
            options_value_df = sorted_df[['option_value']]
            options_list = options_df['question_option'].tolist()
            options_value_list = options_value_df['option_value'].tolist()
            var_df = sorted_df[['var']]
            var_list = var_df['var'].tolist()

            if ui_type == 'text_input':

                option_value = st.text_input(question, 'Enter')
                field = var_list[0]
                form_values[field] = option_value

            if ui_type == 'selectbox':

                option_value = st.selectbox(question, options_list)
                option_index = options_list.index(option_value)
                field = var_list[option_index]
                value = options_value_list[option_index]
                if question_type == 'parent':
                    form_values[field] = 1
                else:
                    form_values[field] = value

            elif ui_type == 'slider':
                print(question,score_min, score_max, score_mean)
                option_value = st.slider(question, min_value=int(score_min),
                                                   max_value=int(score_max),
                                                   value=int(score_mean))
                field = var_list[0]
                form_values[field] = option_value

            elif ui_type == 'checkbox':

                st.write(question)
                for option in options_list:
                    option_index = options_list.index(option)
                    field = var_list[option_index]
                    form_values[field] = int(st.checkbox(option))

        ui_question_order_prev = ui_question_order

    # Load the predictive model from the pickle file
    with open('XGBoost~target_transfertohospice.pkl', 'rb') as model_file:
        predictive_model = pickle.load(model_file)

    # The feature importances can be extracted for RandomForest and many other model types
    importances = predictive_model.feature_importances_

    # Create a DataFrame with features and their importance
    feature_importances_df = pd.DataFrame({'Importance': importances})

    # Sort the DataFrame based on feature importance
    feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Save the DataFrame to a CSV file
    feature_importances_df.to_csv('feature_importances.csv', index=False)

    def make_prediction(input_data):

        import numpy as np

        input_data_2d = np.array(input_data).reshape(1, -1)
        prediction = predictive_model.predict(input_data_2d)
        probability = predictive_model.predict_proba(input_data_2d)[:, 1]

        return prediction, probability

    # Button to trigger the predictive model
    if st.button('Predict'):

        form_df = pd.DataFrame(list(form_values.items()), columns=['form_field', 'form_value'])
        form_df['form_feature'] = form_df['form_field'] + '_' + form_df['form_value'].astype(int).astype(str)
        form_df.to_csv('form_df.csv', index=False)

        # Map model input values
        import numpy as np

        # Create df for storing model inputs and other attribs
        ui_model_df = ui_df
        ui_model_df['model_input'] = np.nan
        ui_model_df = ui_model_df[ui_model_df['feature_order'] >= 0]

        # Map Positive 1 Binary values
        positive_df = form_df.merge(ui_model_df, left_on=['form_field', 'form_feature'], right_on=['var', 'var_feature'])
        positive_df.loc[positive_df['form_feature'] == positive_df['var_feature'], 'model_input'] = 1
        positive_df = positive_df[positive_df['feature_order'].notna() & (positive_df['feature_order'] != '')]

        # Map Continuous or question type's of 'value' to the actual form value
        filtered_df = ui_model_df[ui_model_df['question_type'] == 'value']
        value_df = form_df.merge(filtered_df, left_on=['form_field'], right_on=['var'])
        value_df['model_input'] = value_df['form_value']

        # Combine positive and value df's with only cols needed for model input
        positive_value_df = pd.concat([value_df, positive_df], axis=0)
        cols = ['var_feature', 'feature_order', 'score_max',
                'question_type', 'model_input']
        positive_value_df = positive_value_df[cols]

        # Map Zero Binary Value - any feature left unmapped must be 0
        filtered_df = ui_model_df[ui_model_df['var_feature'].notna()]
        zero_df = filtered_df[~filtered_df['var_feature'].isin(positive_value_df['var_feature'])]
        zero_df = zero_df[cols]
        zero_df['model_input'] = 0

        # Finalize model input df
        model_input_df = pd.concat([positive_value_df, zero_df], axis=0)
        model_input_df = model_input_df.sort_values(by='feature_order')

        # Scale the input value between 0-1
        model_input_df = model_input_df.reset_index(drop=True)
        model_input_df.loc[model_input_df['question_type'] == 'value', 'model_input'] /= model_input_df['score_max']

        model_input_df.to_csv('model_input_df.csv', index=False)

        # Call the predictive model
        prediction, probability = make_prediction(model_input_df['model_input'])
        probability_pct = float(probability * 100)
        probability_pct_str = str(round(probability_pct, 2)) + '%'
        print('Prediction: ', prediction)
        print('Probability: ', probability_pct_str)

        # Display the prediction result
        if prediction == 1:
            st.success('❤️ Patient should be considered for Hospice Care with a high probability of ' + probability_pct_str + '❤️')
        else:
            st.success('💪 Patient does not likely need Hospice Care with a low probability of ' + probability_pct_str + '💪')

if __name__ == '__main__':
    main()
