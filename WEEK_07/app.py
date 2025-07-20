import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setting Up The Page
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customized CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-box {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: white;
    font-weight: bold;
    border: none;
    padding: 0.75rem;
    font-size: 1.1rem;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #45a049, #4CAF50);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data():
    """Loading and preprocessing the training data"""
    try:
        # Loading data
        train_df = pd.read_csv("WEEK_07/train.csv")

        # Handling missing values
        def handle_missing_values(df):
            df_processed = df.copy()

            # Features that should be 'None' when missing
            none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                             'MasVnrType']

            for feature in none_features:
                if feature in df_processed.columns:
                    df_processed[feature] = df_processed[feature].fillna(
                        'None')

            # Features that should be 0 when missing
            zero_features = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                             'GarageYrBlt', 'GarageCars', 'GarageArea']

            for feature in zero_features:
                if feature in df_processed.columns:
                    df_processed[feature] = df_processed[feature].fillna(0)

            # LotFrontage - fill with neighborhood median
            if 'LotFrontage' in df_processed.columns:
                df_processed['LotFrontage'] = df_processed.groupby('Neighborhood')['LotFrontage'].transform(
                    lambda x: x.fillna(x.median())
                )

            # Filling remaining missing values
            for col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    if df_processed[col].dtype == 'object':
                        mode_val = df_processed[col].mode()
                        if len(mode_val) > 0:
                            df_processed[col] = df_processed[col].fillna(
                                mode_val[0])
                        else:
                            df_processed[col] = df_processed[col].fillna(
                                'Unknown')
                    else:
                        df_processed[col] = df_processed[col].fillna(
                            df_processed[col].median())

            return df_processed

        # Feature engineering
        def create_features(df):
            df_engineered = df.copy()

            # House age
            if 'YearBuilt' in df_engineered.columns and 'YrSold' in df_engineered.columns:
                df_engineered['HouseAge'] = df_engineered['YrSold'] - \
                    df_engineered['YearBuilt']
                df_engineered['HouseAge'] = df_engineered['HouseAge'].clip(
                    lower=0)

            # Years since remodeling
            if 'YearRemodAdd' in df_engineered.columns and 'YrSold' in df_engineered.columns:
                df_engineered['YearsSinceRemod'] = df_engineered['YrSold'] - \
                    df_engineered['YearRemodAdd']
                df_engineered['YearsSinceRemod'] = df_engineered['YearsSinceRemod'].clip(
                    lower=0)

            # Total area
            area_features = ['1stFlrSF', '2ndFlrSF',
                             'TotalBsmtSF', 'GarageArea']
            available_area_features = [
                col for col in area_features if col in df_engineered.columns]
            if len(available_area_features) >= 2:
                df_engineered['TotalSF'] = df_engineered[available_area_features].sum(
                    axis=1)

            # Total bathrooms
            total_baths = 0
            if 'FullBath' in df_engineered.columns:
                total_baths += df_engineered['FullBath']
            if 'BsmtFullBath' in df_engineered.columns:
                total_baths += df_engineered['BsmtFullBath']
            if 'HalfBath' in df_engineered.columns:
                total_baths += df_engineered['HalfBath'] * 0.5
            if 'BsmtHalfBath' in df_engineered.columns:
                total_baths += df_engineered['BsmtHalfBath'] * 0.5
            df_engineered['TotalBaths'] = total_baths

            # Overall score
            if 'OverallQual' in df_engineered.columns and 'OverallCond' in df_engineered.columns:
                df_engineered['OverallScore'] = df_engineered['OverallQual'] * \
                    df_engineered['OverallCond']

            return df_engineered

        # Encoding categorical features
        def encode_features(df):
            df_encoded = df.copy()

            # Ordinal features
            ordinal_features = {
                'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
                'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
            }

            for feature, mapping in ordinal_features.items():
                if feature in df_encoded.columns:
                    df_encoded[feature] = df_encoded[feature].map(
                        mapping).fillna(0)

            # One-hot encoding remaining categorical columns
            categorical_cols = df_encoded.select_dtypes(
                include=['object']).columns
            remaining_categorical = [
                col for col in categorical_cols if col not in ordinal_features.keys() and col not in ['Id']]

            if remaining_categorical:
                df_encoded = pd.get_dummies(
                    df_encoded, columns=remaining_categorical, drop_first=True, dummy_na=False)

            return df_encoded

        # Processing the data
        train_processed = handle_missing_values(train_df)
        train_engineered = create_features(train_processed)
        train_encoded = encode_features(train_engineered)

        # Removing Id column if exists
        if 'Id' in train_encoded.columns:
            train_encoded = train_encoded.drop('Id', axis=1)

        return train_df, train_encoded

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


@st.cache_resource
def train_model(X, y):
    """Train the best performing model"""
    # Splitting the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Feature selection
    k_best = SelectKBest(score_func=f_regression,
                         k=min(50, X_train_scaled.shape[1]))
    X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
    X_val_selected = k_best.transform(X_val_scaled)

    # Training Random Forest model (typically performs well)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_selected, y_train)

    # Getting selected feature names
    selected_features = X.columns[k_best.get_support()]

    return model, scaler, k_best, selected_features, X_val_selected, y_val


def predict_price(model, scaler, k_best, user_inputs, feature_names):
    """Make prediction based on user inputs"""
    try:
        # Creating a dataframe with user inputs
        input_df = pd.DataFrame([user_inputs])

        # Ensuring all required columns are present
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reordering columns to match training data
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scaling and selecting features
        input_scaled = scaler.transform(input_df)
        input_selected = k_best.transform(input_scaled)

        # Making prediction
        prediction = model.predict(input_selected)[0]

        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>',
                unsafe_allow_html=True)

    # Loading data and training model
    with st.spinner("Loading data and training model..."):
        original_data, processed_data = load_and_process_data()

        if processed_data is not None:
            # Preparing features and target
            X = processed_data.drop('SalePrice', axis=1)
            y = processed_data['SalePrice']

            # Training model
            model, scaler, k_best, selected_features, X_val_selected, y_val = train_model(
                X, y)

            # Model performance
            y_val_pred = model.predict(X_val_selected)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            r2 = r2_score(y_val, y_val_pred)

    if processed_data is not None:
        st.success("✅ Model trained successfully!")

        # Sidebar for user inputs
        st.sidebar.title("🏠 House Features")
        st.sidebar.markdown("Enter the house characteristics below:")

        # Creating input widgets for key features
        user_inputs = {}

        # Numerical inputs
        user_inputs['GrLivArea'] = st.sidebar.slider(
            "Living Area (sq ft)", 500, 5000, 1500)
        user_inputs['LotArea'] = st.sidebar.slider(
            "Lot Area (sq ft)", 1000, 20000, 8000)
        user_inputs['OverallQual'] = st.sidebar.selectbox(
            "Overall Quality", list(range(1, 11)), index=6)
        user_inputs['OverallCond'] = st.sidebar.selectbox(
            "Overall Condition", list(range(1, 11)), index=4)
        user_inputs['YearBuilt'] = st.sidebar.slider(
            "Year Built", 1872, 2023, 2000)
        user_inputs['YearRemodAdd'] = st.sidebar.slider(
            "Year Remodeled", 1950, 2023, 2000)
        user_inputs['1stFlrSF'] = st.sidebar.slider(
            "1st Floor Area (sq ft)", 300, 3000, 1000)
        user_inputs['2ndFlrSF'] = st.sidebar.slider(
            "2nd Floor Area (sq ft)", 0, 2000, 500)
        user_inputs['TotalBsmtSF'] = st.sidebar.slider(
            "Basement Area (sq ft)", 0, 3000, 1000)
        user_inputs['GarageCars'] = st.sidebar.selectbox(
            "Garage Cars", [0, 1, 2, 3, 4], index=2)
        user_inputs['GarageArea'] = st.sidebar.slider(
            "Garage Area (sq ft)", 0, 1500, 500)
        user_inputs['FullBath'] = st.sidebar.selectbox(
            "Full Bathrooms", [0, 1, 2, 3, 4], index=2)
        user_inputs['BedroomAbvGr'] = st.sidebar.selectbox(
            "Bedrooms Above Ground", [1, 2, 3, 4, 5, 6], index=2)

        # Categorical inputs (encoded)
        user_inputs['ExterQual'] = st.sidebar.selectbox("Exterior Quality",
                                                        ['Poor', 'Fair', 'Average',
                                                            'Good', 'Excellent'],
                                                        index=2)
        quality_mapping = {'Poor': 1, 'Fair': 2,
                           'Average': 3, 'Good': 4, 'Excellent': 5}
        user_inputs['ExterQual'] = quality_mapping[user_inputs['ExterQual']]

        user_inputs['KitchenQual'] = st.sidebar.selectbox("Kitchen Quality",
                                                          ['Poor', 'Fair', 'Average',
                                                              'Good', 'Excellent'],
                                                          index=2)
        user_inputs['KitchenQual'] = quality_mapping[user_inputs['KitchenQual']]

        user_inputs['YrSold'] = 2023
        user_inputs['MoSold'] = 6
        user_inputs['LotFrontage'] = user_inputs['LotArea'] / \
            100  # Rough estimate
        # Bedrooms + kitchen + living room
        user_inputs['TotRmsAbvGrd'] = user_inputs['BedroomAbvGr'] + 2

        st.subheader("🔮 Price Prediction")

        # Adding input validation messages
        validation_messages = []

        # Check if key inputs are reasonable
        if user_inputs['GrLivArea'] < 500:
            validation_messages.append("⚠️ Living area seems unusually small")
        if user_inputs['GrLivArea'] > 4000:
            validation_messages.append("⚠️ Living area seems unusually large")
        if user_inputs['OverallQual'] < 3:
            validation_messages.append("⚠️ Overall quality is quite low")
        if user_inputs['YearBuilt'] < 1900:
            validation_messages.append("⚠️ Year built seems very old")

        # Displaying validation messages
        if validation_messages:
            st.info("Input Validation Notes:")
            for msg in validation_messages:
                st.write(msg)

        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            st.markdown("---")
            prediction_button = st.button(
                "🎯 Predict House Price",
                type="primary",
                use_container_width=True,
                help="Click to get price prediction based on your inputs"
            )

            # Make prediction
            if prediction_button:
                with st.spinner("🔄 Calculating price prediction..."):
                    prediction = predict_price(
                        model, scaler, k_best, user_inputs, X.columns)

                if prediction is not None:
                    # Displaying prediction with enhanced styling
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🏠 Predicted Price</h2>
                        <h1>${prediction:,.0f}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                    # Showing confidence metrics
                    col_pred1, col_pred2 = st.columns(2)

                    with col_pred1:
                        st.metric(
                            label="Predicted Price",
                            value=f"${prediction:,.0f}"
                        )

                    with col_pred2:
                        confidence = "High" if abs(prediction - (original_data['SalePrice'].mean(
                        ) if original_data is not None else prediction)) < prediction * 0.3 else "Medium"
                        st.metric(
                            label="Confidence Level",
                            value=confidence
                        )

                    # Price range estimation
                    st.write("**📊 Price Range Estimate:**")
                    lower_bound = prediction * 0.9
                    upper_bound = prediction * 1.1
                    st.info(
                        f"**Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}**")

                    # Market comparison
                    if original_data is not None:
                        avg_price = original_data['SalePrice'].mean()
                        median_price = original_data['SalePrice'].median()

                        st.write("**🏘️ Market Comparison:**")

                        col_market1, col_market2 = st.columns(2)
                        with col_market1:
                            st.metric(
                                label="Market Average",
                                value=f"${avg_price:,.0f}",
                                delta=f"{prediction - avg_price:+,.0f}"
                            )
                        with col_market2:
                            st.metric(
                                label="Market Median",
                                value=f"${median_price:,.0f}",
                                delta=f"{prediction - median_price:+,.0f}"
                            )

                        # Market positioning
                        if prediction > avg_price * 1.1:
                            st.success(
                                "🔥 **Premium Property** - Above market average")
                        elif prediction < avg_price * 0.9:
                            st.success(
                                "💡 **Value Opportunity** - Below market average")
                        else:
                            st.info(
                                "📊 **Market Rate** - Close to market average")
                else:
                    st.error(
                        "❌ Unable to calculate prediction. Please check your inputs.")

            else:
                # Showing placeholder when no prediction is made
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0; color: black;">
                    <h3 style="color: black;">👆 Click the button above to get your price prediction</h3>
                    <p style="color: black;">Adjust the house features in the sidebar and click "Predict House Price"</p>
                </div>
                """, unsafe_allow_html=True)

        # Input summary section below the prediction
        st.subheader("📋 Input Summary")
        st.write("**Key Features:**")

        # Creating feature name mapping for better readability
        feature_names = {
            # Area features
            'GrLivArea': 'Above Ground Living Area',
            'LotArea': 'Lot Size',
            '1stFlrSF': 'First Floor Area',
            '2ndFlrSF': 'Second Floor Area',
            'TotalBsmtSF': 'Total Basement Area',
            'GarageArea': 'Garage Area',
            # Year and quality features
            'YearBuilt': 'Year Built',
            'YearRemodAdd': 'Year Remodeled',
            'OverallQual': 'Overall Quality Rating',
            'OverallCond': 'Overall Condition Rating',
            'ExterQual': 'Exterior Quality Rating',
            'KitchenQual': 'Kitchen Quality Rating',
            # Room and feature counts
            'GarageCars': 'Garage Car Capacity',
            'FullBath': 'Full Bathrooms',
            'BedroomAbvGr': 'Bedrooms Above Ground'
        }

        # Creating columns for input summary to make it more organized
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.write("**Areas:**")
            area_features = ['GrLivArea', 'LotArea', '1stFlrSF',
                             '2ndFlrSF', 'TotalBsmtSF', 'GarageArea']
            for key in area_features:
                if key in user_inputs:
                    display_name = feature_names.get(key, key)
                    st.write(f"• {display_name}: {user_inputs[key]:,} sq ft")

        with summary_col2:
            st.write("**Years & Quality:**")
            quality_features = ['YearBuilt', 'YearRemodAdd',
                                'OverallQual', 'OverallCond', 'ExterQual', 'KitchenQual']
            for key in quality_features:
                if key in user_inputs:
                    display_name = feature_names.get(key, key)
                    if 'Rating' in display_name:
                        st.write(f"• {display_name}: {user_inputs[key]}/10")
                    else:
                        st.write(f"• {display_name}: {user_inputs[key]}")

        with summary_col3:
            st.write("**Rooms & Features:**")
            room_features = ['GarageCars', 'FullBath', 'BedroomAbvGr']
            for key in room_features:
                if key in user_inputs:
                    display_name = feature_names.get(key, key)
                    st.write(f"• {display_name}: {user_inputs[key]}")

        st.markdown("---")

        # Main content area for charts
        col1, col2 = st.columns([1, 1])

        with col1:
            # Feature importance
            st.subheader("📈 Feature Importance")
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)

                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                             title="Top 10 Most Important Features")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Data visualization
            st.subheader("📊 Price Distribution")
            if original_data is not None:
                fig = px.histogram(original_data, x='SalePrice', nbins=50,
                                   title="Distribution of House Prices")
                fig.update_layout(
                    xaxis_title="Sale Price ($)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            # st.write("")

        st.subheader("📈 Advanced Analytics")

        if original_data is not None:
            feature_to_plot = st.selectbox("Select feature to plot against price:",
                                           ['GrLivArea', 'LotArea', 'OverallQual', 'YearBuilt'])

            fig = px.scatter(original_data, x=feature_to_plot, y='SalePrice',
                             title=f"Price vs {feature_to_plot}")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**🏠 House Price Predictor** By Avinash Yadav")


if __name__ == "__main__":
    main()
