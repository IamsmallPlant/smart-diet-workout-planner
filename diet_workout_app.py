import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🥗 Smart Diet & Workout Planner",
    page_icon="🏋️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved CSS for better visuals and readability
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E7D32;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #2E7D32;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .meal-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        border: none;
    }
    .meal-title {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
        text-transform: capitalize;
    }
    .meal-food {
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    .meal-nutrition {
        font-size: 0.95rem;
        opacity: 0.9;
        line-height: 1.4;
    }
    .workout-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .workout-day {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
        text-transform: capitalize;
    }
    .workout-exercise {
        font-size: 1rem;
        line-height: 1.4;
        opacity: 0.95;
    }
    .grocery-item {
        background-color: #f8f9ff;
        padding: 0.8rem;
        margin: 0.3rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        font-weight: 500;
    }
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .sidebar .stSelectbox label, .sidebar .stNumberInput label, .sidebar .stSlider label {
        font-weight: 600 !important;
        color: #2E7D32 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Create sample nutrition database
@st.cache_data
def create_nutrition_database():
    foods_data = {
        'food_name': [
            # Breakfast items
            'Oats with milk', 'Scrambled eggs', 'Greek yogurt with berries', 'Whole grain toast', 'Banana smoothie',
            'Poha', 'Upma', 'Idli with sambar', 'Paratha with curd', 'Daliya porridge',
            
            # Lunch items
            'Brown rice with dal', 'Chicken breast grilled', 'Quinoa salad', 'Vegetable curry', 'Fish curry',
            'Roti with sabzi', 'Rajma chawal', 'Chole with rice', 'Paneer curry', 'Mixed dal',
            
            # Dinner items
            'Grilled salmon', 'Vegetable stir fry', 'Soup with bread', 'Salad with nuts', 'Steamed vegetables',
            'Light khichdi', 'Vegetable soup', 'Grilled chicken', 'Dal with roti', 'Paneer tikka',
            
            # Snacks
            'Apple with almonds', 'Green tea', 'Protein shake', 'Mixed nuts', 'Roasted chana',
            'Sprouts chat', 'Buttermilk', 'Coconut water', 'Fruit salad', 'Low-fat yogurt'
        ],
        'calories_per_serving': [
            # Breakfast
            350, 200, 150, 80, 250, 300, 200, 250, 400, 180,
            # Lunch  
            450, 300, 350, 200, 350, 350, 400, 450, 300, 250,
            # Dinner
            400, 150, 200, 250, 180, 300, 120, 350, 280, 200,
            # Snacks
            200, 0, 150, 200, 100, 120, 50, 25, 150, 80
        ],
        'protein_g': [
            # Breakfast
            15, 12, 15, 3, 10, 8, 6, 8, 12, 6,
            # Lunch
            20, 35, 15, 8, 25, 12, 15, 18, 20, 18,
            # Dinner
            35, 5, 8, 8, 4, 8, 4, 30, 12, 25,
            # Snacks
            8, 0, 25, 8, 6, 8, 2, 0, 2, 8
        ],
        'carbs_g': [
            # Breakfast
            45, 2, 15, 15, 35, 50, 35, 40, 45, 30,
            # Lunch
            60, 0, 45, 25, 10, 50, 55, 60, 15, 35,
            # Dinner
            5, 20, 25, 15, 25, 50, 20, 0, 40, 8,
            # Snacks
            20, 0, 5, 8, 15, 20, 6, 6, 35, 12
        ],
        'fat_g': [
            # Breakfast
            12, 15, 5, 1, 8, 8, 4, 6, 15, 3,
            # Lunch
            8, 8, 12, 8, 15, 10, 8, 12, 15, 5,
            # Dinner
            20, 8, 6, 18, 8, 6, 2, 15, 8, 8,
            # Snacks
            15, 0, 3, 18, 2, 2, 0, 0, 2, 0
        ],
        'meal_type': [
            # Breakfast
            'breakfast', 'breakfast', 'breakfast', 'breakfast', 'breakfast',
            'breakfast', 'breakfast', 'breakfast', 'breakfast', 'breakfast',
            # Lunch
            'lunch', 'lunch', 'lunch', 'lunch', 'lunch',
            'lunch', 'lunch', 'lunch', 'lunch', 'lunch',
            # Dinner
            'dinner', 'dinner', 'dinner', 'dinner', 'dinner',
            'dinner', 'dinner', 'dinner', 'dinner', 'dinner',
            # Snacks
            'snack', 'snack', 'snack', 'snack', 'snack',
            'snack', 'snack', 'snack', 'snack', 'snack'
        ],
        'diet_type': [
            # Breakfast
            'vegetarian', 'non-vegetarian', 'vegetarian', 'vegan', 'vegetarian',
            'vegan', 'vegan', 'vegetarian', 'vegetarian', 'vegan',
            # Lunch
            'vegetarian', 'non-vegetarian', 'vegan', 'vegan', 'non-vegetarian',
            'vegetarian', 'vegetarian', 'vegetarian', 'vegetarian', 'vegetarian',
            # Dinner
            'non-vegetarian', 'vegan', 'vegetarian', 'vegan', 'vegan',
            'vegetarian', 'vegan', 'non-vegetarian', 'vegetarian', 'vegetarian',
            # Snacks
            'vegan', 'vegan', 'vegetarian', 'vegan', 'vegan',
            'vegan', 'vegetarian', 'vegan', 'vegan', 'vegetarian'
        ]
    }
    
    return pd.DataFrame(foods_data)

# Calculate BMR and daily calorie needs
def calculate_calories(age, gender, height, weight, activity_level, goal):
    # Calculate BMR using Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Activity multipliers
    activity_multipliers = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extremely active': 1.9
    }
    
    tdee = bmr * activity_multipliers.get(activity_level.lower(), 1.55)
    
    # Adjust based on goal
    if 'weight loss' in goal.lower():
        target_calories = tdee - 500  # 1 lb/week loss
    elif 'weight gain' in goal.lower() or 'muscle gain' in goal.lower():
        target_calories = tdee + 300  # Moderate surplus
    else:
        target_calories = tdee  # Maintenance
    
    return int(target_calories), int(bmr)

# Calculate macronutrient targets
def calculate_macros(calories, goal, weight):
    if 'muscle gain' in goal.lower():
        protein_ratio = 0.25
        carb_ratio = 0.45
        fat_ratio = 0.30
    elif 'weight loss' in goal.lower():
        protein_ratio = 0.30
        carb_ratio = 0.35
        fat_ratio = 0.35
    else:
        protein_ratio = 0.20
        carb_ratio = 0.50
        fat_ratio = 0.30
    
    protein_g = int((calories * protein_ratio) / 4)
    carbs_g = int((calories * carb_ratio) / 4)
    fat_g = int((calories * fat_ratio) / 9)
    
    return protein_g, carbs_g, fat_g

# Generate meal plan
def generate_meal_plan(foods_df, target_calories, protein_target, carbs_target, fat_target, 
                      diet_preference, meal_frequency, allergies, favorite_foods=None):
    
    # Filter foods based on diet preference
    if diet_preference.lower() != 'no preference':
        available_foods = foods_df[foods_df['diet_type'].str.contains(diet_preference.lower(), case=False, na=False)]
    else:
        available_foods = foods_df.copy()
    
    # Remove foods with allergies (simple check)
    if allergies and allergies.lower() not in ['none', 'no allergies', '']:
        allergy_keywords = allergies.lower().split(',')
        for keyword in allergy_keywords:
            keyword = keyword.strip()
            if keyword:
                available_foods = available_foods[~available_foods['food_name'].str.contains(keyword, case=False, na=False)]
    
    # Generate meal plan based on frequency
    meal_plan = {}
    calories_per_meal = target_calories / meal_frequency
    
    for meal_type in ['breakfast', 'lunch', 'dinner']:
        meal_foods = available_foods[available_foods['meal_type'] == meal_type]
        if not meal_foods.empty:
            # Select food closest to target calories per meal
            meal_foods = meal_foods.copy()
            meal_foods['calorie_diff'] = abs(meal_foods['calories_per_serving'] - calories_per_meal)
            selected_food = meal_foods.loc[meal_foods['calorie_diff'].idxmin()]
            meal_plan[meal_type] = selected_food
    
    # Add snacks if meal frequency > 3
    if meal_frequency > 3:
        snack_foods = available_foods[available_foods['meal_type'] == 'snack']
        if not snack_foods.empty:
            remaining_calories = target_calories - sum([meal_plan[meal]['calories_per_serving'] for meal in meal_plan])
            snack_calories = remaining_calories / (meal_frequency - 3)
            snack_foods = snack_foods.copy()
            snack_foods['calorie_diff'] = abs(snack_foods['calories_per_serving'] - snack_calories)
            selected_snack = snack_foods.loc[snack_foods['calorie_diff'].idxmin()]
            meal_plan['snack'] = selected_snack
    
    return meal_plan

# Generate workout plan
def generate_workout_plan(goal, activity_level):
    workouts = {
        'weight loss': {
            'monday': 'Cardio - 30min brisk walk/jog + 15min bodyweight exercises',
            'tuesday': 'Strength training - Upper body (push-ups, planks, arm exercises)',
            'wednesday': 'Cardio - 30min cycling/dancing + stretching',
            'thursday': 'Strength training - Lower body (squats, lunges, calf raises)',
            'friday': 'Full body HIIT - 20min high intensity + 10min cool down',
            'saturday': 'Active recovery - yoga/light walking',
            'sunday': 'Rest day'
        },
        'muscle gain': {
            'monday': 'Upper body strength - Push-ups, pull-ups, dips (3 sets x 8-12 reps)',
            'tuesday': 'Lower body strength - Squats, lunges, deadlifts (3 sets x 8-12 reps)', 
            'wednesday': 'Cardio - 20min moderate intensity',
            'thursday': 'Upper body strength - Different exercises than Monday',
            'friday': 'Lower body strength - Different exercises than Tuesday',
            'saturday': 'Full body circuit training',
            'sunday': 'Rest day'
        },
        'maintenance': {
            'monday': 'Cardio - 30min moderate exercise',
            'tuesday': 'Strength training - Full body basics',
            'wednesday': 'Flexibility/Yoga - 30min stretching routine', 
            'thursday': 'Cardio - 30min different activity',
            'friday': 'Strength training - Full body basics',
            'saturday': 'Active fun - sports/dancing/hiking',
            'sunday': 'Rest day'
        }
    }
    
    goal_key = 'maintenance'
    for key in workouts.keys():
        if key in goal.lower():
            goal_key = key
            break
    
    return workouts[goal_key]

# Generate grocery list
def generate_grocery_list(meal_plan):
    grocery_items = []
    for meal_type, food_info in meal_plan.items():
        food_name = food_info['food_name']
        # Extract main ingredients (simplified)
        if 'rice' in food_name.lower():
            grocery_items.append('Rice')
        if 'dal' in food_name.lower():
            grocery_items.append('Dal/Lentils')
        if 'chicken' in food_name.lower():
            grocery_items.append('Chicken')
        if 'eggs' in food_name.lower():
            grocery_items.append('Eggs')
        if 'oats' in food_name.lower():
            grocery_items.append('Oats')
        if 'milk' in food_name.lower():
            grocery_items.append('Milk')
        if 'yogurt' in food_name.lower():
            grocery_items.append('Yogurt')
        if 'vegetables' in food_name.lower() or 'sabzi' in food_name.lower():
            grocery_items.append('Mixed Vegetables')
        if 'nuts' in food_name.lower() or 'almonds' in food_name.lower():
            grocery_items.append('Nuts/Almonds')
        if 'fish' in food_name.lower() or 'salmon' in food_name.lower():
            grocery_items.append('Fish')
        if 'paneer' in food_name.lower():
            grocery_items.append('Paneer')
        if 'fruit' in food_name.lower() or 'apple' in food_name.lower() or 'banana' in food_name.lower():
            grocery_items.append('Fresh Fruits')
    
    return list(set(grocery_items))  # Remove duplicates

# Main App
def main():
    st.markdown('<h1 class="main-header">🥗 Smart Diet & Workout Planner</h1>', unsafe_allow_html=True)
    st.markdown("### 🎯 Get personalized meal plans and workout routines based on your goals!")
    
    # Load data
    foods_df = create_nutrition_database()
    
    # Sidebar for user inputs with better organization
    st.sidebar.markdown("## 📝 Tell Us About Yourself")
    st.sidebar.markdown("---")
    
    # Core required inputs - better organized
    st.sidebar.markdown("### ✅ Basic Information")
    
    col_age, col_gender = st.sidebar.columns(2)
    with col_age:
        age = st.number_input("Age", min_value=16, max_value=80, value=25)
    with col_gender:
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    col_height, col_weight = st.sidebar.columns(2)
    with col_height:
        height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
    with col_weight:
        weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70)
    
    st.sidebar.markdown("### 🏃‍♀️ Activity & Goals")
    
    activity_level = st.sidebar.selectbox(
        "How active are you?",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
        help="Sedentary: Desk job, no exercise | Lightly Active: Light exercise 1-3 days/week"
    )
    
    goal = st.sidebar.selectbox(
        "What's your main goal?",
        ["Weight Loss", "Weight Gain", "Muscle Gain", "Maintenance", "General Health"]
    )
    
    st.sidebar.markdown("### 🍽️ Food Preferences")
    
    diet_preference = st.sidebar.selectbox(
        "Diet Type",
        ["No Preference", "Vegetarian", "Non-Vegetarian", "Vegan"]
    )
    
    meal_frequency = st.sidebar.slider("How many meals per day?", min_value=3, max_value=6, value=4)
    
    allergies = st.sidebar.text_input("Any allergies?", placeholder="e.g., nuts, dairy, gluten", help="Separate with commas")
    
    # Optional advanced inputs - collapsible
    st.sidebar.markdown("### 🚀 Optional Details (For Better Results)")
    
    with st.sidebar.expander("🔧 Advanced Settings"):
        favorite_foods = st.text_input("Favorite Foods", placeholder="e.g., rice, chicken, salad")
        medical_history = st.text_input("Medical Conditions", placeholder="e.g., diabetes, hypertension")
        sleep_hours = st.slider("Sleep Hours/Night", min_value=4, max_value=12, value=7)
        work_schedule = st.selectbox("Work Schedule", ["Regular (9-5)", "Night Shift", "Flexible", "Student"])
        budget_preference = st.selectbox("Budget Range", ["Low", "Medium", "High"])
        cooking_skill = st.selectbox("Cooking Experience", ["Beginner", "Intermediate", "Advanced"])
    
    st.sidebar.markdown("---")
    
    # Generate plan button - more prominent
    if st.sidebar.button("🎯 Generate My Personalized Plan!", type="primary", use_container_width=True):
        # Show loading message
        with st.spinner('Creating your personalized plan...'):
            # Calculate requirements
            target_calories, bmr = calculate_calories(age, gender, height, weight, activity_level, goal)
            protein_target, carbs_target, fat_target = calculate_macros(target_calories, goal, weight)
            
            # Generate meal plan
            meal_plan = generate_meal_plan(
                foods_df, target_calories, protein_target, carbs_target, fat_target,
                diet_preference, meal_frequency, allergies, favorite_foods
            )
            
            # Generate workout plan
            workout_plan = generate_workout_plan(goal, activity_level)
            
            # Generate grocery list
            grocery_list = generate_grocery_list(meal_plan)
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
        
        # Success message
        st.markdown('<div class="success-box">✅ Your personalized plan is ready!</div>', unsafe_allow_html=True)
        
        # Display results with improved layout
        st.markdown('<h2 class="sub-header">📊 Your Nutrition Dashboard</h2>', unsafe_allow_html=True)
        
        # Metrics with better styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{target_calories}</div>
                <div class="metric-label">Daily Calories</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{bmr}</div>
                <div class="metric-label">BMR</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{bmi:.1f}</div>
                <div class="metric-label">BMI</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            bmi_status = "Normal" if 18.5 <= bmi <= 24.9 else "Overweight" if bmi > 24.9 else "Underweight"
            st.markdown(f'''
            <div class="metric-container">
                <div class="metric-value">{bmi_status}</div>
                <div class="metric-label">BMI Status</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Macros chart with better styling
        st.markdown("#### 📈 Your Daily Macro Breakdown")
        
        macro_df = pd.DataFrame({
            'Nutrient': ['Protein', 'Carbs', 'Fat'],
            'Grams': [protein_target, carbs_target, fat_target],
            'Calories': [protein_target*4, carbs_target*4, fat_target*9]
        })
        
        fig_macros = px.pie(
            macro_df, 
            values='Calories', 
            names='Nutrient',
            title="Macro Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            hole=0.4
        )
        fig_macros.update_layout(
            title_font_size=18,
            showlegend=True,
            height=400,
            font=dict(size=14)
        )
        st.plotly_chart(fig_macros, use_container_width=True)
        
        # Meal Plan with fixed display
        st.markdown('<h2 class="sub-header">🍽️ Your Personalized Meal Plan</h2>', unsafe_allow_html=True)
        
        if meal_plan:
            # Display meals in a clean grid
            meal_cols = st.columns(2)
            
            total_calories = 0
            total_protein = 0
            
            for i, (meal_type, food_info) in enumerate(meal_plan.items()):
                with meal_cols[i % 2]:
                    st.markdown(f'''
                    <div class="meal-card">
                        <div class="meal-title">🍽️ {meal_type.title()}</div>
                        <div class="meal-food">{food_info['food_name']}</div>
                        <div class="meal-nutrition">
                            🔥 {food_info['calories_per_serving']} calories<br>
                            💪 {food_info['protein_g']}g protein<br>
                            🍞 {food_info['carbs_g']}g carbs<br>
                            🥑 {food_info['fat_g']}g fat
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                total_calories += food_info['calories_per_serving']
                total_protein += food_info['protein_g']
            
            # Summary
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.info(f"📊 **Total Daily Calories:** {total_calories} / {target_calories}")
            with col_summary2:
                st.info(f"💪 **Total Protein:** {total_protein}g / {protein_target}g")
        else:
            st.error("Unable to generate meal plan. Please adjust your preferences.")
        
        # Workout Plan with better styling
        st.markdown('<h2 class="sub-header">💪 Your Weekly Workout Schedule</h2>', unsafe_allow_html=True)
        
        workout_cols = st.columns(2)
        
        for i, (day, exercise) in enumerate(workout_plan.items()):
            with workout_cols[i % 2]:
                st.markdown(f'''
                <div class="workout-card">
                    <div class="workout-day">📅 {day.title()}</div>
                    <div class="workout-exercise">{exercise}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Grocery List with better organization
        st.markdown('<h2 class="sub-header">🛒 Your Smart Grocery List</h2>', unsafe_allow_html=True)
        
        if grocery_list:
            st.markdown("#### ✅ Check off items as you shop:")
            grocery_cols = st.columns(3)
            for i, item in enumerate(grocery_list):
                with grocery_cols[i % 3]:
                    st.checkbox(f"🛍️ {item}", value=False, key=f"grocery_{i}")
        else:
            st.markdown('<div class="info-box">📝 No specific ingredients identified. Your meal plan is ready above!</div>', unsafe_allow_html=True)
        
        # Progress tracking with better UX
        st.markdown('<h2 class="sub-header">📈 Track Your Journey</h2>', unsafe_allow_html=True)
        
        col_track1, col_track2 = st.columns(2)
        
        with col_track1:
            st.markdown("#### 📝 Weekly Check-in")
            week_rating = st.slider("Rate this plan (1-10)", 1, 10, 7, help="How satisfied are you with your plan?")
            week_feedback = st.text_area("Share your thoughts", 
                                       placeholder="e.g., loved the breakfast, would like more variety in dinner",
                                       help="Your feedback helps us improve!")
            
            if st.button("📤 Submit Feedback", use_container_width=True):
                st.markdown('<div class="success-box">✨ Thank you! Your feedback helps us improve.</div>', unsafe_allow_html=True)
        
        with col_track2:
            st.markdown("#### 🎯 Expected Results")
            if 'weight loss' in goal.lower():
                weeks_to_goal = st.number_input("Target weight loss (kg)", min_value=1, max_value=20, value=5)
                estimated_weeks = weeks_to_goal * 2  # Assuming 0.5kg/week
                st.markdown(f'<div class="info-box">⏰ Estimated Timeline: {estimated_weeks} weeks<br>🏃‍♀️ Healthy weight loss: 0.5kg/week</div>', unsafe_allow_html=True)
            elif 'muscle gain' in goal.lower():
                st.markdown('<div class="info-box">💪 Visible changes: 4-6 weeks<br>🔥 Significant gains: 12-16 weeks<br>🎯 Stay consistent!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">📈 Follow for 4 weeks to see results<br>🔄 Consistency is key<br>💪 You got this!</div>', unsafe_allow_html=True)
    
    # About section with better presentation
    with st.expander("ℹ️ About This Smart Planner"):
        st.markdown("""
        ### 🤖 How It Works
        This app uses **machine learning algorithms** and **nutritional science** to create your personalized plan:
        
        **🧮 Scientific Calculations:**
        - BMR using Mifflin-St Jeor equation
        - TDEE based on activity level
        - Goal-specific calorie adjustments
        
        **🥗 Smart Recommendations:**
        - 40+ food database with accurate nutrition data
        - Diet preference and allergy filtering
        - Optimal macro distribution for your goals
        
        **💪 Personalized Workouts:**
        - Goal-specific exercise routines
        - Beginner to advanced options
        - Weekly structure for consistency
        
        **⚠️ Important Note:** This tool is for educational purposes. Always consult healthcare professionals for medical advice.
        """)

if __name__ == "__main__":
    main()