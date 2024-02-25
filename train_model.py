from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data - replace this with your actual data
point_type_mappings = {
    "Zone Air Temperature Sensor": ["ZN-T", "ZoneTempSensor", "ZnTemp"],
    "Air Flow Sensor": ["SA-F", "AirFlowSens", "FlowSensor"],
    "Damper Position Sensor": ["DPR-O", "DamperPos", "DmpPos"],
    "Supply Air Flow Setpoint": ["SAFLOW-SP", "SupAirFlowSP", "SAF_Setpoint"],
    "Occupancy Sensor": ["OCC-S", "OCC-C", "OccSensor", "OccupancyDet"],
    "Supply Air Temperature Setpoint": ["DAT-SP", "DisAirTempSP", "SAT_Setpoint"],
    "Zone Temperature Setpoint": ["ZNT-SP", "ZoneTempSP", "ZnTempSet"],
    "Heating Coil Valve Command": ["HTG-O", "HeatValveCmd", "HtgCoilCmd"],
    "Cooling Coil Valve Command": ["CLG-O", "CoolValveCmd", "ClgCoilCmd"],
    "Air Quality Sensor": ["ZNT-Q", "AirQualSens", "AQSensor"],
    # ... more categories as needed ...
}

# Convert the dictionary into a list of tuples
training_data = []
for category, points in point_type_mappings.items():
    for point in points:
        training_data.append((point, category))

# Split data into features and labels
X_train, y_train = zip(*training_data)

# Create the model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Example prediction
test_point = ["SomeVendorPointName"]
predicted_brick_name = model.predict(test_point)
predicted_probabilities = model.predict_proba(test_point)

print(f"Predicted Brick Schema Category: {predicted_brick_name[0]}")

# Print probabilities for each class
print("\nProbabilities for each Brick Schema Category:")
for category, probability in zip(model.classes_, predicted_probabilities[0]):
    print(f"{category}: {probability:.4f}")
