import os
import pandas as pd
import numpy as np
import cv2
import pywt
import pickle

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Path where our data is located
train_path = "ASL_Dataset/Train/"
test_path = "ASL_Dataset/Test/"

output_folder_Train = "Segmented_Train"
os.makedirs(output_folder_Train, exist_ok=True)

output_folder_Test = "Segmented_Test"
os.makedirs(output_folder_Test, exist_ok=True)

# Dictionary to save our classes
categories = {0: "A",
              1: "B",
              2: "C",
              3: "D",
              4: "E",
              5: "F",
              6: "G",
              7: "H",
              8: "I",
              9: "G",
              10: "K",
              11: "L",
              12: "M",
              13: "N",
              14: "O",
              15: "P",
              16: "Q",
              17: "R",
              18: "S",
              19: "T",
              20: "U",
              21: "V",
              22: "W",
              23: "X",
              24: "Y",
              25: "Z",
              26: "nothing",
              27: "space",
              }


def add_class_name_prefix(df, col_name):
    var = df[col_name]
    return df


# list conatining all the filenames in the dataset
filenames_list = []
# list to store the corresponding category, as A,B,C
categories_list = []

for category in categories:
    filenames = os.listdir(train_path + categories[category])
    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df_Train = pd.DataFrame({"filename": filenames_list, "category": categories_list})
df_Train = add_class_name_prefix(df_Train, "filename")

# list conatining all the filenames in the dataset
filenames_list = []

# list to store the corresponding category, as A,B,C
categories_list = []

for category in categories:
    filenames = os.listdir(test_path + categories[category])
    filenames_list = filenames_list + filenames
    categories_list = categories_list + [category] * len(filenames)

df_Test = pd.DataFrame({"filename": filenames_list, "category": categories_list})
df_Test = add_class_name_prefix(df_Test, "filename")

# Shuffle the dataframe
df_Train= df_Train.sample(frac=1).reset_index(drop=True)
df_Test= df_Test.sample(frac=1).reset_index(drop=True)

def segment(img):

    # Convert Image to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's Segmentation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Finding contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Finding the Maximum contour
    c = max(contours, key=cv2.contourArea)

    # Getting its coordinates
    x, y, w, h = cv2.boundingRect(c)

    # Creating the box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # Cropping according to the box
    cropped_image = thresh[y:y + h, x:x + w]

    # Dilation filter size
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed

    # Dilation
    dilated_image = cv2.dilate(cropped_image, kernel, iterations=1)

    # Resizing the image
    smaller_image = cv2.resize(dilated_image, (128,128))

    return smaller_image

def feature_extraction(dilated_image):

    # DWT Feature Extraction as Paper
    coeffs = pywt.wavedec2(dilated_image, wavelet='db1', level=4)
    features_list = []
    features_list.append(np.sum(np.square(coeffs[0])))

    for j in range(1, min(len(coeffs), 3)):
        subband = coeffs[j]
        if isinstance(subband, tuple):
            subband = subband[0]
        features_list.append(np.sum(np.square(subband)))

    # Binary Features Extraction as paper using Lower Binary Pattern Method
    lbp = feature.local_binary_pattern(dilated_image, P=8, R=1, method="uniform")

    # Calculate LBP histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0,11))

    # Append histogram values to the features_list
    features_list.extend(hist)


    return features_list

X_Train = []
Y_Train = []
for i in range(len(df_Train)):
    # Image path
    path = ""
    path = os.path.join(train_path, categories[df_Train.category[i]], df_Train.filename[i])

    img = plt.imread(path)

    # Segment Each image
    segmented_imgage = segment(img)

    #Save it in folder
    subfolder = os.path.join(output_folder_Train, categories[df_Train.category[i]])
    os.makedirs(subfolder, exist_ok=True)

    #Save the segmented image in the subfolder
    output_path = os.path.join(subfolder, f"{df_Train.filename[i]}")
    cv2.imwrite(output_path, segmented_imgage)

    # Extract Features of Each image
    features = feature_extraction(segmented_imgage)

    X_Train.append(features)
    Y_Train.append(df_Train.category[i])

    print(str(i)+ "/" + str(len(df_Train)))

X_Test = []
Y_Test = []
for i in range(len(df_Test)):
    # Image path
    path = ""
    path = os.path.join(test_path, categories[df_Test.category[i]], df_Test.filename[i])

    img = plt.imread(path)
    segmented_imgage = segment(img)

    # Save it in folder
    subfolder = os.path.join(output_folder_Test, categories[df_Test.category[i]])
    os.makedirs(subfolder, exist_ok=True)

    # Save the segmented image in the subfolder
    output_path = os.path.join(subfolder, f"{df_Test.filename[i]}")
    cv2.imwrite(output_path, segmented_imgage)

    # Extract Features of Each image
    features = feature_extraction(segmented_imgage)

    X_Test.append(features)
    Y_Test.append(df_Test.category[i])

    print(str(i)+ "/" + str(len(df_Test)))

# Add feature names for each feature and then save the features in a csv file
feature_columns = [f'feature_{i+1}' for i in range(len(X_Train[0]))]
train_data = pd.DataFrame(data=np.array(X_Train), columns=feature_columns)

# Add the 'label' column
train_data['label'] = Y_Train

# Save to CSV
train_data.to_csv('train_data.csv', index=False)

# Add feature names for each feature and then save the features in a csv file
test_feature_columns = [f'feature_{i+1}' for i in range(len(X_Test[0]))]
test_data = pd.DataFrame(data=np.array(X_Test), columns=test_feature_columns)

# Add the 'label' column
test_data['label'] = Y_Test

# Save to CSV
test_data.to_csv('test_data.csv', index=False)


# SVM Model
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Extract features and labels
X_Train = train_data.drop('label', axis=1)
Y_Train = train_data['label']

X_Test = test_data.drop('label', axis=1)
Y_Test = test_data['label']

# Train the classifier

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_Train)
X_test = scaler.transform(X_Test)

# Create an SVM model
svm_model = SVC(kernel='poly', C=1.0, random_state=42)

# Train the model
svm_model.fit(X_train, Y_Train)
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_Test, y_pred)
report = classification_report(Y_Test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

with open('SVM.pkl', 'wb') as file:
    pickle.dump(svm_model, file)