from unicodedata import name

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)


# SETTINGS FOR THE DATABASE
SEED = 38 # This seed is repeatable
CLASSES = ["unacc", "acc", "good", "vgood"]
RARE = ["good", "vgood"] # MINORITY CLASSES
cols=["buying", "maintenance", "doors", "persons", "lug_boot", "safety", "class"]

# LOADING THE DATASET
df= pd.read_csv("car_evaluation.csv", header=None, names=cols)
print("Rows and columns of the dataset: ", df.shape)
print(df["class"].value_counts())

# 70/30 split of the dataset into training and testing sets
train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["class"]) # stratify=df["class"] makes sure that the class mix is in both sets so the rare classes appear in training and testing
assert len(train)+len(test)==len(df) # Makes sure that no rows are duplicated or missing while splitting
assert not set(train.index) & set(test.index) # Uses to make sure that the training and testing sets are not overlapping 

print("\nTrain size:", len(train), "Test size:", len(test))
print("\nClass share in train:\n", train["class"].value_counts(normalize=True).round(3))
print("\nClass share in test:\n", test["class"].value_counts(normalize=True).round(3))

actual=list(test["class"])
results={}


# EVALUATION WITH SCIKIT-LEARN METRICS FUNCTIONS | ChatGPT was used to help writing this evaluation function
def evaluate(name, pred):
    """Score a list of predictions (ONE PER TEST CAR) and print the evaluation"""
    accuracy = accuracy_score(actual, pred)
    print(f"\n=={name}==\nAccuracy: {accuracy:.3f}")
    
    print(classification_report(actual, pred, labels=CLASSES, digits=3, zero_division=0))

    # Macro-F1 = average of the four F1 scores, so rare classes count as much as common ones
    macro_f1 = f1_score(actual, pred, labels=CLASSES, average="macro", zero_division=0)

    # Macro-precision = average precision over the four classes
    macro_precision = precision_score(actual, pred, labels=CLASSES, average="macro", zero_division=0)

    # Average recall on the rare classes: how many good/vgood cars did the system find?
    rare_recall = recall_score(actual, pred, labels=RARE, average="macro", zero_division=0)

    print(f"Macro-precision: {macro_precision:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}\nAverage recall on rare classes (good, vgood): {rare_recall:.3f}")

    # Confusion matrix: rows = actual class, columns = predicted class
    matrix = confusion_matrix(actual, pred, labels=CLASSES)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=CLASSES, columns=CLASSES))

    # Safety critical error: truly unacceptable cars (first row) rated as anything better
    unsafe = int(matrix[0, 1:].sum())
    print("Unacceptable cars wrongly approved:", unsafe, "of", int(matrix[0].sum()))

    results[name] = (accuracy, macro_precision, macro_f1, rare_recall, unsafe)
    return accuracy, macro_precision, macro_f1, rare_recall, unsafe


def print_summary():
    """All systems evaluated so far, side by side."""
    print("\n== SUMMARY ==")
    print(f"{'system':30}{'accuracy':>9}{'macro-P':>9}{'macro-F1':>9}{'rare recall':>12}{'unsafe OK':>10}")
    for name, (acc, prec, f1, rare, unsafe) in results.items():
        print(f"{name:30}{acc:9.3f}{prec:9.3f}{f1:9.3f}{rare:12.3f}{unsafe:10}")



# BASLINE FOR ALWAYS PREDICTING THE MOST COMMON CLASS IN THE TRAINING DATA ("unacc")
baseline=[train["class"].mode()[0]]*len(test)
evaluate("Majority baseline", baseline)

# SYSTEM 1: HANDWRITTEN RULES (IF-ELSE)
